"""Official Gemini REST adapter; explicit purpose/field and free-quota gates.

No default key loading, no paid fallback, no hidden retries or production writes.
Caller supplies a private append-only evaluation ledger and read-only shared usage.
"""
import json
import time
import urllib.error
import urllib.request
from eval import semantic_judge as sj

REQUIRED_FIELDS = ('question', 'gold_answerable', 'reference_answer', 'key_facts',
                   'reference_evidence', 'student_evidence', 'model_answer', 'citations')
PURPOSE = 'motor_semantic_evaluation'


class QuotaError(RuntimeError):
    pass


class InvalidOutput(ValueError):
    def __init__(self, message, raw, usage):
        super().__init__(message)
        self.raw, self.usage = raw, usage


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise PermissionError('Provider redirects forbidden')


class GeminiJudge:
    def __init__(self, key, model, permission, free_receipt, ledger, shared_usage,
                 catalogue, fictional=False, clock=time.time):
        self.key, self.model, self.permission = key, model.removeprefix('models/'), permission
        self.free, self.ledger, self.shared = free_receipt, ledger, shared_usage
        self.fictional, self.clock = fictional, clock
        name='models/'+self.model
        if not any(m['name']==name and 'generateContent' in m.get('supportedGenerationMethods',m.get('methods',[])) for m in catalogue):
            raise ValueError('Model is not in authenticated generateContent catalogue')
        if self.free.get('model') != self.model:
            raise PermissionError('Free quota verified for a different model')
        if self.clock()-self.free.get('verified_at_epoch',0)>48*3600:
            raise PermissionError('Free quota verification is stale')

    def call(self, task, seed):
        if self.fictional:
            # An arbitrary caller flag cannot declassify real data.
            from eval.semantic_fixtures import fixtures
            if task not in [t for t,_ in fixtures()]:
                raise PermissionError('Fictional mode accepts fixed public fixtures only')
        elif not sj.permission_allows(self.permission, 'gemini', REQUIRED_FIELDS, PURPOSE):
            raise PermissionError('No permission for full semantic evaluation payload')
        prompt=sj.PROMPT+'\nTASK:\n'+json.dumps(task,ensure_ascii=False)
        reserve=len(prompt.encode('utf-8'))+1800  # conservative byte/token bound
        now=self.clock(); usage=json.loads(self.shared.read_text())
        own=[json.loads(l) for l in self.ledger.read_text().splitlines() if l.strip()] if self.ledger.exists() else []
        events=[(e['time'],e['quota_tokens']) for e in usage['entries']]+[(e['time'],e['reserved_tokens']) for e in own]
        if usage.get('blocked_until',0)>now or not sj.quota_available(self.free,events,now,reserve):
            raise QuotaError('Shared free quota exhausted; no call made')
        # Reserve BEFORE transport; failures consume the conservative reservation.
        with self.ledger.open('a') as f:
            f.write(json.dumps({'time':now,'reserved_tokens':reserve,'evaluation_id':task['evaluation_id'],'seed':seed})+'\n')
        request=urllib.request.Request('https://generativelanguage.googleapis.com/v1beta/models/'+self.model+':generateContent',
            data=json.dumps({'contents':[{'role':'user','parts':[{'text':prompt}]}],
                'generationConfig':{'temperature':.2,'seed':seed,'maxOutputTokens':1800,'responseMimeType':'application/json'}}).encode(),
            headers={'Content-Type':'application/json','x-goog-api-key':self.key},method='POST')
        start=time.monotonic()
        try:
            with urllib.request.build_opener(NoRedirect()).open(request,timeout=90) as response:
                data=json.load(response)
        except urllib.error.HTTPError as e:
            if e.code==429:raise QuotaError('Gemini HTTP 429; stopped without retry or paid fallback') from None
            raise RuntimeError(f'Gemini HTTP {e.code}; response body suppressed') from None
        candidates=data.get('candidates',[])
        if len(candidates)!=1 or candidates[0].get('finishReason')!='STOP':
            raise ValueError('Missing, blocked or truncated Gemini completion')
        raw=''.join(p.get('text','') for p in candidates[0]['content']['parts'])
        usage={'provider':'gemini','model':data.get('modelVersion',self.model),
                    'usage':data.get('usageMetadata',{}),'estimated_cost_usd':0,
                    'latency_seconds':time.monotonic()-start,'free_quota_verified':True}
        try:sj.validate(raw,task)
        except (ValueError,KeyError,TypeError) as e:raise InvalidOutput(str(e),raw,usage) from None
        return raw,usage
