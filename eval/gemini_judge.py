"""Official Gemini REST adapter; explicit purpose/field and free-quota gates.

No default key loading, no paid fallback, no hidden retries or production writes.
Caller supplies a private append-only evaluation ledger and read-only shared usage.
"""
import json
import time
import urllib.error
import urllib.request
from eval import semantic_judge as sj
from eval import judge_span_ids as indexed

REQUIRED_FIELDS = ('question', 'gold_answerable', 'reference_answer', 'key_facts',
                   'reference_evidence', 'student_evidence', 'model_answer', 'citations')
PURPOSE = 'motor_semantic_evaluation'
BATCH_WRAPPER=indexed.WRAPPER


def batch_schema(tasks):
    return indexed.schema(tasks)


def batch_bound(tasks):
    return len(indexed.prompt(tasks).encode())+len(json.dumps(batch_schema(tasks)).encode())+batch_output_budget(tasks)


def batch_output_budget(tasks):
    return min(32768,4096+3072*len(tasks))


class QuotaError(RuntimeError):
    def __init__(self, message, retry_after=None):
        super().__init__(message)
        self.retry_after=retry_after


class InvalidOutput(ValueError):
    def __init__(self, message, raw, usage):
        super().__init__(message)
        self.raw, self.usage = raw, usage


class ProviderError(RuntimeError):
    def __init__(self,code,details):
        super().__init__(f'Gemini HTTP {code}; response body retained privately')
        self.details=details


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise PermissionError('Provider redirects forbidden')


def research_events(paths):
    """Copied journals share timestamps: count each reservation exactly once."""
    rows=[]
    for path in paths:
        if path.exists():rows.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    actual={e['request_time']:e.get('quota_actual_tokens',e['actual_tokens']) for e in rows if 'actual_tokens' in e}
    reserved={}
    for e in rows:
        if 'reserved_tokens' in e:reserved[e['time']]=max(reserved.get(e['time'],0),e['reserved_tokens'])
    return [(t,actual.get(t,n)) for t,n in reserved.items()]


class GeminiJudge:
    def __init__(self, key, model, permission, free_receipt, ledger, shared_usage,
                 catalogue, fictional=False, clock=time.time, reserve_requests=0, thinking_level='high',related_ledgers=()):
        self.key, self.model, self.permission = key, model.removeprefix('models/'), permission
        self.free, self.ledger, self.shared = free_receipt, ledger, shared_usage
        self.fictional, self.clock = fictional, clock
        self.reserve_requests=reserve_requests
        self.thinking_level=thinking_level
        self.related_ledgers=list(related_ledgers)
        if type(reserve_requests) is not int or reserve_requests<0 or reserve_requests>=self.free.get('rpd',0):
            raise ValueError('Invalid production quota reserve')
        name='models/'+self.model
        if not any(m['name']==name and 'generateContent' in m.get('supportedGenerationMethods',m.get('methods',[])) for m in catalogue):
            raise ValueError('Model is not in authenticated generateContent catalogue')
        if self.free.get('model') != self.model:
            raise PermissionError('Free quota verified for a different model')
        if self.clock()-self.free.get('verified_at_epoch',0)>48*3600:
            raise PermissionError('Free quota verification is stale')

    def _authorise(self, task):
        if self.fictional:
            # An arbitrary caller flag cannot declassify real data.
            from eval.semantic_fixtures import fixtures
            if task not in [t for t,_ in fixtures()]:
                raise PermissionError('Fictional mode accepts fixed public fixtures only')
        elif not sj.permission_allows(self.permission, 'gemini', REQUIRED_FIELDS, PURPOSE):
            raise PermissionError('No permission for full semantic evaluation payload')

    def _request(self, prompt, schema, max_tokens, identity, seed):
        reserve=len(prompt.encode('utf-8'))+len(json.dumps(schema).encode())+max_tokens
        now=self.clock(); usage=json.loads(self.shared.read_text())
        events=[(e['time'],e['quota_tokens']) for e in usage['entries']]+research_events([self.ledger]+self.related_ledgers)
        quota={**self.free,'rpd':self.free['rpd']-self.reserve_requests,
               'rpm':max(1,self.free['rpm']-1),'tpm':max(1,self.free['tpm']-50000)}
        if len([t for t,n in events if now-86400<t<=now])>=quota['rpd']:
            raise QuotaError('Daily free evaluation allowance exhausted; production reserve retained')
        if reserve>quota['tpm']:
            raise ValueError('Batch exceeds conservative token bound; split without truncation')
        if usage.get('blocked_until',0)>now or not sj.quota_available(quota,events,now,reserve):
            recent=[t for t,n in events if now-60<t<=now]
            wait=max(1,min(recent)+61-now) if recent else None
            raise QuotaError('Shared minute free quota exhausted; no call made',wait)
        # Reserve BEFORE transport; failures consume the conservative reservation.
        with self.ledger.open('a') as f:
            f.write(json.dumps({'time':now,'reserved_tokens':reserve,'evaluation_id':identity,'seed':seed})+'\n')
        request=urllib.request.Request('https://generativelanguage.googleapis.com/v1beta/models/'+self.model+':generateContent',
            data=json.dumps({'contents':[{'role':'user','parts':[{'text':prompt}]}],
                'generationConfig':{'temperature':.2,'seed':seed,'maxOutputTokens':max_tokens,'responseMimeType':'application/json',
                    'thinkingConfig':{'thinkingLevel':self.thinking_level},
                    'responseJsonSchema':schema}}).encode(),
            headers={'Content-Type':'application/json','x-goog-api-key':self.key},method='POST')
        start=time.monotonic()
        try:
            with urllib.request.build_opener(NoRedirect()).open(request,timeout=180) as response:
                data=json.load(response)
        except urllib.error.HTTPError as e:
            if e.code==429:raise QuotaError('Gemini HTTP 429; stopped without retry or paid fallback') from None
            try:details=json.load(e)
            except (ValueError,TypeError):details={'status':'unparseable_error_body'}
            raise ProviderError(e.code,details) from None
        except (urllib.error.URLError,TimeoutError,OSError):
            raise RuntimeError('Gemini transport failed; credentials and response body suppressed') from None
        usage={'provider':'gemini','model':data.get('modelVersion',self.model),
                    'usage':data.get('usageMetadata',{}),'estimated_cost_usd':0,
                    'thinking_level':self.thinking_level,
                    'latency_seconds':time.monotonic()-start,'free_quota_verified':True}
        total=usage['usage'].get('totalTokenCount')
        if type(total) is int:
            with self.ledger.open('a') as f:f.write(json.dumps({'request_time':now,'actual_tokens':total,
                'quota_actual_tokens':usage['usage'].get('promptTokenCount',total)})+'\n')
        candidates=data.get('candidates',[])
        usage['finish_reason']=candidates[0].get('finishReason') if len(candidates)==1 else None
        raw=''.join(p.get('text','') for c in candidates for p in c.get('content',{}).get('parts',[]))
        if len(candidates)!=1 or candidates[0].get('finishReason')!='STOP':
            raise InvalidOutput('Missing, blocked or truncated Gemini completion',raw,usage)
        return raw,usage

    def call(self, task, seed):
        self._authorise(task)
        raw,usage=self._request(sj.PROMPT+'\nTASK:\n'+json.dumps(task,ensure_ascii=False),sj.response_schema(task),1800,task['evaluation_id'],seed)
        try:sj.validate(raw,task)
        except (ValueError,KeyError,TypeError) as e:raise InvalidOutput(str(e),raw,usage) from None
        return raw,usage

    def call_many(self, tasks, seed):
        if not tasks or len({t['evaluation_id'] for t in tasks})!=len(tasks):raise ValueError('Empty/duplicate batch')
        for task in tasks:self._authorise(task)
        schema=batch_schema(tasks)
        raw,usage=self._request(indexed.prompt(tasks),schema,batch_output_budget(tasks),[t['evaluation_id'] for t in tasks],seed)
        try:
            root=json.loads(raw,object_pairs_hook=sj._unique_pairs)
            if not isinstance(root,dict) or set(root)!={'evaluations'} or not isinstance(root['evaluations'],list):raise ValueError('Invalid batch container')
            values={v['evaluation_id']:v for v in root['evaluations']}
            if len(values)!=len(root['evaluations']) or set(values)!={t['evaluation_id'] for t in tasks}:raise ValueError('Batch ID population mismatch')
        except (ValueError,TypeError,KeyError) as e:raise InvalidOutput(str(e),raw,usage) from None
        results=[]
        for t in tasks:
            text=json.dumps(values[t['evaluation_id']],ensure_ascii=False)
            try:results.append({'valid':True,'raw':text,'judgement':indexed.normalise(values[t['evaluation_id']],t)})
            except (ValueError,KeyError,TypeError) as e:results.append({'valid':False,'raw':text,'error':str(e)})
        return results,usage
