"""Lossless judge transport: source IDs resolve mechanically to verbatim spans."""
import json
import re
from eval import semantic_judge as sj

TRANSPORT='indexed-pointwise-v3'
WRAPPER='''\nINDEXED BATCH TRANSPORT v3. Score each task independently; never compare
answers between tasks. model_answer and reference_evidence are LOSSLESS ordered
text segments. The original text is their concatenation. Use answer_span_ids
(one ID or null per key_fact) and reference_span_ids (a list of IDs) instead of
quotations. An ID must justify the corresponding decision; do not infer support
merely because an ID exists. For absent facts score 0 and use null. Consecutive
segments may be referenced as A0-A2 or R0S1-R0S3 (same source chunk only).
Local validation resolves these IDs to exact original text. All other rubric
fields and grading anchors are unchanged. Return {"evaluations":[one canonical
judgement per evaluation_id]}; every input ID must occur exactly once.
TASKS:\n'''


def segments(text, prefix):
    ends=[m.end() for m in re.finditer(r'(?<=[.!?])\s+|\n+',text)]
    result=[];start=0
    for end in ends+[len(text)]:
        if end>start:
            result.append({'id':prefix+str(len(result)),'text':text[start:end]})
            start=end
    assert ''.join(s['text'] for s in result)==text
    return result


def view(task):
    value={k:v for k,v in task.items() if k not in ('model_answer','reference_evidence')}
    value['model_answer']=segments(task['model_answer'],'A')
    value['reference_evidence']=[{**{k:v for k,v in c.items() if k!='text'},
                                 'segments':segments(c['text'],f'R{i}S')} for i,c in enumerate(task['reference_evidence'])]
    return value


def prompt(tasks):
    text=sj.PROMPT
    text=text.replace('evidence_spans must be verbatim substrings of model_answer (empty\nonly for absent facts)',
                      'answer_span_ids must identify exact provided model_answer segments (null for absent facts)')
    text=text.replace('reference_spans are verbatim excerpts from reference_evidence;',
                      'reference_span_ids identify exact segments of reference_evidence;')
    text=text.replace('"evidence_spans"','"answer_span_ids"').replace('"reference_spans"','"reference_span_ids"')
    return text+WRAPPER+json.dumps([view(t) for t in tasks],ensure_ascii=False)


def schema(tasks):
    item=sj.response_schema(tasks[0]);props=item['properties']
    props.pop('evidence_spans');props.pop('reference_spans')
    props['answer_span_ids']={'type':'array','items':{'type':['string','null']},
        'description':'Exactly one supplied answer segment ID (or consecutive range) per key_fact; null only for absent facts.'}
    props['reference_span_ids']={'type':'array','items':{'type':'string'},
        'description':'Supplied reference segment IDs/ranges genuinely supporting the decision; empty for absent support or refusal.'}
    props['fact_scores'].pop('minItems',None);props['fact_scores'].pop('maxItems',None)
    if all(not t['key_facts'] for t in tasks):
        # A zero-fact singleton retry has no legitimate fact spans. This trivial
        # bound avoids the complex nested finite grammars of mixed real batches.
        for key in ('fact_scores','answer_span_ids'):props[key].update(minItems=0,maxItems=0)
    props['evaluation_id']['enum']=[t['evaluation_id'] for t in tasks]
    item['required']=sorted(props)
    return {'type':'object','properties':{'evaluations':{'type':'array','items':item}},
            'required':['evaluations'],'additionalProperties':False}


def resolve(code, sources):
    if code is None or code=='':return ''
    if not isinstance(code,str):raise ValueError('Invalid span ID type')
    for source in sources:
        ids=[s['id'] for s in source]
        if code in ids:return source[ids.index(code)]['text']
        if '-' in code:
            first,last=code.split('-',1)
            if first in ids and last in ids:
                a,b=ids.index(first),ids.index(last)
                if b<a:raise ValueError('Reversed span range')
                return ''.join(s['text'] for s in source[a:b+1])
    raise ValueError('Invented or cross-source span ID')


def normalise(value, task):
    expected=(sj.FIELDS-{'evidence_spans','reference_spans'})|{'answer_span_ids','reference_span_ids'}
    if set(value)!=expected:raise ValueError('Unexpected indexed schema')
    if not isinstance(value['answer_span_ids'],list) or not isinstance(value['reference_span_ids'],list):
        raise ValueError('Invalid indexed span arrays')
    answer=[segments(task['model_answer'],'A')]
    reference=[segments(c['text'],f'R{i}S') for i,c in enumerate(task['reference_evidence'])]
    result={k:v for k,v in value.items() if k not in ('answer_span_ids','reference_span_ids')}
    result['evidence_spans']=[resolve(c,answer) for c in value['answer_span_ids']]
    result['reference_spans']=[resolve(c,reference) for c in value['reference_span_ids']]
    return sj.validate(json.dumps(result,ensure_ascii=False),task)
