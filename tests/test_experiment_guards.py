"""Synthetic safety checks; no private records required in CI."""
import json
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiments import core, prepare_sft, train

class Tokenizer:
    def apply_chat_template(self,messages,tokenize=True,add_generation_prompt=False, return_dict=False):
        s=''.join(m['content'] for m in messages)
        return list(s.encode())
    def decode(self,ids,skip_special_tokens=True):return bytes(ids).decode()

MESSAGES=[{'role':'system','content':'rules'},{'role':'user','content':'question'},
          {'role':'assistant','content':'answer'}]

def test_assistant_only_labels():
    r=core.encode_training(Tokenizer(),MESSAGES,100)
    assert r['labels'][:13]==[-100]*13
    assert r['labels'][13:]==list(b'answer')

def test_no_silent_truncation():
    with pytest.raises(ValueError,match='no truncation'):core.encode_training(Tokenizer(),MESSAGES,15)

def test_template_boundary_mismatch_fails():
    class Bad(Tokenizer):
        def apply_chat_template(self,messages,**kw):return [0]+super().apply_chat_template(messages,**kw) if len(messages)==3 else [1]
    with pytest.raises(ValueError,match='prefix'):core.encode_training(Bad(),MESSAGES,100)

def test_empty_completion_fails():
    m=MESSAGES[:-1]+[{'role':'assistant','content':''}]
    with pytest.raises(ValueError):core.encode_training(Tokenizer(),m,100)

def test_public_output_rejected():
    with pytest.raises(ValueError):core.private_output(core.ROOT/'outputs/x')

def test_existing_output_rejected(tmp_path):
    with pytest.raises(FileExistsError):core.private_output(tmp_path)

def seed(**kw):
    return dict({'id':'seed-a','split':'seed','question':'Synthetic question alpha?',
        'gold_answer':'Synthetic answer alpha.','review_status':'verified'},**kw)

@pytest.mark.parametrize('split',['test','dev'])
def test_nonseed_rejected(split):
    assert 'not_seed' in prepare_sft.leakage_reasons(seed(split=split),[],set())

def test_duplicate_question_rejected():
    other=seed(id='test-a',split='test')
    assert 'question_overlap' in prepare_sft.leakage_reasons(seed(),[other],set())

def test_duplicate_id_rejected():
    assert 'forbidden_id' in prepare_sft.leakage_reasons(seed(),[seed(split='test')],set())

def test_intent_overlap_rejected():
    assert 'test_or_dev_intent_cluster' in prepare_sft.leakage_reasons(seed(),[],{'seed-a'})

def test_unverified_rejected():
    assert 'unverified' in prepare_sft.leakage_reasons(seed(review_status='pending'),[],set())

def test_no_valid_smoke_means_no_full(tmp_path):
    p=tmp_path/'receipt.json';p.write_text('{}')
    with pytest.raises(ValueError,match='blocked'):train.require_smoke(p,{})

def test_smoke_stale_hash_rejected(tmp_path):
    p=tmp_path/'receipt.json';p.write_text(json.dumps({'mode':'smoke','fingerprint':'old','gates':dict.fromkeys(
        ['leakage','tokenisation','masking','finite_loss','language_only','adapter_reload'],True)}))
    with pytest.raises(ValueError):train.require_smoke(p,{})

def test_vision_modules_never_targeted():
    class M:
        def named_modules(self):return [(n,None) for n in ['model.language_model.layers.0.q_proj',
            'model.vision_tower.layers.0.q_proj','model.multi_modal_projector.q_proj']]
    assert train.language_targets(M())==['model.language_model.layers.0.q_proj']

def test_no_language_modules_fails():
    class M:
        def named_modules(self):return [('vision.q_proj',None)]
    with pytest.raises(ValueError):train.language_targets(M())

def test_duplicate_population_fails():
    with pytest.raises(ValueError):core.unique([{'id':'x'},{'id':'x'}],'id')

def test_empty_population_fails():
    with pytest.raises(ValueError):core.unique([],'id')

def test_support_must_exist():
    s={'gold_citations':[{'insurer':'Example','doc_file':'fake.pdf','quote':'not present','page':'1'}]}
    with pytest.raises(ValueError,match='quote'):prepare_sft.select_evidence(s,[])

def test_heldout_insurer_excluded():
    s={'gold_citations':[{'insurer':'LV='}]}
    with pytest.raises(ValueError,match='held_out'):prepare_sft.select_evidence(s,[])

def test_independent_shared_evidence_allowed():
    s={'gold_citations':[{'insurer':'Example','doc_file':'fake.pdf','quote':'sample support','page':'1'}]}
    c={'id':'shared','text':'A sample support passage.','meta':{'insurer':'Example','filename':'fake.pdf','page':1}}
    assert prepare_sft.select_evidence(s,[c])==[c]

def test_score_does_not_publish_partial_composite(tmp_path,monkeypatch):
    from experiments import evaluate
    g=[{'id':'a','split':'dev','question':'Question?','gold_answer':'Answer.',
        'key_facts':['Fact.'],'answerable':True}]
    home=tmp_path/'home';(home/'data/gold').mkdir(parents=True)
    (home/'data/gold/gold_v2.jsonl').write_text('synthetic')
    bundle=tmp_path/'bundle';bundle.mkdir()
    core.write_rows(bundle/'contexts.jsonl',[{'qid':'a','chunks':[]}])
    core.write_json(bundle/'manifest.json',{'split':'dev','gold_sha256':core.sha(home/'data/gold/gold_v2.jsonl'),
        'contexts_sha256':core.sha(bundle/'contexts.jsonl')})
    ans=tmp_path/'answers.jsonl';core.write_rows(ans,[{'qid':'a','row':'synthetic','answer':'An answer.'}])
    monkeypatch.setattr(evaluate,'benchmark',lambda _:g)
    evaluate.score(home,bundle,ans,None,tmp_path/'score')
    assert not (tmp_path/'score/metrics.json').exists()
    blind=core.rows(tmp_path/'score/judge_tasks.jsonl')
    assert 'row' not in blind[0] and 'qid' not in blind[0]
    assert json.loads((tmp_path/'score/status.json').read_text())['missing']==1
