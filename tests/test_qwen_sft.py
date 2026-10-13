"""Matched-SFT safety/regression tests using synthetic records only."""
import json
from types import SimpleNamespace as NS
from pathlib import Path
import pytest
from experiments import core, train, tokenise, kaggle_stage, reload_check
from scripts.error_analysis import sft_effects
from scripts.comparison_report import render_effects


def qwen_model(layers=2):
    modules={f'model.layers.{i}.{block}.{name}':NS(weight=NS(ndim=2))
        for i in range(layers) for block,names in [
            ('self_attn',['q_proj','k_proj','v_proj','o_proj']),
            ('mlp',['gate_proj','up_proj','down_proj'])] for name in names}
    modules.update({'model.embed_tokens':NS(weight=NS(ndim=2)),
                    'model.norm':NS(weight=NS(ndim=1)), 'lm_head':NS(weight=NS(ndim=2))})
    return NS(config=NS(model_type='qwen2',num_hidden_layers=layers),named_modules=lambda:modules.items()),modules


def test_qwen_targets_all_and_only_seven_projections_per_layer():
    m,_=qwen_model(36);targets=train.language_targets(m,'qwen3b')
    assert len(targets)==252
    assert not any(x in n for n in targets for x in ['embed','norm','lm_head'])


@pytest.mark.parametrize('violation',['missing','architecture','nonlinear'])
def test_invalid_qwen_architecture_fails(violation):
    m,modules=qwen_model()
    if violation=='missing':modules.pop('model.layers.0.mlp.down_proj')
    elif violation=='architecture':m.config.model_type='other'
    else:modules['model.layers.0.mlp.down_proj'].weight.ndim=1
    with pytest.raises(ValueError):train.language_targets(m,'qwen3b')


def test_ministral_selection_and_default_are_unchanged():
    names=['model.language_model.layers.0.self_attn.q_proj','model.language_model.layers.0.mlp.down_proj',
        'model.vision_tower.q_proj','model.multi_modal_projector.q_proj','lm_head']
    model=NS(named_modules=lambda:[(n,None) for n in names])
    assert train.language_targets(model)==names[:2]
    assert train.language_targets(model,'ministral3b')==names[:2]


@pytest.mark.parametrize('family',['ministral3b','qwen3b'])
def test_exact_lora_trainable_population(family):
    targets=['model.language_model.layers.0.q_proj'] if family=='ministral3b' else ['model.layers.0.self_attn.q_proj']
    params={f'base_model.model.{targets[0]}.lora_{s}.default.weight':NS(requires_grad=True) for s in ['A','B']}
    params['base_model.model.model.embed_tokens.weight']=NS(requires_grad=False)
    model=NS(named_parameters=lambda:params.items())
    assert len(train.validate_trainable(model,targets))==2
    params['base_model.model.model.embed_tokens.weight'].requires_grad=True
    with pytest.raises(ValueError):train.validate_trainable(model,targets)
    params.pop('base_model.model.model.embed_tokens.weight');params.pop(next(iter(params)))
    with pytest.raises(ValueError):train.validate_trainable(model,targets)


def test_smoke_fingerprint_is_model_specific_and_ministral_shape_is_unchanged():
    manifest={'test':'synthetic'}
    assert train.fingerprint(manifest)==core.digest({'training_manifest':manifest,'config':core.config(),'code':train.code_hash()})
    assert train.fingerprint(manifest,'qwen3b')!=train.fingerprint(manifest)


def test_qwen_cannot_reuse_ministral_receipt(tmp_path):
    p=tmp_path/'receipt.json';core.write_json(p,{'mode':'smoke','fingerprint':train.fingerprint({},'qwen3b'),
        'model_key':'ministral3b','gates':dict.fromkeys(['leakage','tokenisation','masking','finite_loss','language_only','adapter_reload'],True)})
    with pytest.raises(ValueError):train.require_smoke(p,{},'qwen3b')


def test_qwen_rejects_ministral_tokenisation_and_changed_population():
    m={'tokenisation_checked':True,'tokenizer':core.config()['models']['ministral3b'],'train_rows':21,
        'matched_train_sha256':'original','train_sha256':'original'}
    train.validate_training_manifest(m,'ministral3b')
    with pytest.raises(ValueError):train.validate_training_manifest(m,'qwen3b')
    m['tokenizer']=core.config()['models']['qwen3b'];train.validate_training_manifest(m,'qwen3b')
    m['train_sha256']='changed'
    with pytest.raises(ValueError):train.validate_training_manifest(m,'qwen3b')
    m['train_sha256']='original';m['train_rows']=20
    with pytest.raises(ValueError):train.validate_training_manifest(m,'qwen3b')


class Tokenizer:
    def apply_chat_template(self,messages,**kwargs):return list(''.join(m['content'] for m in messages).encode())
    def decode(self,ids,**kwargs):return bytes(ids).decode()


def bundle(tmp_path,long=False):
    p=tmp_path/'input';p.mkdir();rows=[]
    for i in range(21):
        msgs=[{'role':'system','content':'rules'},{'role':'user','content':'q'*(2048 if long and i==0 else 20)},
              {'role':'assistant','content':'synthetic answer'}]
        rows.append({'source_record_id':str(i),'messages':msgs,'provenance':{'source_split':'seed',
            'source_file':'gold_v2.jsonl','messages_sha256':core.digest(msgs),'evidence':[{'synthetic':True}]}})
    core.write_rows(p/'train.jsonl',rows)
    core.write_json(p/'manifest.json',{'schema':'seed-sft/1','leakage_checks_passed':True,'train_rows':21,
        'train_sha256':core.sha(p/'train.jsonl'),'tokenisation_checked':True,
        'tokenizer':core.config()['models']['ministral3b']})
    return p


def test_matched_tokenisation_preserves_training_bytes(tmp_path,monkeypatch):
    p=bundle(tmp_path);monkeypatch.setattr(tokenise,'load_tokenizer',lambda _:Tokenizer())
    tokenise.prepare_matched(p,tmp_path/'out')
    assert (p/'train.jsonl').read_bytes()==(tmp_path/'out/train.jsonl').read_bytes()
    m=json.loads((tmp_path/'out/manifest.json').read_text());assert len(m['token_lengths'])==21
    train.validate_training_manifest(m,'qwen3b')


def test_one_overlength_row_stops_without_producing_smaller_bundle(tmp_path,monkeypatch):
    p=bundle(tmp_path,long=True);monkeypatch.setattr(tokenise,'load_tokenizer',lambda _:Tokenizer())
    with pytest.raises(ValueError,match='STOP'):tokenise.prepare_matched(p,tmp_path/'out')
    assert not (tmp_path/'out').exists()


@pytest.mark.parametrize('model',['ministral3b','qwen3b'])
def test_smoke_stage_runs_requested_model_privately(tmp_path,model):
    p=bundle(tmp_path);kaggle_stage.stage(p,tmp_path/'stage','synthetic','smoke',model)
    script=(tmp_path/'stage/kernel/run.py').read_text()
    assert ("'--model', 'qwen3b'" in script)==(model=='qwen3b')
    meta=json.loads((tmp_path/'stage/kernel/kernel-metadata.json').read_text())
    assert meta['is_private'] is True and meta['id'].endswith(model)
    assert {f.name for f in (tmp_path/'stage/dataset/payload').iterdir()}=={'train.jsonl','manifest.json'}


def test_reload_runtime_reports_actual_model_identity():
    base=NS(config=NS(_name_or_path='synthetic'))
    m=NS(get_base_model=lambda:base,named_modules=lambda:[],named_parameters=lambda:[])
    assert reload_check.runtime(m,'qwen3b')['checkpoint']==core.config()['models']['qwen3b']
    assert reload_check.runtime(m)['checkpoint']==core.config()['models']['ministral3b']


def test_cross_model_changes_are_within_family_and_pending_stays_missing():
    models={f'{family}-{kind}-openbook':{'full':{'citation':v,'parse_rate':v}}
        for family,kind,v in [('ministral3b','base',.2),('ministral3b','sft',.5),('qwen3b','base',.7),('qwen3b','sft',.6)]}
    r=sft_effects(models)['full']
    assert r['citation']['ministral3b']['delta']==pytest.approx(.3)
    assert r['citation']['qwen3b']['delta']==pytest.approx(-.1)
    assert r['correctness']['qwen3b']['delta'] is None
    assert 'Pending' in render_effects({'models':models})
    assert sft_effects(models)['motor']['citation']['qwen3b']['delta'] is None
