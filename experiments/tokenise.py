"""Tokenize audited seeds, reject whole overlength rows, preserve private provenance."""
import argparse
from .core import config, encode_training, load_tokenizer, private_output, sha, token_ids, write_json, write_rows
from .prepare_sft import validate_bundle


def prepare(bundle,output):
    data,manifest=validate_bundle(bundle)
    tokenizer=load_tokenizer('ministral3b');limit=config()['training']['max_length']
    accepted=[];excluded=[];lengths=[]
    for row in data:
        n=len(token_ids(tokenizer,row['messages']))
        if n>limit:
            excluded.append({'seed_id':row['source_record_id'],'reasons':['sequence_over_2048'], 'tokens':n});continue
        encoded=encode_training(tokenizer,row['messages'],limit)
        accepted.append(row);lengths.append({'seed_id':row['source_record_id'],'tokens':n,
            'supervised_tokens':sum(t!=-100 for t in encoded['labels'])})
    if not accepted:raise ValueError('No training examples remain under sequence limit')
    out=private_output(output);write_rows(out/'train.jsonl',accepted)
    write_json(out/'manifest.json',{**manifest,'train_rows':len(accepted),'train_sha256':sha(out/'train.jsonl'),
        'excluded':manifest['excluded']+excluded,'tokenisation_checked':True,
        'tokenizer':config()['models']['ministral3b'],'fix_mistral_regex':True,
        'max_length':limit,'token_lengths':lengths})
    print(f'{len(accepted)} rows passed real-tokenizer masking; {len(excluded)} whole overlength rows excluded.')

def prepare_matched(bundle,output,model_key='qwen3b'):
    """Retokenise the completed 21-example population; never filter or truncate."""
    import shutil
    from pathlib import Path
    data,manifest=validate_bundle(bundle)
    if len(data)!=21 or manifest.get('tokenisation_checked') is not True:
        raise ValueError('Expected the completed 21-example Ministral bundle')
    if manifest.get('tokenizer')!=config()['models']['ministral3b']:
        raise ValueError('Matched source must be the audited Ministral training bundle')
    tokenizer=load_tokenizer(model_key);limit=config()['training']['max_length']
    lengths=[]
    for r in data:
        ids=token_ids(tokenizer,r['messages'])
        lengths.append({'seed_id':r['source_record_id'],'tokens':len(ids)})
    if any(r['tokens']>limit for r in lengths):
        raise ValueError('Matched design STOP: overlength examples; do not drop or truncate rows')
    for r,length in zip(data,lengths):
        encoded=encode_training(tokenizer,r['messages'],limit)
        length['supervised_tokens']=sum(t!=-100 for t in encoded['labels'])
    out=private_output(output);shutil.copy2(Path(bundle)/'train.jsonl',out/'train.jsonl')
    write_json(out/'manifest.json',{**manifest,'tokenizer':config()['models'][model_key],
        'fix_mistral_regex':model_key=='ministral3b','token_lengths':lengths,
        'matched_train_sha256':manifest['train_sha256'],
        'matched_source_manifest_sha256':sha(Path(bundle)/'manifest.json'),
        'max_length':limit})
    print(f'All {len(data)} matched examples passed tokenisation and assistant-only masking; no rows changed.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--bundle',required=True);p.add_argument('--output',required=True)
    p.add_argument('--model',choices=['ministral3b','qwen3b'],default='ministral3b')
    a=p.parse_args()
    if a.model=='ministral3b':prepare(a.bundle,a.output)
    else:prepare_matched(a.bundle,a.output,a.model)
