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

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--bundle',required=True);p.add_argument('--output',required=True)
    prepare(**vars(p.parse_args()))
