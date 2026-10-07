"""Small IO and runtime helpers; no model dependencies imported at module load."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import common
import rag


def load_module(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rows(path):
    with Path(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]


def unique(records, key):
    result = {r[key]: r for r in records}
    if len(result) != len(records) or not result:
        raise ValueError(f'Empty or duplicate {key} population')
    return result


def private_output(path):
    path = Path(path).resolve()
    if path == ROOT or ROOT in path.parents:
        raise ValueError('Data/model outputs must be outside the public code repository')
    # Never overwrite an existing experiment or an inherited data directory.
    if path.exists():
        raise FileExistsError(f'Choose a new output directory: {path}')
    path.mkdir(parents=True)
    return path


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def write_rows(path, records):
    Path(path).write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in records))


def config():
    return json.loads((ROOT / 'configs/insurance_3b.json').read_text())


def benchmark(home):
    home = Path(home).resolve()
    gold = rows(home / 'data/gold/gold_v2.jsonl')
    unique(gold, 'id')
    freeze = json.loads((home / 'data/gold/FREEZE_v2.json').read_text())
    test = [r for r in gold if r['split'] == 'test']
    if len(test) != 140 or {r['id'] for r in test} != set(freeze['test_ids']):
        raise ValueError('Frozen test population changed')
    if any(digest(r) != freeze['record_sha256'][r['id']] for r in test):
        raise ValueError('Frozen test record changed')
    if sha(home / 'data/chunks.jsonl') != freeze['index_receipt']['chunks_sha256']:
        raise ValueError('Frozen corpus changed')
    return gold


def token_ids(tokenizer, messages, generation=False):
    ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=generation, return_dict=False)
    if not isinstance(ids, list) or not ids or not isinstance(ids[0], int):
        raise ValueError('Expected a single token-ID list from the pinned chat template')
    return ids


def encode_training(tokenizer, messages, max_length):
    if [m['role'] for m in messages] != ['system', 'user', 'assistant']:
        raise ValueError('Expected exactly system/user/assistant')
    prompt = token_ids(tokenizer, messages[:-1], True)
    full = token_ids(tokenizer, messages)
    if full[:len(prompt)] != prompt or len(full) <= len(prompt):
        raise ValueError('Chat template does not preserve the prompt prefix; refuse unsafe masking')
    if len(full) > max_length:
        raise ValueError(f'Training sequence has {len(full)} tokens > {max_length}; no truncation')
    completion = tokenizer.decode(full[len(prompt):], skip_special_tokens=True)
    if completion.strip() != messages[-1]['content'].strip():
        raise ValueError('Supervised token span does not decode to the complete target')
    return {'input_ids': full, 'attention_mask': [1]*len(full),
            'labels': [-100]*len(prompt) + full[len(prompt):]}


def load_tokenizer(key):
    from transformers import AutoTokenizer
    c = config()['models'][key]
    kw = {'fix_mistral_regex': True} if key == 'ministral3b' else {}
    return AutoTokenizer.from_pretrained(c['id'], revision=c['revision'], **kw)


def load_model(key):
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, Mistral3ForConditionalGeneration, BitsAndBytesConfig
    if not torch.cuda.is_available():
        raise RuntimeError('A CUDA GPU is required for this 4-bit runtime; use the private Kaggle stage')
    cfg = config()['models'][key]
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype)
    model_config = AutoConfig.from_pretrained(cfg['id'], revision=cfg['revision'])
    if getattr(model_config, 'quantization_config', None):
        model_config.quantization_config['bnb_4bit_compute_dtype'] = str(dtype).split('.')[-1]
    cls = Mistral3ForConditionalGeneration if cfg['class'] == 'mistral3' else AutoModelForCausalLM
    model = cls.from_pretrained(cfg['id'], revision=cfg['revision'], config=model_config, quantization_config=quant,
        device_map={'': 0}, dtype=dtype, attn_implementation='sdpa')
    if not getattr(model, 'is_loaded_in_4bit', False):
        raise RuntimeError('Expected 4-bit model loading')
    tokenizer = load_tokenizer(key)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def generate(model, tokenizer, messages, max_new_tokens=700):
    import torch
    ids = token_ids(tokenizer, messages, True)
    if len(ids) + max_new_tokens > config()['evaluation']['max_total_tokens']:
        raise ValueError('Evaluation prompt exceeds runtime budget; no evidence truncation permitted')
    x = torch.tensor([ids], device=model.device)
    model.eval()
    with torch.inference_mode():
        y = model.generate(input_ids=x, attention_mask=torch.ones_like(x), do_sample=False,
            max_new_tokens=max_new_tokens, pad_token_id=tokenizer.pad_token_id)
    return common.strip_think(tokenizer.decode(y[0, len(ids):], skip_special_tokens=True)), len(ids)
