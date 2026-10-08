"""Private diagnostics for adapter reload; all numerical and functional gates remain strict."""
from .core import config, digest, token_ids, write_json


def mismatches(left, right):
    return [k for k in sorted(set(left) | set(right)) if left.get(k) != right.get(k)]


def runtime(model):
    base = model.get_base_model()
    quant = getattr(base.config, 'quantization_config', {})
    if hasattr(quant, 'to_dict'):
        quant = quant.to_dict()
    return {'checkpoint': config()['models']['ministral3b'],
        'loaded_name': base.config._name_or_path,
        'commit': getattr(base.config, '_commit_hash', None),
        'quantization': quant, 'config_dtype': str(getattr(base.config, 'dtype', None)),
        'attention': getattr(base.config, '_attn_implementation', None),
        'parameter_dtypes': {n: str(p.dtype) for n,p in model.named_parameters()},
        'compute_dtypes': {n: str(m.compute_dtype) for n,m in model.named_modules()
                           if hasattr(m, 'compute_dtype')}}


def probe(model, tokenizer, ids_cpu, mask_cpu):
    import torch
    model.eval()
    model.gradient_checkpointing_disable()
    model.config.use_cache = True
    if any(m.training for m in model.modules()):
        raise ValueError('A reloaded module is still in training mode')
    ids, mask = ids_cpu.to(model.device), mask_cpu.to(model.device)
    with torch.inference_mode():
        logits = model(input_ids=ids, attention_mask=mask).logits[:, -1, :].float().cpu()
        # Greedy decoding is the temperature-zero behaviour; temperature is not
        # supplied because it is a sampling-only parameter in Transformers.
        generated = model.generate(input_ids=ids, attention_mask=mask,
            do_sample=False, num_beams=1, max_new_tokens=32,
            pad_token_id=tokenizer.pad_token_id)
    completion = generated[0, ids.shape[1]:].cpu().tolist()
    return {'logits': logits, 'tokens': completion,
            'text': tokenizer.decode(completion, skip_special_tokens=True),
            'runtime': runtime(model)}


def compare(before, after):
    import torch
    difference = (before['logits'] - after['logits']).abs()
    return {'max_absolute_difference': difference.max().item(),
        'mean_absolute_difference': difference.mean().item(),
        'logits_within_original_tolerance': bool(torch.allclose(
            before['logits'], after['logits'], atol=0.02, rtol=0.01)),
        'atol': 0.02, 'rtol': 0.01,
        'before_top1': before['logits'].argmax(-1).tolist(),
        'after_top1': after['logits'].argmax(-1).tolist(),
        'top1_equal': bool(torch.equal(before['logits'].argmax(-1), after['logits'].argmax(-1))),
        'generated_tokens_equal': before['tokens'] == after['tokens'],
        'generated_text_equal': before['text'] == after['text'],
        'generation_nonempty': bool(before['text'].strip() and after['text'].strip()),
        'runtime_equal': before['runtime'] == after['runtime'],
        'dtype_mismatches': mismatches(before['runtime']['parameter_dtypes'], after['runtime']['parameter_dtypes'])}


def require_comparison(report):
    required = ['logits_within_original_tolerance', 'top1_equal',
                'generated_tokens_equal', 'generated_text_equal', 'generation_nonempty', 'runtime_equal']
    if any(report.get(k) is not True for k in required):
        raise ValueError('Adapter reload gate failed; inspect reload_diagnostics.json')


def adapter_check(expected, actual, trained_names):
    import torch
    # PEFT removes the adapter name ('default') from serialized tensor keys.
    required = {n.replace('.default.', '.') for n in trained_names}
    missing_trained = sorted(required - set(actual))
    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))
    different = sorted(k for k in set(expected) & set(actual)
                       if expected[k].dtype != actual[k].dtype or
                       not torch.equal(expected[k], actual[k].detach().cpu()))
    return {'trained_parameter_count': len(required), 'tensor_count': len(actual),
            'missing_trained': missing_trained, 'missing': missing, 'extra': extra,
            'different': different, 'equal': not (missing_trained or missing or extra or different)}
