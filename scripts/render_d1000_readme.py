#!/usr/bin/env python3
"""Refresh README findings and slide text from canonical verified aggregates."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from build_d1000_results import validate,pct,NAMES,CONDITIONS,comparison_markdown


def findings(payload):
    rr=[r for r in payload['comparison'] if r['available']]
    index={(r['model_key'],r['condition']):r for r in rr}
    best=max(rr,key=lambda r:r['correctness']);worst=min(rr,key=lambda r:r['correctness'])
    rag_improves=all(index[k,c+'-rag']['correctness']>index[k,c]['correctness'] for k in ('qwen3b','ministral3b','smollm1b7') for c in ('base','sft'))
    rag_statement='RAG raises observed correctness in each model family, with and without SFT' if rag_improves else 'RAG effects vary across model and fine-tuning conditions'
    lines=['### Findings supported by this snapshot','',
        '- **'+rag_statement+'** in this snapshot. Semantic coverage differs between arms; these are observed-subset comparisons, not unbiased population effects.',
        f'- **Highest observed correctness: {NAMES[best["model_key"]]} {CONDITIONS[best["condition"]]}, {pct(best["correctness"])}** '
        f'({best["correctness_numerator"]:g} rubric score points / {best["correctness_denominator"]} judged questions; '
        f'conditional question-bootstrap interval {pct(best["correctness_ci_low"])}–{pct(best["correctness_ci_high"])}). '
        f'Its balanced abstention is {pct(best["balanced_abstention"])} across all 76 saved answers. This is an observed ranking, not a statistically established universal winner.']
    changes=[]
    for key,before,after in [('qwen3b','base','sft'),('qwen3b','base-rag','sft-rag'),('ministral3b','base','sft'),('smollm1b7','base-rag','sft-rag')]:
        a=index[key,before];b=index[key,after]
        changes.append(f'{NAMES[key]} {CONDITIONS[before]}→{CONDITIONS[after]}: {pct(a["correctness"])}→{pct(b["correctness"])}')
    lines.append('- **SFT is not a uniform correctness improvement.** '+'; '.join(changes)+'. '
                 '[Paired common-question sensitivity analysis](reports/d1000_final/sft_effects.json) reports matched-question deltas and intervals separately.')
    a=index['ministral3b','sft'];b=index['ministral3b','sft-rag']
    lines.append(f'- **Refusal sensitivity can hide over-refusal.** Ministral SFT refuses {a["appropriate_refusal_rate_numerator"]}/9 unanswerable questions '
                 f'but also {a["false_refusal_rate_numerator"]}/67 answerable ones ({pct(a["false_refusal_rate"])} false refusal). '
                 f'With RAG this becomes {b["appropriate_refusal_rate_numerator"]}/9 appropriate refusals and '
                 f'{b["false_refusal_rate_numerator"]}/67 false refusals; balanced abstention is {pct(b["balanced_abstention"])}.')
    lines.append(f'- **Citation matching is not semantic support.** Ministral SFT+RAG resolves '
                 f'{b["citation_resolution_numerator"]}/{b["citation_resolution_denominator"]} IDs, while its available semantic citation-support mean is '
                 f'{pct(b["citation_support"])} across {b["citation_support_denominator"]} scored occurrences. '
                 'SmolLM Base+RAG has only one resolving ID and zero scored support occurrences; its resolution is not a grounding-reliability finding.')
    lines.append(f'- **Lowest observed correctness: {NAMES[worst["model_key"]]} {CONDITIONS[worst["condition"]]}, '
                 f'{pct(worst["correctness"])} ({worst["judged"]}/76 judged).** A balanced-abstention score of 50% can result from answering every question and refusing none.')
    return '\n'.join(lines)


def render(source,template,output):
    payload=validate(json.loads(Path(source).read_text()));p=payload['provenance']
    replacements={'CHECKPOINT':str(p['checkpoint']),'JUDGED':str(p['unique_judged']),
                  'COVERAGE_PERCENT':f'{100*p["unique_judged"]/912:.1f}','MISSING':str(p['remaining']),
                  'EXHAUSTED':str(p['remaining_gemini_terminal_states'].get('known_invalid_attempt_bound_exhausted',0)),
                  'QUARANTINED':str(p['quarantined_unjudged']),'STATUS':p['status'],
                  'GROQ_VOTES':str(p['provider_votes']['groq']),'GEMINI_VOTES':str(p['provider_votes']['gemini']),
                  'GROQ_SELECTED':str(p['selected_provider_counts'].get('groq',0)),
                  'GEMINI_SELECTED':str(p['selected_provider_counts'].get('gemini',0)),
                  'COMPARISON_TABLE':comparison_markdown(payload),'FINDINGS':findings(payload)}
    text=Path(template).read_text()
    for key,value in replacements.items():text=text.replace('{{'+key+'}}',value)
    if '{{' in text:raise ValueError('Unresolved README template token')
    Path(output).write_text(text)
    summary=f'''# Slide-ready D1000 summary — 3 November 2026

**Status: {p['status']}, provisional.** Verified checkpoint v{p['checkpoint']}: {p['unique_judged']}/912 judged; {p['remaining']} missing, including {p['quarantined_unjudged']} quarantined unjudged responses. No full-coverage claim.

**Research question:** Can relatively small language models reliably answer UK motor insurance questions, and what benefits do SFT and RAG provide?

**Method:** Three successfully trained families (Qwen2.5-3B, Ministral-3B, SmolLM2-1.7B), four conditions (Base, Base+RAG, SFT, SFT+RAG), one frozen 76-question test (67 answerable, 9 unanswerable), 912 saved responses. Mixed-provider single-valid-judge evaluation using calibrated Groq GPT-OSS-120B and Gemini3.5 Flash-Lite, unchanged rubric/schema and original nine-case calibration. Deterministic metrics cover all saved answers. D1000 source: 1,000 records, 462 verified and 538 unverified; actual split 751 training / 83 validation.

{findings(payload)}

**Limitations:** Mixed-provider judging; unequal, potentially nonrandom semantic missingness; no completed human review; mixed-quality source examples; a small benchmark with only nine unanswerable items; one fixed training run; unavailable DeepSeek arm. Conditional bootstrap intervals do not capture judge error, training variation or missing-data bias. Four historical unknown requests remain protected.

**Takeaway:** The evidence supports supplying retrieved policy passages, but fine-tuning must be checked for correctness and over-refusal. Citation matching and semantic support answer different questions. Present the valid evidence with sample sizes and missingness.

**Suggested slides:** Figure1 main correctness; Figure2 positive/negative SFT effects; Figure3 refusal trade-offs; Figure4 citation reliability; Figure5 coverage/limitations. Figure6 is an optional compact summary. PNGs are 16:9; SVG/PDF are vector alternatives. No PowerPoint file was generated.
'''
    Path(source).parent.joinpath('presentation_summary.md').write_text(summary)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--input',default='reports/d1000_final/metrics_summary.json');ap.add_argument('--template',default='reports/d1000_final/README.template.md');ap.add_argument('--output',default='README.md')
    args=ap.parse_args();render(args.input,args.template,args.output)
