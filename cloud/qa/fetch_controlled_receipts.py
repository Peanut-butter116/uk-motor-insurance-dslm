"""Retrieve aggregate receipts only; never paginate through private QA outputs."""
import argparse
import json
from pathlib import Path


def fetch(output):
    from kaggle import api
    kernel = 'scarletthe0116/uk-motor-qa-checkpoint-check'
    status = api.kernels_status(kernel).status.name
    print(json.dumps({'notebook_status': status}), flush=True)
    if status != 'COMPLETE':
        return
    # An explicit empty cursor requests only the first page. The CLI otherwise
    # scans thousands of scratch files even with a restrictive file pattern.
    paths, _ = api.kernels_output(kernel, output, force=True, quiet=True,
        page_token='', page_size=200,
        file_pattern=r'^(controlled_batch_report|persistence_receipt)\.json$')
    for path in paths:
        if Path(path).name not in ('controlled_batch_report.json', 'persistence_receipt.json'):
            raise ValueError('Unexpected downloaded output')
    print(json.dumps({'aggregate_files': [Path(p).name for p in paths]}))
    report = Path(output)/'controlled_batch_report.json'
    if report.exists():
        data = json.loads(report.read_text())
        print(json.dumps({k: data.get(k) for k in (
            'test_success', 'repair_success', 'total_accepted', 'accepted_delta', 'generated_delta',
            'status', 'persisted_dataset_version', 'changed_candidate_dispositions',
            'changed_candidate_failure_reasons', 'exception_class')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--output', required=True)
    fetch(**vars(parser.parse_args()))
