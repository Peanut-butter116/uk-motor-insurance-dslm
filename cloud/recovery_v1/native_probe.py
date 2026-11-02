"""One bounded Kaggle-native auth probe. Never calls Secrets or judge APIs."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

STORE = 'scarletthe0116/uk-motor-d1000-models-results'
SCRATCH = 'scarletthe0116/uk-motor-d1000-auth-probe-20261101'


def safe_error(exc):
    chain = []; seen = set(); current = exc
    while current is not None and id(current) not in seen and len(chain) < 8:
        seen.add(id(current))
        item = {'type': type(current).__name__, 'module': type(current).__module__}
        status = getattr(current, 'code', None)
        if not isinstance(status, int):
            status = getattr(getattr(current, 'response', None), 'status_code', None)
        if isinstance(status, int): item['http_status'] = status
        chain.append(item); current = current.__cause__ or current.__context__
    return chain


def run():
    report = {'kind': 'native_auth_probe_v1', 'new_judge_calls': 0,
              'secret_service_calls': 0, 'production_dataset_writes': 0,
              'scratch_dataset': SCRATCH, 'status': 'BLOCKED'}
    try:
        import importlib.metadata
        report['kaggle_version'] = importlib.metadata.version('kaggle')
        # Only the presence of the SDK-supported notebook credential is inspected.
        token_path = os.environ.get('KAGGLE_API_V1_TOKEN')
        report['native_token_file_available'] = bool(token_path and Path(token_path).is_file())
        if not report['native_token_file_available']:
            report['blocked_at'] = 'notebook_native_token_unavailable'
            return report
        from kaggle import api
        from kagglesdk.datasets.types.dataset_api_service import ApiGetDatasetRequest
        def info(dataset):
            req = ApiGetDatasetRequest(); req.owner_slug, req.dataset_slug = dataset.split('/')
            with api.build_kaggle_client() as client:
                return client.datasets.dataset_api_client.get_dataset(req)
        api.authenticate()
        report['authentication_completed'] = True
        d = info(STORE); assert d.is_private is True
        report['private_results_readable'] = True
        report['latest_checkpoint'] = d.current_version_number
        assert d.current_version_number == 239, 'Reconcile newer checkpoint before continuing'
        with tempfile.TemporaryDirectory() as tmp:
            api.dataset_download_file(STORE+'/239', 'progress.json', path=tmp, quiet=True, force=True, unzip=True)
            state = json.loads((Path(tmp)/'progress.json').read_text())
            report['progress_readable'] = True
            report['archive_pointer_matches_239_receipt'] = state['artifacts_sha256'] == 'aa9e5eba75d908c13da03c0c64ce47c55794e9aaeb74d16a25312fd90f86a604'
        # Scratch contains a random nonce only, never research artifacts or credentials.
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp); payload = os.urandom(32).hex().encode()
            (folder/'probe.txt').write_bytes(payload)
            (folder/'dataset-metadata.json').write_text(json.dumps({'id': SCRATCH,
                'title': SCRATCH.split('/')[1], 'licenses': [{'name':'other'}]}))
            result = api.dataset_create_new(tmp, public=False, quiet=True, convert_to_csv=False)
            assert not getattr(result, 'error', None), 'Scratch creation rejected'
            for _ in range(12):
                d = info(SCRATCH); assert d.is_private is True
                if api.dataset_status(SCRATCH) == 'ready': break
                time.sleep(5)
            else: raise RuntimeError('Scratch readiness timeout')
            with tempfile.TemporaryDirectory() as readback:
                api.dataset_download_file(SCRATCH+'/'+str(d.current_version_number), 'probe.txt',
                    path=readback, quiet=True, force=True, unzip=True)
                assert (Path(readback)/'probe.txt').read_bytes() == payload
            report['scratch_private_write_readback'] = True
        report['status'] = 'PASS_NATIVE_STORAGE_ONLY'
        report['judge_credentials_verified'] = False
        report['checkpoint_archive_restored'] = False
    except BaseException as exc:
        report['exception_chain'] = safe_error(exc)
    return report


if __name__ == '__main__':
    Path('/kaggle/working/evaluation_diagnostic_receipt.json').write_text(json.dumps(run(), indent=2))
