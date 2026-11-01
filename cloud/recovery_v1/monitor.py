"""Aggregate-only notebook progress after original checkpoint readback."""
def safe_progress(state):
    persisted = state['last_persistence']
    if persisted.get('private') is not True or persisted.get('readback_verified') is not True:
        raise RuntimeError('Checkpoint readback must complete before reporting success')
    counts = state['judge_counts']
    return {'event':'private_checkpoint_readback','checkpoint':persisted['version'],
            'private':True,'readback_verified':True,'archive_sha256':state['artifacts_sha256'],
            'judge_counts':{p:counts[p] for p in ('groq','gemini')},
            'responses_preserved':state['responses_completed'],
            'providers':{p:{k:v for k,v in value.items() if k in
                ('status','completed','calibration','resume_after_epoch')}
                for p,value in state.get('judges',{}).items()}}


def install(evaluation, sink):
    original = evaluation.checkpoint
    def observed_checkpoint(state):
        result = original(state)
        report=safe_progress(state)
        from pathlib import Path
        archive=Path('/kaggle/working/publish/artifacts.bin')
        if archive.is_file():report.update(saved_vote_counts(archive))
        sink(report)
        return result
    evaluation.checkpoint = observed_checkpoint


def saved_vote_counts(archive):
    """Read only vote IDs from the just-readback archive inside the cloud VM."""
    import json
    import tarfile
    votes={};duplicates={}
    wanted={'judging/groq.jsonl':'groq','judging/gemini.jsonl':'gemini'}
    with tarfile.open(archive,mode='r|gz') as snapshot:
        for member in snapshot:
            if member.name not in wanted:continue
            provider=wanted[member.name]
            rows=[json.loads(line) for line in snapshot.extractfile(member) if line.strip()]
            ids={row['evaluation_id'] for row in rows}
            votes[provider]=ids;duplicates[provider]=len(rows)-len(ids)
            if len(votes)==2:break
    if set(votes)!={'groq','gemini'}:raise RuntimeError('Persisted vote journals absent')
    return {'persisted_judge_counts':{p:len(votes[p]) for p in votes},
            'persisted_paired':len(votes['groq'] & votes['gemini']),
            'persisted_duplicate_votes':duplicates,'persisted_vote_journals_audited':True}
