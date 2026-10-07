"""Reproduce the inherited embedding index from frozen chunks, without re-ingestion."""
import argparse
from pathlib import Path
from .core import benchmark,common,private_output,rows,sha,write_json


def build(home,output):
    benchmark(home)
    from openai import OpenAI
    import chromadb
    chunks=rows(Path(home)/'data/chunks.jsonl')
    out=private_output(output)
    client=OpenAI(base_url=common.LMSTUDIO_BASE_URL,api_key='lm-studio')
    col=chromadb.PersistentClient(path=str(out/'chroma')).create_collection(
        common.COLLECTION,metadata={'hnsw:space':'cosine'})
    for start in range(0,len(chunks),32):
        batch=chunks[start:start+32]
        texts=['search_document: '+common.chunk_header(c['meta'])+'\n'+c['text'] for c in batch]
        embeddings=client.embeddings.create(model=common.EMBED_MODEL,input=texts)
        col.add(ids=[c['id'] for c in batch],documents=[c['text'] for c in batch],
            metadatas=[c['meta'] for c in batch],embeddings=[e.embedding for e in embeddings.data])
    if col.count()!=len(chunks):raise RuntimeError('Incomplete corpus index')
    write_json(out/'index_manifest.json',{'chunks_sha256':sha(Path(home)/'data/chunks.jsonl'),
        'n_chunks':col.count(),'embedding_model':common.EMBED_MODEL})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--home',required=True);p.add_argument('--output',required=True)
    build(**vars(p.parse_args()))
