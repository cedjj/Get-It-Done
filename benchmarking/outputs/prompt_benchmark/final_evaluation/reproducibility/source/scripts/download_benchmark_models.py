"""Explicitly download only the local vision model tags authorized for this run."""
import argparse
import json
from pathlib import Path
import shutil
import time
import requests

from project_paths import ROOT

TAGS = ['qwen3-vl:8b-instruct-q4_K_M', 'gemma3:4b', 'qwen3-vl:4b-instruct-q4_K_M']


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',action='append',dest='models',help='Explicit local model tags to download')
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/prompt_benchmark/multimodal_v2/model_downloads.json')
    args=parser.parse_args()
    tags=args.models or TAGS
    assert all('cloud' not in t and '/' not in t for t in tags), 'Only explicit local library models supported'
    args.output.parent.mkdir(parents=True,exist_ok=True)
    assert shutil.disk_usage(ROOT).free > 18 * 1024**3, 'Insufficient space for selected downloads'
    records = []
    for tag in tags:
        start, last = time.perf_counter(), 0
        print(json.dumps({'event': 'pull_start', 'model': tag}), flush=True)
        with requests.post('http://127.0.0.1:11434/api/pull', json={'model': tag, 'stream': True},
                           stream=True, timeout=(10, 180)) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line:
                    continue
                item = json.loads(line)
                if 'error' in item:
                    raise RuntimeError(item['error'])
                if time.perf_counter() - last > 10 or item.get('status') == 'success':
                    print(json.dumps({'model': tag, **item}), flush=True)
                    last = time.perf_counter()
        records.append({'model': tag, 'seconds': time.perf_counter() - start})
        args.output.write_text(json.dumps(records, indent=2), encoding='utf-8')
