"""Download immutable public RFCs and the pinned Laya English checkpoint."""
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).parent
REVISION = '1c5edc17a7acd8701df6fc341c0d179f1c62c982'

def corpus():
    dest = ROOT / 'corpus'
    dest.mkdir(exist_ok=True)
    manifest = []
    for number in [8259, 7519, 9110, 9111, 6455]:
        url = f'https://www.rfc-editor.org/rfc/rfc{number}.txt'
        raw = urlopen(url, timeout=60).read()
        name = f'rfc{number}.txt'
        (dest / name).write_bytes(raw)
        manifest.append({'id':f'rfc{number}', 'url':url, 'file':name,
                         'sha256':hashlib.sha256(raw).hexdigest()})
    (dest / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print('Corpus saved:', len(manifest), flush=True)

if __name__ == '__main__':
    corpus()
    from huggingface_hub import snapshot_download
    path = snapshot_download('convaiinnovations/laya', revision=REVISION,
        allow_patterns=['model.safetensors','rl_agent_config.json','encoder/*','tokenizer/*'])
    (ROOT / 'model-path.txt').write_text(path)
    print('Pinned checkpoint downloaded', flush=True)
