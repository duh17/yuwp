#!/usr/bin/env -S uv run --python 3.14 --script
"""Acquire the frozen public assets; never load or infer. --include-qwen adds the baseline."""
import concurrent.futures, hashlib, json, pathlib, subprocess, sys
root=pathlib.Path('/tmp/yuwp-english-parity')
meta=json.loads(pathlib.Path('benchmarks/fixtures/english-parity/model-sources.json').read_text())
def get(pair):
 m,f=pair
 dest=root/'models'/m['repo'].split('/')[-1]/f['rfilename']
 dest.parent.mkdir(parents=True,exist_ok=True)
 url=f"https://huggingface.co/{m['repo']}/resolve/{m['revision']}/{f['rfilename']}"
 p=subprocess.run(['curl','--fail','--location','--silent','--show-error','--retry','1','--max-time','240','--output',str(dest)+'.partial',url],capture_output=True,text=True)
 if p.returncode: raise RuntimeError(f"{url}: {p.stderr}")
 tmp=pathlib.Path(str(dest)+'.partial'); data=tmp.read_bytes(); sha=hashlib.sha256(data).hexdigest()
 assert len(data)==f['size'],(dest,len(data),f['size'])
 if 'lfs' in f: assert sha==f['lfs']['sha256'],dest
 else: assert hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()==f['blobId'],dest
 tmp.rename(dest)
 print(json.dumps({'path':str(dest),'sha256':sha,'size':len(data)}),flush=True)
jobs=[(m,f) for m in meta['models'] if m['repo'].startswith('FluidInference/') or '--include-qwen' in sys.argv for f in m['selected_files']]
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
 for _ in pool.map(get,jobs): pass
