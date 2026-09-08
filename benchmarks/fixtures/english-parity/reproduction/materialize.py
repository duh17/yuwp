#!/usr/bin/env -S uv run --python 3.14 --script
"""Public fixture preparation only. No inference or result reads."""
import array
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import wave

ROOT = Path('/tmp/yuwp-english-parity/data')
REPO = Path.cwd()
OUT = REPO / 'benchmarks/fixtures/english-parity'
SOURCE = ROOT / 'source/extracted/LibriSpeech'
OUT.mkdir(parents=True, exist_ok=True)
(ROOT / 'audio').mkdir(exist_ok=True)

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def save_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')

def pcm(path):
    return subprocess.check_output(['ffmpeg', '-v', 'error', '-i', str(path), '-f', 's16le', '-ac', '1', '-ar', '16000', '-'])

def frames(path):
    b = path.read_bytes()[:42]
    assert b[:4] == b'fLaC' and b[4] & 127 == 0
    return int.from_bytes(b[18:26], 'big') & ((1 << 36) - 1)

def wav(name, data):
    path = ROOT / 'audio' / (name + '.wav')
    with wave.open(str(path), 'wb') as w:
        w.setparams((1, 2, 16000, 0, 'NONE', 'not compressed'))
        w.writeframes(data)
    return path

def row(name, data, text, **extra):
    path = wav(name, data)
    return dict(id=name, audio=str(path), language='en', metric='wer', reference=text,
                audio_sha256=sha(path), pcm_sha256=hashlib.sha256(data).hexdigest(),
                reference_sha256=hashlib.sha256(text.encode()).hexdigest(),
                samples=len(data)//2, duration_seconds=len(data)/32000, **extra)

archive = ROOT / 'source/test-clean.tar.gz'
md5 = hashlib.md5(archive.read_bytes()).hexdigest()
assert any(line.split()[0] == md5 and line.split()[-1] == 'test-clean.tar.gz'
           for line in (ROOT / 'source/md5sum.txt').read_text().splitlines())
speakers = sorted((SOURCE/'test-clean').iterdir(), key=lambda p: hashlib.sha256(('english-parity-v1:' + p.name).encode()).hexdigest())[:26]
rows = {'dev': [], 'heldout': []}
for si, speaker in enumerate(speakers):
    split = 'dev' if si < 6 else 'heldout'
    texts = {}
    for f in sorted(speaker.glob('*/*.trans.txt')):
        for line in f.read_text().splitlines():
            ident, text = line.split(' ', 1)
            texts[ident] = (text, f)
    files = sorted(speaker.glob('*/*.flac'))
    short = next(f for f in files if 3*16000 <= frames(f) <= 10*16000)
    # A long case joins whole source utterances from one chapter, never trims words.
    long = []
    for chapter in sorted(speaker.iterdir()):
        candidate = []
        for f in sorted(chapter.glob('*.flac')):
            if f == short:
                continue
            total = sum(frames(x) for x in candidate) + frames(f) + 4000*len(candidate)
            if total > 40*16000:
                candidate = []
                total = frames(f)
            candidate.append(f)
            if total >= 20*16000:
                long = candidate
                break
        if long:
            break
    assert long and short not in long
    for kind, selected in [('short', [short]), ('long', long)]:
        pieces = [pcm(f) for f in selected]
        data = (b'\0'*8000).join(pieces) # exactly 250ms between source utterances
        text = ' '.join(texts[f.stem][0] for f in selected)
        components = []
        offset = 0
        for f, part in zip(selected, pieces):
            components.append(dict(id=f.stem, archive_path=str(f.relative_to(SOURCE.parent)),
                                   source_sha256=sha(f), transcript_path=str(texts[f.stem][1].relative_to(SOURCE.parent)),
                                   transcript_sha256=sha(texts[f.stem][1]), start_sample=offset, samples=len(part)//2))
            offset += len(part)//2 + 4000
        rows[split].append(row(f'ls-{split}-{speaker.name}-{kind}', data, text, split=split,
                               corpus='LibriSpeech-test-clean', speaker=speaker.name, cluster=speaker.name,
                               kind=kind, components=components))

# Fixed transformed probes: preserve base references except deliberately ambiguous cutoff.
diag = []
for split in ['dev', 'heldout']:
    base = next(r for r in rows[split] if r['kind'] == 'long')
    with wave.open(base['audio'], 'rb') as w:
        data = w.readframes(w.getnframes())
    a = array.array('h'); a.frombytes(data)
    if sys.byteorder != 'little': a.byteswap()
    cut = base['components'][1]['start_sample']
    variants = [('silence-pad', b'\0'*32000 + data + b'\0'*64000, base['reference'], {'leading_seconds':1,'trailing_seconds':2}),
                ('pause', data[:cut*2] + b'\0'*96000 + data[cut*2:], base['reference'], {'insert_sample':cut,'pause_seconds':3}),
                ('cutoff', data[:-10240], '', {'removed_samples':5120,'lexical_gold':'unavailable; do not score full reference as gold'})]
    # Portable deterministic uniform noise: LCG32, no random-library or Gaussian-version dependence.
    state = 20260908
    noise = []
    for _ in a:
        state = (1664525*state + 1013904223) & 0xffffffff
        noise.append((state >> 16) - 32768)
    signal_rms = math.sqrt(sum(x*x for x in a)/len(a))
    noise_rms = math.sqrt(sum(x*x for x in noise)/len(noise))
    gain = signal_rms/(10*noise_rms) # 20dB SNR, entire-clip RMS
    mixed = array.array('h', [max(-32768,min(32767,round(x+gain*n))) for x,n in zip(a,noise)])
    if sys.byteorder != 'little': mixed.byteswap()
    variants.append(('noise20db', mixed.tobytes(), base['reference'], {'snr_db':20,'seed':20260908,'clipped_samples':sum(abs(x+gain*n)>32767 for x,n in zip(a,noise))}))
    for kind, transformed, text, details in variants:
        diag.append(row(base['id']+'-'+kind, transformed, text, split=split, kind=kind, base_id=base['id'], cluster=base['cluster'], diagnostic=True, transformation=details))
    diag.append(row(f'probe-{split}-silence', b'\0'*160000, '', split=split, kind='silence-only', diagnostic=True))
    state = 20260908 if split == 'dev' else 20260909
    noise = array.array('h')
    for _ in range(80000):
        state = (1664525*state + 1013904223) & 0xffffffff
        noise.append(round(((state>>16)-32768)*0.02))
    if sys.byteorder != 'little': noise.byteswap()
    diag.append(row(f'probe-{split}-noise', noise.tobytes(), '', split=split, kind='noise-only', diagnostic=True, transformation={'seed':20260908 if split=='dev' else 20260909,'lcg_scale':0.02}))

# Existing AMI source is public; all same-speaker segments stay dev-only, not heldout evidence.
ami = REPO/'benchmarks/fixtures/subtitle-long/ami-en2002b-d-0765-0945'
full = pcm(ami/'audio.m4a')
for i, seg in enumerate(json.loads((ami/'human-segments.json').read_text())['segments']):
    if not any(c.isalnum() for c in seg['text']):
        continue
    start, end = round(seg['start']*16000), round(seg['end']*16000)
    diag.append(row(f'ami-dev-segment-{i:02}', full[start*2:end*2], seg['text'], split='dev', kind='spontaneous',
                    diagnostic=True, cluster='AMI-EN2002b-D', source_start_sample=start, source_end_sample=end,
                    source_audio_sha256=sha(ami/'audio.m4a'), source_reference_sha256=sha(ami/'human-segments.json')))
for split, cases in rows.items():
    (OUT/f'{split}.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False,sort_keys=True)+'\n' for r in cases))
(OUT/'diagnostics.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False,sort_keys=True)+'\n' for r in diag))
summary = dict(schema_version=1, status='materialized-before-inference; owner approval required',
               archive={'url':'https://www.openslr.org/resources/12/test-clean.tar.gz','bytes':archive.stat().st_size,'sha256':sha(archive),'md5':md5},
               seed='english-parity-v1', selection='SHA256(seed + colon + speakerID), ascending; first 6 dev, next 20 heldout; numeric-text lexical file order within speaker; first 3-10s short; first same-chapter whole-utterance concatenation reaching 20-40s excluding short; 250ms join silence',
               splits={s:{'clips':len(rs),'speakers':len(set(r['speaker'] for r in rs)), 'seconds':sum(r['duration_seconds'] for r in rs),'short':sum(r['kind']=='short' for r in rs),'long':sum(r['kind']=='long' for r in rs)} for s,rs in rows.items()},
               diagnostics={'count':len(diag),'dev':sum(r['split']=='dev' for r in diag),'heldout':sum(r['split']=='heldout' for r in diag)},
               ffmpeg=subprocess.check_output(['ffmpeg','-version'],text=True).splitlines()[0],
               python=sys.version, materializer_sha256=sha(Path(__file__)),
               normalization_sha256=sha(REPO/'benchmarks/lib/transcript_metrics.py'),
               manifests={f:sha(OUT/f) for f in ['dev.jsonl','heldout.jsonl','diagnostics.jsonl']})
save_json(OUT/'materialization.json',summary)
print(json.dumps(summary,indent=2))
