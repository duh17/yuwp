import collections,hashlib,json,pathlib,sys
sys.path.insert(0,str(pathlib.Path.cwd()))
from benchmarks.lib.english_parity import summarize_trials, distribution
root=pathlib.Path('/tmp/yuwp-english-parity')
source=root/'dev-run.jsonl'
records=[json.loads(x) for x in source.read_text().splitlines() if x.strip()]
assert records[-1]['type']=='summary','do not summarize incomplete run as final'
rows=[r for r in records if r.get('type')=='trial']
def stratum(r):
 m=r.get('metadata',{})
 if m.get('kind')=='spontaneous':return 'AMI-spontaneous-9'
 if not m.get('diagnostic'):return 'LibriSpeech-dev-12'
 return 'synthetic-diagnostics-6'
result={'label':'NON-ACCEPTANCE','uncertainty':'No confidence/parity inference: single development pass under uncontrolled load; synthetic repeats are not independent. Undefined endpoints and failures retained.','source_sha256':hashlib.file_digest(source.open('rb'),'sha256').hexdigest(),'strata':{},'failed_trials':[]}
for name in ('LibriSpeech-dev-12','AMI-spontaneous-9','synthetic-diagnostics-6'):
 sub=[r for r in rows if stratum(r)==name]
 s=summarize_trials(sub)
 for model in ('Qwen','Nemotron','Parakeet'):
  rr=[r for r in sub if r['system']==model];a=s['systems'][model]
  a['planned']=12 if name.startswith('Libri') else 9 if name.startswith('AMI') else 6
  a['attempted']=sum(r['status']!='not-run' for r in rr)
  a['missing']=a['planned']-len(rr)+sum(r['status']=='not-run' for r in rr)
  available=[r for r in rr if r.get('metrics',{}).get('scores') is not None]
  a['available_final_including_failed']={'n':len(available),'errors':sum(r['metrics']['scores']['errors'] for r in available),'reference_words':sum(r['metrics']['scores']['reference_units'] for r in available),'failed_final_count':sum(r['status']!='ok' for r in available)}
 result['strata'][name]=s
for r in rows:
 if r['status']!='ok':result['failed_trials'].append({'system':r['system'],'case_id':r['case_id'],'status':r['status'],'error':r.get('error'),'hypothesis':r.get('metrics',{}).get('hypothesis'),'metrics':r.get('metrics'),'stderr':r.get('stderr_path')})
(root/'stratified-summary.json').write_text(json.dumps(result,indent=2)+'\n')
for name,s in result['strata'].items():
 print(name)
 for model,a in s['systems'].items():
  print(model,'counts',a['planned'],a['attempted'],a['failed'],a['missing'],'WER',a['quality']['micro_wer'],'words',a['quality']['reference_units'])
  for k in ('first_nonempty_s','first_useful_s','stable_useful_s','stop_to_final_s'):print(' ',k,a['latencies'][k])
