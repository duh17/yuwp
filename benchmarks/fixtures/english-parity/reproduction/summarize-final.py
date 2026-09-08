import hashlib,json,pathlib,sys
sys.path.insert(0,str(pathlib.Path.cwd()))
from benchmarks.lib.english_parity import summarize_trials
root=pathlib.Path('/tmp/yuwp-english-parity')
def load(name,count):
 p=root/name;data=[json.loads(x) for x in p.read_text().splitlines()]
 assert data[-1]['type']=='summary'
 rows=[x for x in data if x.get('type')=='trial'];assert len(rows)==count
 return p,rows
p,original=load('dev-run.jsonl',81);q,repaired=load('parakeet-retry.jsonl',27)
rows=[x for x in original if x['system']!='Parakeet']+repaired
result={'label':'NON-ACCEPTANCE','uncertainty':'No confirmation or inferential confidence intervals: development-only, uncontrolled load, sparse samples and missing useful-text endpoints. P1 repaired pass is noninterleaved; no causal/parity latency comparison to earlier systems.','original_trials':81,'original_P1_failures':sum(x['status']!='ok' for x in original if x['system']=='Parakeet'),'supplemental_trials':27,'artifacts':{str(x):hashlib.file_digest(x.open('rb'),'sha256').hexdigest() for x in (p,q)},'strata':{},'empty_speech':[],'failed_trials':[]}
def group(r):
 m=r['metadata']
 return 'AMI-spontaneous-9' if m['kind']=='spontaneous' else 'LibriSpeech-dev-12' if not m['diagnostic'] else 'synthetic-diagnostics-6'
for name in ('LibriSpeech-dev-12','AMI-spontaneous-9','synthetic-diagnostics-6'):
 s=summarize_trials([r for r in rows if group(r)==name]);s['paired']['Parakeet']['latency_comparison_valid']=False
 s['paired']['Parakeet']['latency_comparison_reason']='noninterleaved uncontrolled supplemental pass; these arithmetic deltas are not a causal/parity result'
 for model in ('Qwen','Nemotron','Parakeet'):
  rr=[r for r in rows if group(r)==name and r['system']==model];a=s['systems'][model]
  a['planned']=12 if name.startswith('Libri') else 9 if name.startswith('AMI') else 6
  a['attempted']=sum(r['status']!='not-run' for r in rr);a['missing']=a['planned']-a['attempted']
  available=[r for r in rr if r.get('metrics',{}).get('scores') is not None]
  a['available_finals_including_failed']={'n':len(available),'errors':sum(r['metrics']['scores']['errors'] for r in available),'reference_words':sum(r['metrics']['scores']['reference_units'] for r in available),'failed_final_count':sum(r['status']!='ok' for r in available)}
 result['strata'][name]=s
for r in rows:
 m=r.get('metrics',{})
 if m.get('lexical_eligible') and m.get('hypothesis')=='':result['empty_speech'].append({'system':r['system'],'case_id':r['case_id'],'reference':r['reference'],'scores':m['scores'],'transport_status':r['status']})
 if r['status']!='ok':result['failed_trials'].append({'system':r['system'],'case_id':r['case_id'],'error':r.get('error'),'metrics':m,'stderr':r.get('stderr_path')})
(root/'combined-dev-summary.json').write_text(json.dumps(result,indent=2)+'\n')
for n,s in result['strata'].items():
 print(n)
 for model,a in s['systems'].items():
  print(model,'planned/attempted/failed/missing',a['planned'],a['attempted'],a['failed'],a['missing'],'quality',a['quality'])
  for k in ('first_nonempty_s','first_useful_s','stable_useful_s','stop_to_final_s'):print(k,a['latencies'][k])
 print('quality deltas',{k:v['micro_wer_delta'] for k,v in s['paired'].items()})
print('empty speech',result['empty_speech']);print('failed trials',[(r['system'],r['case_id'],r['error']) for r in result['failed_trials']])
