"""Offline, independently checked analysis for the six-model leadership notebook.

This module does not import the inference runner and makes no network/model calls.
"""
from __future__ import annotations
import base64
from collections import defaultdict
import csv
import hashlib
import json
import math
import random
from pathlib import Path
import sqlite3
import statistics
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from IPython.display import HTML, display

MODEL_NAMES={'qwen3-8b-v2':'Qwen3 8B','qwen3-vl-8b':'Qwen3-VL 8B','gemma3-4b':'Gemma 3 4B',
             'qwen3-vl-4b':'Qwen3-VL 4B','llama3.1-8b':'Llama 3.1 8B','ministral3-8b':'Ministral 3 8B'}
FAMILIES={'qwen3-8b-v2':('Alibaba / Qwen','Qwen3'),'qwen3-vl-8b':('Alibaba / Qwen','Qwen3-VL'),
          'qwen3-vl-4b':('Alibaba / Qwen','Qwen3-VL'),'gemma3-4b':('Google','Gemma 3'),
          'llama3.1-8b':('Meta','Llama 3.1'),'ministral3-8b':('Mistral AI','Ministral 3')}
MODES={'text':'Description','text_location':'Description + location','text_photos':'Description + photos',
       'text_location_photos':'Description + location + photos'}
SHORT_MODES={'text':'Text','text_location':'Text + location','text_photos':'Text + photos','text_location_photos':'All inputs'}
NAVY='#182B49'; TEAL='#00827F'; GOLD='#C69214'; BLUE='#3274A1'; GRAY='#65758B'
COLORS=[NAVY,TEAL,GOLD,BLUE,'#8561A8','#B15B50']
plt.rcParams.update({'figure.dpi':125,'savefig.dpi':180,'font.family':'DejaVu Sans','font.size':10.5,
 'axes.titlesize':14,'axes.titleweight':'bold','axes.labelsize':11,'axes.spines.top':False,'axes.spines.right':False,
 'axes.edgecolor':'#C4CED8','axes.labelcolor':NAVY,'text.color':NAVY,'xtick.color':GRAY,'ytick.color':GRAY,
 'grid.color':'#DFE6ED','grid.alpha':.7,'figure.facecolor':'white','axes.facecolor':'white','legend.frameon':False})

def canonical(value):
    return json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False)

def digest(value):
    return hashlib.sha256(value if isinstance(value,bytes) else canonical(value).encode()).hexdigest()

def file_hash(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''): h.update(b)
    return h.hexdigest()

def read_json(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def safe_ratio(a,b): return a/b if b else None

def independent_metrics(records,aliases):
    """Abstentions on positives count as FN; all abstentions fail flag accuracy."""
    tp=sum(r['prediction']['misroute'] is True and r['truth'] for r in records)
    fp=sum(r['prediction']['misroute'] is True and not r['truth'] for r in records)
    fn=sum(r['prediction']['misroute'] is not True and r['truth'] for r in records)
    tn=sum(r['prediction']['misroute'] is False and not r['truth'] for r in records)
    n=len(records)
    result={'completed':n,'weak_tp':tp,'weak_fp':fp,'weak_fn':fn,'weak_tn':tn,
        'weak_flag_n':n,'weak_precision':safe_ratio(tp,tp+fp),'weak_recall':safe_ratio(tp,tp+fn),
        'weak_f1':safe_ratio(2*tp,2*tp+fp+fn),'weak_flag_accuracy':safe_ratio(tp+tn,n),
        'weak_false_positive_rate':safe_ratio(fp,sum(not r['truth'] for r in records)),
        'flag_abstention_rate':safe_ratio(sum(r['prediction']['misroute'] is None for r in records),n),
        'category_abstention_rate':safe_ratio(sum(r['prediction']['recommended_category'] is None for r in records),n)}
    for field,name in [('original','original_agreement'),('current','current_agreement')]:
        usable=[r for r in records if r[field].strip()]
        result[name]=safe_ratio(sum(r['prediction']['recommended_category'] is not None and
            r['prediction']['recommended_category']==aliases.get(r[field],r[field]) for r in usable),len(usable))
    for key in ('input_tokens','output_tokens','latency_seconds','cost_usd'):
        values=[r['usage'][key] for r in records]
        result['mean_'+key]=statistics.mean(values) if values and all(v is not None for v in values) else None
    return result

def read_bundle(bundle):
    bundle=Path(bundle); plan=read_json(bundle/'plan.json')
    assert digest({k:v for k,v in plan.items() if k!='id'})==plan['id'],'Plan hash mismatch'
    assert plan['input_policy']=='blinded' and len(plan['cases'])==100
    cases={c['id']:c for c in plan['cases']}; models={m['id']:m for m in plan['config']['models']}
    assert list(models)==list(MODEL_NAMES),'Unexpected model set/order'
    with (bundle/'data/scoring_labels.csv').open(encoding='utf-8',newline='') as f:
        labels={r['ID']:r for r in csv.DictReader(f)}
    assert set(labels)==set(cases)
    for c in cases.values():
        assert set(c['inputs'])=={'DESCRIPTION','ORIGINAL_TYPE'} and c['weak_misroute'] is None and c['verified']=={}
        assert labels[c['id']]['input_sha256']==c['evaluation_join_hash']
        assert labels[c['id']]['ORIGINAL_TYPE']==c['inputs']['ORIGINAL_TYPE']
        assert labels[c['id']]['label'] in {'0','1'}
    conn=sqlite3.connect(f'file:{(bundle/"checkpoint.sqlite").as_posix()}?mode=ro',uri=True); conn.row_factory=sqlite3.Row
    assert conn.execute('PRAGMA integrity_check').fetchone()[0]=='ok'
    rows=[dict(r) for r in conn.execute('SELECT * FROM predictions')]
    attempts=[dict(r) for r in conn.execute('SELECT id,experiment_id,case_id,started_at,latency_seconds,input_tokens,output_tokens,cost_usd,error FROM attempts ORDER BY id')]
    fingerprints={r['key'].split(':',1)[1]:json.loads(r['value']) for r in conn.execute("SELECT * FROM meta WHERE key LIKE 'model_fingerprint:%'")}
    conn.close()
    groups=defaultdict(list); usage=defaultdict(list)
    for a in attempts: usage[a['experiment_id'],a['case_id']].append(a)
    records=[]
    for row in rows:
        assert row['input_hash']==digest(cases[row['case_id']])
        prompt=plan['prompts'][row['prompt_variant']]
        assert row['prompt_hash']==prompt['hash'] and row['prompt_version']==prompt['version']
        if row['status']!='completed': continue
        p=json.loads(row['prediction_json'])
        assert set(p)=={'misroute','recommended_category','reason'}
        assert type(p['misroute']) in (bool,type(None))
        assert p['recommended_category'] in [None]+plan['categories']
        assert isinstance(p['reason'],str) and p['reason'].strip()
        label=labels[row['case_id']]; values=usage[row['experiment_id'],row['case_id']]
        consumption={k:sum(a[k] for a in values) if values and all(a[k] is not None for a in values) else None
                     for k in ('input_tokens','output_tokens','latency_seconds','cost_usd')}
        r={**row,'prediction':p,'truth':label['label']=='1','original':label['ORIGINAL_TYPE'],'current':label['CURRENT_TYPE'],
           'usage':consumption,'has_photos':bool(cases[row['case_id']]['photos'])}
        groups[row['model_id'],row['modality'],row['prompt_variant']].append(r); records.append(r)
    perf=pd.read_csv(bundle/'report/performance.csv',keep_default_na=True)
    assert len(perf)==48 and len(rows)==4800,'Missing planned experiments/case statuses'
    checked=0
    for _,r in perf.iterrows():
        independent=independent_metrics(groups[r['model'],r['modality'],r['variant']],plan['aliases'])
        for k,v in independent.items():
            if v is None: assert pd.isna(r[k]),(r['model'],k)
            else: assert math.isclose(float(r[k]),v,rel_tol=1e-10,abs_tol=1e-10),(r['model'],r['modality'],r['variant'],k,v,r[k])
            checked+=1
        assert r['verified_flag_n']==0 and r['verified_category_n']==0 and pd.isna(r['verified_category_accuracy'])
    perf['model_name']=perf['model'].map(MODEL_NAMES); perf['input_configuration']=perf['modality'].map(MODES)
    perf['prompt']=perf['variant'].str.title(); perf['total_tokens']=perf.mean_input_tokens+perf.mean_output_tokens
    perf['modalities_n']=perf['modality'].map({'text':1,'text_location':2,'text_photos':2,'text_location_photos':3})
    perf['prompt_rank']=perf['variant'].map({'simple':0,'detailed':1})
    complete=perf[perf.completed.eq(100)&perf.failed.eq(0)].copy()
    best=complete.sort_values(['weak_f1','mean_latency_seconds'],ascending=[False,True]).iloc[0]
    near=complete[complete.weak_f1.ge(best.weak_f1-.01)].copy()
    selected=near.sort_values(['modalities_n','prompt_rank','total_tokens','mean_latency_seconds']).iloc[0]
    result={'bundle':bundle,'plan':plan,'cases':cases,'models':models,'labels':labels,'rows':rows,'attempts':attempts,
            'fingerprints':fingerprints,'groups':groups,'records':records,'performance':perf,'best':best,'selected':selected,
            'metric_values_checked':checked}
    result['comparison_values_checked'],result['bootstrap_intervals_checked']=verify_saved_comparisons(result)
    result['metric_values_checked']+=result['comparison_values_checked']
    return result

def verify_saved_comparisons(data):
    """Check every saved model/modality/prompt delta and prompt F1 bootstrap band."""
    checked=intervals=0; groups=data['groups']; aliases=data['plan']['aliases']
    fields=['weak_precision','weak_recall','weak_f1','weak_flag_accuracy','flag_abstention_rate',
        'category_abstention_rate','original_agreement','current_agreement','mean_input_tokens',
        'mean_output_tokens','mean_latency_seconds','mean_cost_usd']
    def joined(first,second):
        a={r['case_id']:r for r in first}; b={r['case_id']:r for r in second}; ids=sorted(a.keys()&b.keys())
        return [a[i] for i in ids],[b[i] for i in ids]
    def verify(row,left,right):
        nonlocal checked
        assert int(row['paired_n'])==len(left)==len(right)
        a,b=independent_metrics(left,aliases),independent_metrics(right,aliases)
        for key in fields:
            expected=b[key]-a[key] if a[key] is not None and b[key] is not None else None
            actual=row['delta_'+key]
            assert (expected is None and pd.isna(actual)) or (expected is not None and math.isclose(expected,float(actual),abs_tol=1e-10)),(key,expected,actual)
            checked+=1
    for _,r in pd.read_csv(data['bundle']/'report/model_comparisons.csv').iterrows():
        verify(r,*joined(groups[r.baseline_model,r.modality,r.variant],groups[r.comparison_model,r.modality,r.variant]))
    for _,r in pd.read_csv(data['bundle']/'report/modality_differences.csv').iterrows():
        verify(r,*joined(groups[r.model,r.baseline,r.variant],groups[r.model,r.comparison,r.variant]))
    iterations=read_json(data['bundle']/'report/report.json')['policy']['bootstrap_iterations']
    for _,r in pd.read_csv(data['bundle']/'report/paired_differences.csv').iterrows():
        left,right=joined(groups[r.model,r.modality,'simple'],groups[r.model,r.modality,'detailed'])
        verify(r,left,right)
        if not left:
            assert pd.isna(r.delta_weak_f1_ci_low) and pd.isna(r.delta_weak_f1_ci_high)
            continue
        def outcome(record):
            p,y=record['prediction']['misroute'],record['truth']
            return [int(p is True and y),int(p is True and not y),int(p is not True and y)]
        a=np.array([outcome(x) for x in left]); b=np.array([outcome(x) for x in right])
        seed=int(digest([data['plan']['seed'],r.model,r.modality,'weak_f1'])[:8],16)
        rng=random.Random(seed)
        indices=np.array([rng.choices(range(len(left)),k=len(left)) for _ in range(iterations)])
        def f1(samples):
            counts=samples[indices].sum(axis=1); denominator=2*counts[:,0]+counts[:,1]+counts[:,2]
            return np.divide(2*counts[:,0],denominator,out=np.full(len(counts),np.nan),where=denominator!=0)
        deltas=f1(b)-f1(a); values=sorted(x for x in deltas if np.isfinite(x))
        if len(values)<iterations*.9:
            assert pd.isna(r.delta_weak_f1_ci_low) and pd.isna(r.delta_weak_f1_ci_high)
        else:
            low=values[int((len(values)-1)*.025)]; high=values[int((len(values)-1)*.975)]
            assert math.isclose(low,r.delta_weak_f1_ci_low,abs_tol=1e-10) and math.isclose(high,r.delta_weak_f1_ci_high,abs_tol=1e-10)
        intervals+=1
    return checked,intervals

def audit_requests(data):
    """Verify every actual model request/response and imported raw row, not just CSVs."""
    b=data['bundle']; plan=data['plan']; cases=data['cases']; rows={(r['experiment_id'],r['case_id']):r for r in data['rows']}
    inputs_manifest=read_json(b/'data/source_manifest.json')
    assert digest(read_json(b/'config/service_definitions.json'))==plan['reference_sha256']
    assert read_json(b/'config/output_schema.json')==plan['schema']
    assert file_hash(b/'data/scoring_labels.csv')==inputs_manifest['scoring_labels_sha256']
    assert file_hash(b/'data/blinded_cases.jsonl')==inputs_manifest['blinded_cases_sha256']
    assert [json.loads(line) for line in (b/'data/blinded_cases.jsonl').read_text(encoding='utf-8').splitlines()]==plan['cases']
    original_labels=Path(inputs_manifest['original_evaluation_source'])
    if original_labels.exists():
        assert file_hash(original_labels)==inputs_manifest['original_evaluation_sha256']
        with original_labels.open(encoding='utf-8-sig',newline='') as f: original={r['ID']:r for r in csv.DictReader(f)}
        for case_id,label in data['labels'].items():
            assert all(label[k]==original[case_id].get(k,'') for k in ('ORIGINAL_TYPE','CURRENT_TYPE','label','LABEL_PROVENANCE'))
    photos=read_json(b/'assets/photo_ownership.json'); ownership={p['source']:p for p in photos}
    for c in cases.values():
        for p in c['photos']:
            o=ownership[p['source']]
            assert o['case_id']==c['id'] and o['case_ownership_verified']
            assert file_hash(b/o['bundle_path'])==p['sha256']==o['sha256']
    conn=sqlite3.connect(f'file:{(b/"checkpoint.sqlite").as_posix()}?mode=ro',uri=True); conn.row_factory=sqlite3.Row
    source_predictions={tuple(r[k] for k in ('experiment_id','case_id')):dict(r) for r in conn.execute('SELECT * FROM prediction_origins')}
    source_attempts={r['id']:dict(r) for r in conn.execute('SELECT * FROM attempt_origins')}
    imported_n=0
    for key,o in source_predictions.items():
        reconstructed=dict(rows[key]); reconstructed['experiment_id']=o['source_experiment_id']
        assert digest(reconstructed)==o['source_row_hash'],'Imported prediction changed'
        imported_n+=1
    attempts_checked=images_checked=imported_attempts_checked=0; success={}; errors=[]
    resources=defaultdict(lambda:{'attempts':0,'request_seconds':0.,'input_tokens':0,'output_tokens':0,'cached_tokens':0,'load_seconds':0.,'prompt_seconds':0.,'decode_seconds':0.})
    for raw_row in conn.execute('SELECT * FROM attempts ORDER BY id'):
        a=dict(raw_row); key=a['experiment_id'],a['case_id']; r=rows[key]; c=cases[r['case_id']]
        request=json.loads(a['request_json']); user=json.loads(request['user']); prompt=plan['prompts'][r['prompt_variant']]
        assert digest(request)==a['request_hash'] and request['system']==prompt['system'] and request['schema']==plan['schema']
        allowed={'DESCRIPTION','ORIGINAL_TYPE'}|({'LOCATION'} if 'location' in r['modality'] else set())|({'PHOTOS'} if 'photos' in r['modality'] else set())
        assert set(user)==allowed and all(user[k]==v for k,v in c['inputs'].items())
        if 'LOCATION' in user: assert user['LOCATION']==c['location']
        if 'PHOTOS' in user: assert user['PHOTOS']=={'available':len(c['photos']),'status':'provided' if c['photos'] else 'not_available'}
        expected=c['photos'] if 'photos' in r['modality'] else []
        assert len(request['images'])==len(expected)
        for encoded,p in zip(request['images'],expected):
            assert digest(base64.b64decode(encoded))==p['sha256']; images_checked+=1
        response=json.loads(a['raw_response'])
        assert response['model']==data['models'][r['model_id']]['model'] and not response.get('synthetic')
        assert response.get('created_at') and response['total_duration']>0 and response['done']
        if a['error'] is None:
            assert response.get('done_reason')!='length'
            success[key]=json.loads(response['message']['content'])
        else: errors.append({'model':r['model_id'],'modality':r['modality'],'prompt':r['prompt_variant'],'case_id':r['case_id'],'error':a['error'],'attempt_id':a['id']})
        if a['id'] in source_attempts:
            original=dict(a); original['experiment_id']=source_attempts[a['id']]['source_experiment_id']; original['id']=source_attempts[a['id']]['source_id']
            assert digest(original)==source_attempts[a['id']]['source_row_hash']; imported_attempts_checked+=1
        assert a['cost_usd']==0
        u=resources[r['model_id']]; u['attempts']+=1; u['request_seconds']+=a['latency_seconds']
        for k in ('input_tokens','output_tokens'): u[k]+=a[k] or 0
        u['cached_tokens']+=response.get('prompt_eval_cached_count',0)
        for raw_key,name in [('load_duration','load_seconds'),('prompt_eval_duration','prompt_seconds'),('eval_duration','decode_seconds')]: u[name]+=response.get(raw_key,0)/1e9
        attempts_checked+=1
    conn.close()
    assert all(json.loads(r['prediction_json'])==success[k] for k,r in rows.items() if r['status']=='completed')
    parent=json.loads(json.dumps(plan)); parent.pop('id'); parent.pop('parent_plan_id'); parent['config']['models']=parent['config']['models'][:4]
    assert digest(parent)==plan['parent_plan_id'],'Frozen parent cohort or prompts changed'
    source_manifest=read_json(b/'provenance/source_files.json'); source=Path(read_json(b/'provenance/import.json')['source'])
    files_checked=0
    if source.exists():
        for relative,h in source_manifest.items(): assert file_hash(source/relative)==h; files_checked+=1
    result={'audit_passed':True,'schema_valid_completed_predictions':len(success),'attempts_checked':attempts_checked,
        'case_owned_image_payloads_checked':images_checked,'imported_prediction_rows_unchanged':imported_n,
        'imported_attempt_rows_unchanged':imported_attempts_checked,'prior_archive_files_unchanged':files_checked,
        'independent_metric_values_checked':data['metric_values_checked'],'input_allowlist_and_common_schema_valid':True,
        'paired_comparison_values_checked':data['comparison_values_checked'],'paired_prompt_bootstrap_intervals_checked':data['bootstrap_intervals_checked'],
        'same_100_cases_and_prompts_preserved':True,'errors':errors,'paid_api_calls':0,'verified_accuracy_available':False}
    data['resources']=pd.DataFrame([{'model':k,'model_name':MODEL_NAMES[k],**v} for k,v in resources.items()])
    data['audit']=result
    (b/'validation').mkdir(exist_ok=True)
    (b/'validation/independent_audit.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    return result

def pretty_table(frame,formats=None):
    style=frame.style.hide(axis='index').format(formats or {},na_rep='—')
    style=style.set_table_styles([
        {'selector':'th','props':[('background-color',NAVY),('color','white'),('padding','10px 12px'),('text-align','left'),('font-size','12px')]},
        {'selector':'td','props':[('padding','9px 12px'),('border-bottom','1px solid #E1E8EF'),('font-size','12px')]},
        {'selector':'tbody tr:nth-child(even)','props':[('background-color','#F3F6F9')]},
        {'selector':'table','props':[('border-collapse','collapse'),('width','100%')]}])
    display(style)

def save_figure(data,fig,name):
    out=data['bundle']/'figures'; out.mkdir(exist_ok=True)
    for suffix in ('png','svg'): fig.savefig(out/f'{name}.{suffix}',bbox_inches='tight',facecolor='white')
    plt.show(); plt.close(fig)

def executive_summary(data):
    s=data['selected']; best=data['best']; p=data['performance']; statuses=pd.Series([r['status'] for r in data['rows']]).value_counts()
    cards=[('Case cohort',str(len(data['cases'])),'Identical cases across every experiment'),
           ('Local models',str(len(data['models'])),'Two prompts × four input configurations'),
           ('Valid predictions',f"{statuses.get('completed',0):,}",f"{statuses.get('failed',0)} failed; {statuses.get('skipped',0)} unsupported skips"),
           ('Paid API charges','$0','Hardware and energy costs not measured')]
    html='<div class="metric-grid">'+''.join(f'<div class="metric-card"><div class="metric-label">{a}</div><div class="metric-value">{v}</div><div class="metric-note">{n}</div></div>' for a,v,n in cards)+'</div>'
    html+=f'<div class="finding"><b>Provisional configuration:</b> {s.model_name} · {s.prompt} prompt · {s.input_configuration.lower()}. Weak-label F1 <b>{s.weak_f1:.3f}</b>, precision {s.weak_precision:.3f}, recall {s.weak_recall:.3f}, mean request latency {s.mean_latency_seconds:.2f} seconds.</div>'
    html+=f'<p>The highest fully completed score is <b>{best.weak_f1:.3f}</b> ({best.model_name}, {best.prompt.lower()}, {best.input_configuration.lower()}). Configurations within one percentage point of that score are treated as practically tied; the recommendation favors fewer input modalities, simpler prompts, then fewer reported tokens. This is an explicit <b>post-hoc presentation heuristic</b>, not a preregistered selection rule.</p>'
    html+='<div class="caveat"><b>Interpretation boundary:</b> All performance scores use weak development labels. Earlier Qwen3 screening influenced cohort/label construction and can favor Qwen-family agreement. Historical category agreement is not verified routing accuracy. Independent adjudication is required before deployment decisions.</div>'
    display(HTML(html))

def model_overview(data):
    runtimes={}
    for path,keys in [(data['bundle']/'provenance/existing_four/model_execution.json',['runtime']),
                      (data['bundle']/'runtime/new_models_execution.json',['full_100_runtime','smoke_10_runtime'])]:
        if path.exists():
            for record in read_json(path):
                for key in keys:
                    models=record.get(key,{}).get('models',[])
                    if models: runtimes[record['model']]=models[0]; break
    result=[]; perf=data['performance']
    for key,m in data['models'].items():
        runtime=runtimes.get(key,{}); details=runtime.get('details',{}); rows=perf[perf.model.eq(key)]
        result.append({'Model':MODEL_NAMES[key],'Developer':FAMILIES[key][0],'Architecture':FAMILIES[key][1],
            'Reported size':details.get('parameter_size','See pinned tag'),'Quantization':details.get('quantization_level','Q4_K_M'),
            'Vision':'Yes' if m['supports_photos'] else 'No','Valid predictions':int(rows.completed.sum()),
            'Full configurations':int(rows.completed.eq(100).sum()),'Failed cases':int(rows.failed.sum()),
            'Digest':data['fingerprints'].get(key,{}).get('digest','')[:12]})
    return pd.DataFrame(result)

def metric_table(frame):
    columns={'model_name':'Model','input_configuration':'Inputs','prompt':'Prompt','completed':'N','weak_precision':'Precision',
        'weak_recall':'Recall','weak_f1':'F1','weak_fp':'False positives','flag_abstention_rate':'Flag abstention',
        'original_agreement':'Original agreement','current_agreement':'Current agreement','mean_latency_seconds':'Latency (s)','total_tokens':'Reported tokens'}
    return frame[list(columns)].rename(columns=columns)

def prompt_overview(data):
    rows=[]
    for variant,p in data['plan']['prompts'].items():
        rows.append({'Prompt':variant.title(),'Version':p['version'],'System characters':len(p['system']),
            'Whitespace words':len(p['system'].split()),'System SHA-256':p['hash'][:16]})
    return pd.DataFrame(rows)

def common_text_ranking(data):
    frame=data['performance'].query("modality=='text' and variant=='simple'").sort_values('weak_f1',ascending=False)
    fig,ax=plt.subplots(figsize=(10.5,5.2),layout='constrained')
    bars=ax.barh(frame.model_name,frame.weak_f1,color=[COLORS[list(MODEL_NAMES).index(x)] for x in frame.model],height=.62)
    for bar,value in zip(bars,frame.weak_f1): ax.text(value+.012,bar.get_y()+bar.get_height()/2,f'{value:.3f}',va='center',fontweight='bold')
    ax.invert_yaxis(); ax.set_xlim(0,1); ax.set_xlabel('Weak-label misroute F1'); ax.set_title('A common baseline: simple prompt, description inputs')
    ax.grid(axis='x'); ax.set_axisbelow(True)
    save_figure(data,fig,'01_common_text_ranking')
    return frame

def score_heatmap(data):
    p=data['performance']; modes=list(MODES); keys=[(v,m) for v in ('simple','detailed') for m in modes]
    arr=np.full((6,8),np.nan); annotations={}
    for i,model in enumerate(MODEL_NAMES):
        for j,(variant,mode) in enumerate(keys):
            r=p[(p.model==model)&(p.variant==variant)&(p.modality==mode)].iloc[0]
            if r.completed:
                arr[i,j]=r.weak_f1; annotations[i,j]=f'{r.weak_f1:.3f}'+('†' if r.completed!=100 else '')
            else: annotations[i,j]='Unsupported' if r.skipped else 'No result'
    fig,ax=plt.subplots(figsize=(13.8,5),layout='constrained'); cmap=plt.get_cmap('YlGnBu').copy(); cmap.set_bad('#E7EBF0')
    image=ax.imshow(np.ma.masked_invalid(arr),cmap=cmap,vmin=0,vmax=1,aspect='auto')
    for (i,j),text in annotations.items(): ax.text(j,i,text,ha='center',va='center',fontsize=9,color='white' if arr[i,j]>.68 else NAVY)
    ax.set_yticks(range(6),MODEL_NAMES.values()); ax.set_xticks(range(8),[f'{v.title()}\n{SHORT_MODES[m]}' for v,m in keys],fontsize=9)
    ax.axvline(3.5,color='white',linewidth=4); ax.set_title('Complete experiment matrix | weak-label F1')
    fig.colorbar(image,ax=ax,label='F1',shrink=.8,pad=.02)
    save_figure(data,fig,'02_all_configurations_heatmap')

def prompt_cost_chart(data):
    p=data['performance'].query("modality=='text'")
    fig,axes=plt.subplots(1,2,figsize=(13,4.7),layout='constrained'); y=np.arange(6)
    for ax,field,title in zip(axes,['total_tokens','mean_latency_seconds'],['Reported tokens per completed case','Mean request latency (seconds)']):
        for offset,variant,color in [(-.17,'simple',TEAL),(.17,'detailed',GOLD)]:
            values=[p[(p.model==m)&(p.variant==variant)].iloc[0][field] for m in MODEL_NAMES]
            ax.barh(y+offset,values,height=.31,color=color,label=variant.title())
        ax.set_yticks(y,MODEL_NAMES.values()); ax.invert_yaxis(); ax.set_title(title); ax.grid(axis='x'); ax.set_axisbelow(True)
        ax.legend(loc='upper center',bbox_to_anchor=(.5,-.08),ncol=2)
    fig.suptitle('The cost of longer instructions | description inputs',fontweight='bold',fontsize=16)
    save_figure(data,fig,'03_prompt_cost')

def agreement_chart(data):
    p=data['performance'].query("modality=='text'"); fig,axes=plt.subplots(1,2,figsize=(13,4.8),layout='constrained'); y=np.arange(6)
    for ax,field,title in zip(axes,['original_agreement','current_agreement'],['Agreement with original category','Agreement with current / historical category']):
        for offset,variant,color in [(-.17,'simple',TEAL),(.17,'detailed',GOLD)]:
            values=[p[(p.model==m)&(p.variant==variant)].iloc[0][field] for m in MODEL_NAMES]
            ax.barh(y+offset,values,height=.31,color=color,label=variant.title())
        ax.set_yticks(y,MODEL_NAMES.values()); ax.invert_yaxis(); ax.set_xlim(0,1); ax.set_title(title); ax.grid(axis='x'); ax.set_axisbelow(True)
        ax.legend(loc='upper center',bbox_to_anchor=(.5,-.08),ncol=2)
    fig.suptitle('Category recommendation is a separate task | agreement, not verified accuracy',fontsize=14,fontweight='bold')
    save_figure(data,fig,'04_category_agreement')

def bootstrap_difference(left,right,iterations=1000,seed=42):
    """Identical-case paired percentile bootstrap; both sides resampled together."""
    left={r['case_id']:r for r in left}; right={r['case_id']:r for r in right}; ids=sorted(left.keys()&right.keys())
    if not ids: return {'paired_n':0,'delta_f1':np.nan,'ci_low':np.nan,'ci_high':np.nan}
    def counts(r):
        p,y=r['prediction']['misroute'],r['truth']
        return [int(p is True and y),int(p is True and not y),int(p is not True and y)]
    a=np.array([counts(left[i]) for i in ids]); b=np.array([counts(right[i]) for i in ids])
    def score(c):
        denominator=2*c[...,0]+c[...,1]+c[...,2]
        return np.divide(2*c[...,0],denominator,out=np.full_like(denominator,np.nan,dtype=float),where=denominator!=0)
    rng=np.random.default_rng(seed); samples=rng.integers(0,len(ids),size=(iterations,len(ids)))
    deltas=score(b[samples].sum(axis=1))-score(a[samples].sum(axis=1))
    low,high=np.nanpercentile(deltas,[2.5,97.5])
    return {'paired_n':len(ids),'delta_f1':float(score(b.sum(axis=0))-score(a.sum(axis=0))),'ci_low':float(low),'ci_high':float(high)}

def ablations(data):
    transitions=[('Location added','text','text_location'),('Photos added','text','text_photos'),
                 ('Photos with location','text_location','text_location_photos'),('Location with photos','text_photos','text_location_photos')]
    rows=[]
    for model in MODEL_NAMES:
        for variant in ('simple','detailed'):
            for label,a,b in transitions:
                first=data['groups'][model,a,variant]; second=data['groups'][model,b,variant]
                if not first or not second: continue
                seed=int(digest([model,variant,label])[:8],16)
                rows.append({'model':model,'model_name':MODEL_NAMES[model],'prompt':variant.title(),'effect':label,
                    'baseline':a,'comparison':b,**bootstrap_difference(first,second,seed=seed)})
    frame=pd.DataFrame(rows); out=data['bundle']/'analysis'; out.mkdir(exist_ok=True)
    frame.to_csv(out/'paired_modality_ablations.csv',index=False)
    photo_rows=[]
    for model in MODEL_NAMES:
        for variant in ('simple','detailed'):
            first=[r for r in data['groups'][model,'text',variant] if r['has_photos']]
            second=[r for r in data['groups'][model,'text_photos',variant] if r['has_photos']]
            if second: photo_rows.append({'Model':MODEL_NAMES[model],'Prompt':variant.title(),**bootstrap_difference(first,second)})
    pd.DataFrame(photo_rows).to_csv(out/'photo_available_subset.csv',index=False)
    return frame,pd.DataFrame(photo_rows)

def ablation_chart(data,frame):
    effects=['Location added','Photos added','Photos with location','Location with photos']; fig,axes=plt.subplots(1,2,figsize=(14,5),layout='constrained')
    for ax,variant in zip(axes,['Simple','Detailed']):
        arr=np.full((6,4),np.nan)
        for i,m in enumerate(MODEL_NAMES):
            for j,e in enumerate(effects):
                row=frame[(frame.model==m)&(frame.prompt==variant)&(frame.effect==e)]
                if len(row): arr[i,j]=row.iloc[0].delta_f1
        cmap=plt.get_cmap('RdYlGn').copy(); cmap.set_bad('#E7EBF0')
        image=ax.imshow(np.ma.masked_invalid(arr),cmap=cmap,norm=TwoSlopeNorm(vmin=-.3,vcenter=0,vmax=.3),aspect='auto')
        for i in range(6):
            for j in range(4): ax.text(j,i,'—' if np.isnan(arr[i,j]) else f'{arr[i,j]:+.3f}',ha='center',va='center',fontsize=9)
        ax.set_yticks(range(6),MODEL_NAMES.values()); ax.set_xticks(range(4),['+ location\n(text baseline)','+ photos\n(text baseline)','+ photos\n(with location)','+ location\n(with photos)'],fontsize=8)
        ax.set_title(f'{variant} prompt')
    fig.colorbar(image,ax=axes,label='Paired F1 change',shrink=.85,pad=.02)
    fig.suptitle('Input ablations | changes on identical completed cases',fontweight='bold',fontsize=16)
    save_figure(data,fig,'05_modality_ablations')

def prompt_difference_chart(data):
    paired=pd.read_csv(data['bundle']/'report/paired_differences.csv'); fig,axes=plt.subplots(1,4,figsize=(14,4.8),sharey=True,layout='constrained')
    for ax,(mode,title) in zip(axes,SHORT_MODES.items()):
        for i,m in enumerate(MODEL_NAMES):
            r=paired[(paired.model==m)&(paired.modality==mode)].iloc[0]
            if r.paired_n:
                color=TEAL if r.delta_weak_f1>=0 else '#B15B50'
                ax.hlines(i,r.delta_weak_f1_ci_low,r.delta_weak_f1_ci_high,color=color,lw=2)
                ax.plot(r.delta_weak_f1,i,'o',color=color,ms=6)
        ax.axvline(0,color=GRAY,lw=1); ax.set_title(title,fontsize=11); ax.set_xlim(-.4,.35); ax.set_xlabel('Detailed − simple F1'); ax.grid(axis='x')
        ax.set_yticks(range(6),MODEL_NAMES.values())
    axes[0].invert_yaxis(); fig.suptitle('Longer instructions | paired 95% bootstrap intervals',fontweight='bold',fontsize=16)
    save_figure(data,fig,'06_prompt_differences')
    return paired[paired.paired_n.gt(0)]

def resource_chart(data):
    p=data['performance']; completed=p[p.completed.gt(0)]
    fig,ax=plt.subplots(figsize=(11,6),layout='constrained')
    for i,m in enumerate(MODEL_NAMES):
        for variant,marker in [('simple','o'),('detailed','^')]:
            rows=completed[(completed.model==m)&(completed.variant==variant)]
            ax.scatter(rows.mean_latency_seconds,rows.weak_f1,color=COLORS[i],marker=marker,s=64,alpha=.8,
                       label=MODEL_NAMES[m] if variant=='simple' else None,edgecolors='white',linewidths=.8)
    s=data['selected']; ax.scatter([s.mean_latency_seconds],[s.weak_f1],marker='*',s=260,facecolors='none',edgecolors=NAVY,linewidth=1.8,zorder=4)
    ax.annotate('Provisional selection',(s.mean_latency_seconds,s.weak_f1),xytext=(22,-25),textcoords='offset points',fontsize=10,
                arrowprops={'arrowstyle':'-','color':GRAY})
    ax.set_xlabel('Mean request latency per completed case (seconds)'); ax.set_ylabel('Weak-label misroute F1')
    ax.set_title('Performance versus local inference time'); ax.grid(); ax.set_axisbelow(True)
    ax.legend(loc='upper center',bbox_to_anchor=(.5,-.15),ncol=3,fontsize=9)
    fig.text(.5,-.04,'Circles: simple prompt. Triangles: detailed prompt. Each point is an input configuration; the partial configuration is included.',ha='center',fontsize=9,color=GRAY)
    save_figure(data,fig,'07_latency_tradeoff')

def findings(data,paired,effects):
    comparisons=paired[paired.paired_n.gt(0)]; photos=effects[effects.effect.eq('Photos added')]
    n=len(comparisons); up=int(comparisons.delta_weak_f1.gt(0).sum()); positive=int(comparisons.delta_weak_f1_ci_low.gt(0).sum())
    s=data['selected']; best=data['best']
    statements=[f'<b>Provisional choice:</b> {s.model_name}, {s.prompt.lower()} prompt, {s.input_configuration.lower()} (F1 {s.weak_f1:.3f}). This uses the simplest configuration within one percentage point of the best full-batch score ({best.weak_f1:.3f}).',
        f'<b>Longer instructions:</b> Detailed prompts increased F1 in {up} of {n} paired comparisons. {positive} had a strictly positive lower 95% bootstrap bound. These exploratory intervals are not corrected for multiple comparisons.',
        f'<b>Photographs:</b> Adding photos to description inputs increased F1 in {int(photos.delta_f1.gt(0).sum())} of {len(photos)} supported model/prompt comparisons. Read the photo-available subset separately; missing photos were never substituted or resampled.',
        '<b>Task distinction:</b> Category agreement and misroute detection measure different behavior. An increase in historical-category agreement does not establish better misroute detection or verified category accuracy.',
        '<b>Why effects can differ:</b> Detailed definitions change category boundaries and overlap handling. Labels were constructed using text/historical screening, so useful visual evidence could disagree with a weak target; photograph timing is also unknown. These are plausible interpretations, not isolated causes. Lower agreement does not establish that photographs harm real routing accuracy. Location inputs were coordinates only, with no GIS lookup.',
        '<b>Operational implication:</b> Use the result to prioritize independent, blinded human adjudication and review workload analysis. No configuration is validated for automatic routing by this weak-label study.']
    display(HTML('<ol class="findings">'+''.join('<li>'+s+'</li>' for s in statements)+'</ol>'))
