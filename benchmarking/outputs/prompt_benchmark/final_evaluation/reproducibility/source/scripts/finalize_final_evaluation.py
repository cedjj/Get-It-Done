"""Regenerate consolidated reports and execute/export the final offline notebook."""
import argparse
import asyncio
import json
import os
import re
from pathlib import Path
import shutil
import time
import sys
import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter
from traitlets.config import Config
from consolidate_final_benchmark import FINAL, file_hash
from project_paths import ROOT
from prompt_benchmark.core import load_plan, write_json, now
from prompt_benchmark.evaluate import evaluate
from build_final_evaluation_notebook import build

def finalize():
    start=time.perf_counter(); plan=load_plan(FINAL)
    for variable,folder in [('IPYTHONDIR','notebook_ipython'),('JUPYTER_RUNTIME_DIR','notebook_jupyter'),('MPLCONFIGDIR','matplotlib')]:
        path=FINAL/'runtime'/folder; path.mkdir(parents=True,exist_ok=True); os.environ[variable]=str(path)
    if sys.platform=='win32': asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    print('Generating consolidated metrics and paired comparisons...',flush=True)
    evaluate(plan,FINAL,evaluation_path=FINAL/'data/scoring_labels.csv',raw_export=False)
    # The SQLite checkpoint is the authoritative raw audit. No duplicate large JSONL export.
    snapshots=FINAL/'reproducibility/source'; snapshots.mkdir(parents=True,exist_ok=True)
    sources=[ROOT/'scripts'/name for name in ['final_benchmark_analysis.py','build_final_evaluation_notebook.py',
        'consolidate_final_benchmark.py','execute_final_benchmark.py','execute_multimodal_benchmark.py',
        'finalize_final_evaluation.py','prepare_final_assets.py','download_benchmark_models.py',
        'run_prompt_benchmark.py','project_paths.py','clean_final_evaluation_caches.py']]
    sources+=list((ROOT/'scripts/prompt_benchmark').rglob('*.py'))
    sources+=list((ROOT/'scripts/prompt_benchmark/templates').glob('*blinded_v2.txt'))
    sources+=list((ROOT/'tests').glob('test*benchmark*.py'))
    sources.append(ROOT/'results/labeling_summary.md')
    sources.append(ROOT/'semantic_labeling.py')
    manifest={}
    for source in sources:
        relative=source.relative_to(ROOT); target=snapshots/relative
        target.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(source,target)
        manifest[str(relative).replace('\\','/')]=file_hash(target)
    write_json(FINAL/'reproducibility/source_hashes.json',manifest)
    shutil.copy2(ROOT/'scripts/final_benchmark_analysis.py',FINAL/'reproducibility/final_benchmark_analysis.py')
    notebook=build()
    print('Executing the complete notebook offline...',flush=True)
    nb=nbformat.read(notebook,as_version=4)
    def completed(cell,cell_index,**kwargs):
        print(f'Executed notebook cell {cell_index+1}/{len(nb.cells)}',flush=True)
        nbformat.write(nb,notebook)
    client=NotebookClient(nb,timeout=600,kernel_name='python3',resources={'metadata':{'path':str(notebook.parent)}},on_cell_executed=completed)
    try:
        client.execute()
    finally:
        # Preserve partial outputs if a real execution error needs diagnosis.
        nbformat.write(nb,notebook)
    code_cells=[c for c in nb.cells if c.cell_type=='code']
    assert all(c.execution_count is not None for c in code_cells)
    errors=[o for c in code_cells for o in c.outputs if o.output_type=='error']
    assert not errors,errors
    assert [c.execution_count for c in code_cells]==list(range(1,len(code_cells)+1))
    config=Config(); config.HTMLExporter.exclude_input=True; config.HTMLExporter.exclude_input_prompt=True; config.HTMLExporter.exclude_output_prompt=True
    config.HTMLExporter.mathjax_url=''
    config.HTMLExporter.require_js_url=''
    exporter=HTMLExporter(config=config,template_name='lab')
    body,_=exporter.from_notebook_node(nb)
    # All plots are embedded static images; native details disclosures need no JS.
    # Remove unused template runtimes, including empty-src tags, for offline sharing.
    body=re.sub(r'<script\b[^>]*>[\s\S]*?</script\s*>','',body,flags=re.IGNORECASE)
    assert not re.search(r'<script\b',body,re.IGNORECASE),'Unexpected presentation script'
    html=notebook.with_suffix('.html'); html.write_text(body,encoding='utf-8')
    result={'passed':True,'finished_at':now(),'wall_seconds':time.perf_counter()-start,
            'code_cells_executed':len(code_cells),'error_outputs':0,'notebook':str(notebook),'html':str(html),
            'notebook_sha256':file_hash(notebook),'html_sha256':file_hash(html),'paid_api_calls':0,
            'notebook_performs_model_inference':False}
    write_json(FINAL/'validation/notebook_execution.json',result)
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__': finalize()
