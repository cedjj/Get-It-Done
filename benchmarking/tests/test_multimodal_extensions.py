import asyncio
import base64
import csv
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import project_paths
from prompt_benchmark.core import make_plan, request_for, experiments, write_json, connect
from prompt_benchmark.cloud import RateLimitedHTTP, build_request, parse_response
from prompt_benchmark.photos import verified_links, download_image, attachment_id
from prompt_benchmark.runner import run, OllamaAdapter
from prompt_benchmark.evaluate import evaluate


class MultimodalTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.out = self.root / 'run'
        self.url = 'https://getitdone.sandiego.gov/servlet/servlet.FileDownload?file=00P8z000000KZsJEAW'
        buffer = io.BytesIO()
        Image.new('RGB', (20, 10), 'blue').save(buffer, format='JPEG')
        self.blob = buffer.getvalue()
        (self.root / 'photo.jpg').write_bytes(self.blob)
        write_json(self.root / 'photos.json', {self.url:'photo.jpg'})
        with (self.root / 'inputs.csv').open('w', newline='', encoding='utf-8') as f:
            w=csv.writer(f)
            w.writerow(['ID','DESCRIPTION','ORIGINAL_TYPE','PUBLIC_PHOTO_URL_1'])
            w.writerow(['one','Blue graffiti on wall','Graffiti',self.url])
            w.writerow(['two','Please help','Curb',''])
        self.model={'id':'fixture','adapter':'mock','supports_photos':True}

    def make(self, model=None, **kwargs):
        config={'models':[model or self.model],'prompts':{'detailed':{
            'template':str(project_paths.ROOT/'scripts/prompt_benchmark/templates/detailed_blinded_v2.txt'),
            'version':'detailed-test-v2'}}}
        write_json(self.root/'config.json',config)
        return make_plan(self.root/'inputs.csv',self.root/'config.json',self.out,
                         blinded=True,photo_manifest=self.root/'photos.json',**kwargs)

    def request(self):
        plan=self.make()
        exp=next(e for e in experiments(plan) if e['modality']=='text_photos')
        case=next(c for c in plan['cases'] if c['id']=='one')
        return request_for(plan,exp,case)

    def test_each_paid_provider_is_blocked_before_http(self):
        for provider in ('openai','gemini','anthropic'):
            with self.subTest(provider=provider), patch('requests.post',side_effect=AssertionError('Paid HTTP invoked')):
                with self.assertRaises(PermissionError):
                    asyncio.run(RateLimitedHTTP().generate({'adapter':provider},{}))

    def test_runner_blocks_paid_models_even_if_key_exists(self):
        model={'id':'api','adapter':'openai','model':'test-vision','supports_photos':True}
        plan=self.make(model)
        with patch.dict(os.environ,{'OPENAI_API_KEY':'secret-test-key'}), patch('requests.post',side_effect=AssertionError('HTTP invoked')):
            with self.assertRaises(PermissionError):
                asyncio.run(run(plan,self.out))

    def test_all_provider_payloads_use_real_image_bytes_and_same_schema(self):
        request=self.request()
        for provider in ('openai','gemini','anthropic'):
            url,body=build_request(provider,{'model':'authorized-model'},request)
            self.assertTrue(url.startswith('https://'))
            self.assertIn(request['images'][0],json.dumps(body))
            self.assertNotIn('CURRENT_TYPE',json.dumps(body))
            if provider=='openai':
                self.assertEqual(body['response_format']['json_schema']['schema'],request['schema'])
            elif provider=='gemini':
                self.assertEqual(body['generationConfig']['responseJsonSchema'],request['schema'])
            else:
                self.assertEqual(body['output_config']['format']['schema'],request['schema'])

    def test_real_adapter_rejects_current_type_and_labels(self):
        request=self.request()
        for field in ('CURRENT_TYPE','label','STATUS','ID'):
            bad=dict(request,user=json.dumps({'DESCRIPTION':'test','ORIGINAL_TYPE':'Graffiti',field:'forbidden'}))
            with patch('prompt_benchmark.local_transport.api',side_effect=AssertionError('HTTP invoked')), self.assertRaises(ValueError):
                asyncio.run(OllamaAdapter().generate({'model':'installed'},bad))
        for field in ('LOCATION','PHOTOS'):
            bad=dict(request,user=json.dumps({'DESCRIPTION':'test','ORIGINAL_TYPE':'Graffiti',field:{'CURRENT_TYPE':'forbidden'}}))
            with patch('prompt_benchmark.local_transport.api',side_effect=AssertionError('HTTP invoked')),self.assertRaises(ValueError):
                asyncio.run(OllamaAdapter().generate({'model':'installed'},bad))

    def test_provider_usage_counts_include_thinking_and_cached_tokens(self):
        google=parse_response('gemini',{'candidates':[{'finishReason':'STOP','content':{'parts':[{'text':'{}'}]}}],
                               'usageMetadata':{'promptTokenCount':10,'candidatesTokenCount':4,'thoughtsTokenCount':3}})
        self.assertEqual(google['output_tokens'],7)
        claude=parse_response('anthropic',{'stop_reason':'end_turn','content':[{'type':'text','text':'{}'}],
                              'usage':{'input_tokens':10,'cache_read_input_tokens':20,'output_tokens':4}})
        self.assertEqual(claude['input_tokens'],30)
        openai=parse_response('openai',{'choices':[{'finish_reason':'length','message':{'content':'{'}}],
                                     'usage':{'prompt_tokens':10,'completion_tokens':3}})
        self.assertIsNotNone(openai['transport_error'])

    def test_429_is_audited_with_retry_backoff(self):
        request=self.request()
        response=MagicMock(status_code=429,ok=False,text='rate-limited',headers={'Retry-After':'5'})
        adapter=RateLimitedHTTP(authorized=True)
        model={'adapter':'openai','model':'test','requests_per_minute':60}
        with patch.dict(os.environ,{'OPENAI_API_KEY':'secret'}),patch('requests.post',return_value=response):
            result=asyncio.run(adapter.generate(model,request))
        self.assertEqual(result['raw'],'rate-limited')
        self.assertIn('429',result['transport_error'])
        self.assertTrue(adapter.last)

    def test_error_audit_redacts_provider_echo_of_key(self):
        request=self.request()
        response=MagicMock(status_code=401,ok=False,text='Invalid key secret-key-value')
        with patch.dict(os.environ,{'OPENAI_API_KEY':'secret-key-value'}),patch('requests.post',return_value=response):
            result=asyncio.run(RateLimitedHTTP(authorized=True).generate({'adapter':'openai','model':'test'},request))
        self.assertNotIn('secret-key-value',result['raw'])
        self.assertIn('[REDACTED_API_KEY]',result['raw'])

    def test_photo_link_requires_parent_case_and_number(self):
        source=self.root/'source.csv'
        with source.open('w',newline='',encoding='utf-8') as f:
            w=csv.writer(f);w.writerow(['ID','CASENUMBER','PUBLIC_ATTACHMENT_URLS_JSON','PUBLIC_ATTACHMENT_IDS_JSON','PUBLIC_ATTACHMENT_FILENAMES_JSON'])
            w.writerow(['one','100',json.dumps([self.url]),json.dumps([attachment_id(self.url)]),json.dumps(['p.jpg'])])
        export=self.root/'attachments.csv'
        for parent,number,valid in [('one','100',True),('other','100',False),('one','200',False)]:
            with export.open('w',newline='',encoding='utf-8') as f:
                w=csv.writer(f);w.writerow(['Id','Name','ParentId','Parent.Name']);w.writerow([attachment_id(self.url),'p.jpg',parent,number])
            if valid:
                self.assertEqual(len(verified_links(source,export,{'one'})),1)
            else:
                with self.assertRaisesRegex(ValueError,'mismatch'):
                    verified_links(source,export,{'one'})

    def test_download_validates_image_and_reuses_verified_bytes(self):
        output=self.root/'cache'
        for name in ('originals','images','metadata'):
            (output/name).mkdir(parents=True)
        response=MagicMock(is_redirect=False,headers={'Content-Type':'image/jpeg'})
        response.iter_content.return_value=[self.blob]
        response.__enter__.return_value=response
        item={'source':self.url,'case_id':'one','case_number':'100','filename':'p.jpg'}
        with patch('requests.get',return_value=response) as get:
            first=download_image(item,output)
            second=download_image(item,output)
        self.assertEqual(first['sha256'],second['sha256'])
        self.assertEqual(get.call_count,1)
        self.assertTrue(first['case_ownership_verified'])
        changed=dict(item,case_id='other')
        with self.assertRaisesRegex(ValueError,'different case'):
            download_image(changed,output)

    def test_anchor_cannot_resample_or_truncate_cases(self):
        anchor=self.make()
        other=self.root/'other'
        with self.assertRaisesRegex(ValueError,'truncate or resample'):
            make_plan(self.root/'inputs.csv',self.root/'config.json',other,blinded=True,
                      anchor_path=self.out,limit=1)

    def test_pilot_report_scope_does_not_change_checkpoint(self):
        plan=self.make()
        asyncio.run(run(plan,self.out,retries=0))
        report=evaluate(plan,self.out,charts=False,bootstrap=0,case_ids={'one'},report_name='pilot')
        self.assertTrue(all(r['planned']==1 and r['completed']==1 for r in report['summary']))
        conn=connect(self.out)
        self.assertEqual(conn.execute('SELECT count(*) FROM predictions').fetchone()[0],16)
        conn.close()

    def test_independent_labels_attach_only_in_scoring(self):
        plan=self.make()
        asyncio.run(run(plan,self.out,retries=0))
        path=self.root/'verified.csv'
        path.write_text('ID,independently_verified,verification_provenance,verified_misroute,verified_category\none,true,Blinded human fixture review,false,Graffiti\n',encoding='utf-8')
        report=evaluate(plan,self.out,charts=False,bootstrap=10,verified_path=path)
        self.assertEqual(report['summary'][0]['verified_flag_n'],1)
        self.assertEqual(report['summary'][0]['verified_category_accuracy'],1)
        self.assertTrue(all(c['verified']=={} for c in plan['cases']))

    def test_local_inference_serializes_gpu_even_with_many_workers(self):
        from unittest.mock import AsyncMock
        plan=self.make({'id':'local','adapter':'ollama','model':'fixture','supports_photos':True})
        class LocalFixture:
            active=0
            peak=0
            async def generate(inner,model,request):
                inner.active+=1;inner.peak=max(inner.peak,inner.active)
                await asyncio.sleep(.005)
                inner.active-=1
                return {'content':json.dumps({'misroute':None,'recommended_category':None,'reason':'Fixture.'}),
                        'raw':'{}','input_tokens':10,'output_tokens':10}
        adapter=LocalFixture()
        with patch('prompt_benchmark.runner.preflight',new=AsyncMock()):
            asyncio.run(run(plan,self.out,workers=8,retries=0,adapters={'ollama':adapter}))
        self.assertEqual(adapter.peak,1)


if __name__=='__main__':
    unittest.main()
