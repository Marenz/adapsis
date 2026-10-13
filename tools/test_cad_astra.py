import io,json,unittest
from unittest.mock import patch
from urllib.error import HTTPError, URLError
import cad_astra

class AstraRoutingTests(unittest.TestCase):
    def test_revision_keeps_brief_source_and_actual_failure(self):
        response={'model':cad_astra.CAD_MODEL,'choices':[{'finish_reason':'stop','message':{'content':'x=2'}}]}
        with patch('cad_astra.urlopen',return_value=io.BytesIO(json.dumps(response).encode())) as call:
            cad_astra.generate('rack','20 ring pegs','PLA','x=1','rack-left: 11 components')
        payload=json.loads(call.call_args.args[0].data)
        self.assertEqual(payload['model'],cad_astra.CAD_MODEL)
        self.assertIn('20 ring pegs',payload['messages'][1]['content'])
        self.assertEqual(payload['messages'][2],{'role':'assistant','content':'x=1'})
        self.assertIn('11 components',payload['messages'][3]['content'])

    def test_pins_astra_and_high_effort(self):
        response={'model':'chatgpt/gpt-6-astra','id':'test','choices':[{'finish_reason':'stop','message':{'content':'x = 1'}}]}
        with patch('cad_astra.urlopen',return_value=io.BytesIO(json.dumps(response).encode())) as call:
            source,author=cad_astra.generate('part','12 mm cube','PLA')
            payload=json.loads(call.call_args.args[0].data)
        self.assertEqual(payload['model'],'chatgpt/gpt-6-astra')
        self.assertEqual(payload['reasoning_effort'],'high')
        self.assertEqual(source,'x = 1')
        self.assertEqual(author['requested_model'],cad_astra.CAD_MODEL)

    def test_other_model_is_rejected_without_fallback(self):
        response={'model':'some-other-model','choices':[]}
        with patch('cad_astra.urlopen',return_value=io.BytesIO(json.dumps(response).encode())) as call:
            with self.assertRaisesRegex(RuntimeError,'must be Astra'):
                cad_astra.generate('part','cube','PLA')
        self.assertEqual(call.call_count,1)

    def test_truncation_rejected(self):
        response={'model':cad_astra.CAD_MODEL,'choices':[{'finish_reason':'length','message':{'content':'x=1'}}]}
        with patch('cad_astra.urlopen',return_value=io.BytesIO(json.dumps(response).encode())) as call:
            with self.assertRaisesRegex(RuntimeError,'truncated'):cad_astra.generate('part','cube','PLA')
        self.assertEqual(call.call_count,1)

    def test_transport_failure_retries_then_reports_plain_error(self):
        with patch('cad_astra.urlopen',side_effect=TimeoutError('timed out')) as call:
            with self.assertRaisesRegex(RuntimeError,
                    r'CAD author \(Astra\) did not respond within \d+s after 2 attempts') as caught:
                cad_astra.generate('part','cube','PLA')
        self.assertEqual(call.call_count,cad_astra.ATTEMPTS)
        self.assertNotIn('timed out',str(caught.exception))

    def test_connection_error_retries(self):
        with patch('cad_astra.urlopen',side_effect=ConnectionError('reset by peer')) as call:
            with self.assertRaisesRegex(RuntimeError,'after 2 attempts'):cad_astra.generate('part','cube','PLA')
        self.assertEqual(call.call_count,cad_astra.ATTEMPTS)

    def test_url_error_retries(self):
        with patch('cad_astra.urlopen',side_effect=URLError('timed out')) as call:
            with self.assertRaisesRegex(RuntimeError,'after 2 attempts'):cad_astra.generate('part','cube','PLA')
        self.assertEqual(call.call_count,cad_astra.ATTEMPTS)

    def test_transport_failure_then_success_recovers(self):
        response={'model':cad_astra.CAD_MODEL,'id':'x','choices':[{'finish_reason':'stop','message':{'content':'x = 1'}}]}
        with patch('cad_astra.urlopen',side_effect=[TimeoutError('timed out'),io.BytesIO(json.dumps(response).encode())]) as call:
            source,author=cad_astra.generate('part','cube','PLA')
        self.assertEqual(call.call_count,2)
        self.assertEqual(source,'x = 1')
        self.assertEqual(author['response_model'],cad_astra.CAD_MODEL)

    def test_http_error_reports_status_without_timeout_claim(self):
        with patch('cad_astra.urlopen',side_effect=HTTPError(cad_astra.GATEWAY,402,'Payment Required',{},None)) as call:
            with self.assertRaisesRegex(RuntimeError,'HTTP 402') as caught:
                cad_astra.generate('part','cube','PLA')
        self.assertEqual(call.call_count,1)
        self.assertNotIn('did not respond within',str(caught.exception))
        self.assertNotIn('timed out',str(caught.exception))

    def _assert_malformed(self,body):
        with patch('cad_astra.urlopen',return_value=io.BytesIO(body)) as call:
            with self.assertRaisesRegex(RuntimeError,'malformed response') as caught:
                cad_astra.generate('part','cube','PLA')
        self.assertEqual(call.call_count,1)
        self.assertIn('Astra',str(caught.exception))

    def test_malformed_response_is_wrapped_not_retried(self):
        cases={
            'JSONDecodeError':b'not json at all',
            'IndexError':json.dumps({'model':cad_astra.CAD_MODEL,'choices':[]}).encode(),
            'KeyError':json.dumps({'model':cad_astra.CAD_MODEL}).encode(),
            'AttributeError':json.dumps({'model':cad_astra.CAD_MODEL,'choices':[{'finish_reason':'stop','message':{'content':123}}]}).encode(),
            'SyntaxError':json.dumps({'model':cad_astra.CAD_MODEL,'choices':[{'finish_reason':'stop','message':{'content':'def (:'}}]}).encode(),
        }
        for name,body in cases.items():
            with self.subTest(shape=name):self._assert_malformed(body)

if __name__=='__main__':unittest.main()
