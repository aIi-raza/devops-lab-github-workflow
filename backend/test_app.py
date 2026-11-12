import io
import unittest
from app import app, parse_ground_truth

class ApiTests(unittest.TestCase):
    def setUp(self): self.client = app.test_client()
    def test_home_reports_ok(self):
        r=self.client.get('/'); self.assertEqual(r.status_code,200); self.assertEqual(r.get_json()['status'],'ok')
    def test_results_returns_six_models(self): self.assertEqual(len(self.client.get('/api/results').get_json()['models']),6)
    def test_every_model_has_all_metrics(self):
        for model in self.client.get('/api/results').get_json()['models']:
            for key in ('name','short','source','accuracy','precision','recall','f1'): self.assertIn(key,model)
    def test_scores_are_valid_percentages(self):
        for model in self.client.get('/api/results').get_json()['models']:
            for key in ('accuracy','precision','recall','f1'): self.assertGreaterEqual(model[key],0); self.assertLessEqual(model[key],100)
    def test_ours_and_paper_rows_both_present(self): self.assertEqual({m['source'] for m in self.client.get('/api/results').get_json()['models']},{'ours','paper'})
    def test_cors_header(self): self.assertEqual(self.client.get('/api/results',headers={'Origin':'http://localhost:4200'}).headers['Access-Control-Allow-Origin'],'http://localhost:4200')
    def test_unknown_route_returns_404(self): self.assertEqual(self.client.get('/nope').status_code,404)
    def test_health_reports_missing_artifacts_without_crashing(self):
        r=self.client.get('/api/health'); self.assertEqual(r.status_code,200); self.assertIn('prediction_ready',r.get_json())
    def test_models_endpoint_lists_four(self): self.assertEqual(len(self.client.get('/api/models').get_json()['models']),4)
    def test_missing_features_rejected(self):
        r=self.client.post('/api/predict',json={'model':'random_forest','features':{'duration':0}}); self.assertEqual(r.status_code,400)
    def test_unknown_model_rejected(self):
        r=self.client.post('/api/predict',json={'model':'made_up','features':{}}); self.assertEqual(r.status_code,400)
    def test_csv_empty_rejected(self):
        r=self.client.post('/api/predict/csv',data={'file':(io.BytesIO(b''),'empty.csv')},content_type='multipart/form-data'); self.assertEqual(r.status_code,400)
    def test_csv_extension_rejected(self):
        r=self.client.post('/api/predict/csv',data={'file':(io.BytesIO(b'a,b\n1,2'),'data.txt')},content_type='multipart/form-data'); self.assertEqual(r.status_code,400)
    def test_ground_truth_parser_accepts_binary_labels(self):
        self.assertEqual(parse_ground_truth('Normal', 2), 0)
        self.assertEqual(parse_ground_truth('Attack', 3), 1)
        self.assertEqual(parse_ground_truth('0', 4), 0)
        self.assertEqual(parse_ground_truth('1', 5), 1)
    def test_ground_truth_parser_rejects_unknown_label(self):
        with self.assertRaises(ValueError): parse_ground_truth('maybe', 2)
    def test_triage_missing_features_rejected(self):
        r=self.client.post('/api/triage',json={'features':{'duration':0}})
        self.assertEqual(r.status_code,400)
    def test_triage_empty_body_rejected(self):
        r=self.client.post('/api/triage',json={})
        self.assertEqual(r.status_code,400)
    def test_triage_csv_empty_rejected(self):
        r=self.client.post('/api/triage/csv',data={'file':(io.BytesIO(b''),'empty.csv')},content_type='multipart/form-data')
        self.assertEqual(r.status_code,400)

import json, pathlib
from app import _bundles
READY = 'random_forest' in _bundles and 'xgboost' in _bundles
BORDERLINE = {"duration":0,"protocol_type":"udp","service":"private","flag":"SF","src_bytes":0,"dst_bytes":0,"wrong_fragment":0,"hot":0,"num_failed_logins":0,"logged_in":0,"count":1,"srv_count":1,"serror_rate":0.0,"same_srv_rate":1.0,"diff_srv_rate":0.0,"dst_host_count":4,"dst_host_srv_count":1,"dst_host_same_srv_rate":0.25,"dst_host_diff_srv_rate":0.5,"dst_host_same_src_port_rate":0.25,"dst_host_srv_diff_host_rate":0.0,"dst_host_serror_rate":0.0,"dst_host_srv_serror_rate":0.0,"dst_host_rerror_rate":0.0,"dst_host_srv_rerror_rate":0.0}
DEMO = pathlib.Path(__file__).resolve().parents[1] / 'frontend' / 'public' / 'samples' / 'sample_triage_demo.csv'

@unittest.skipUnless(READY, 'model artifacts not available')
class TriageBehaviourTests(unittest.TestCase):
    def setUp(self): self.client = app.test_client()
    def test_models_disagree_goes_to_triage(self):
        r = self.client.post('/api/triage', json={'features': BORDERLINE, 'margin': 0.10}).get_json()
        self.assertFalse(r['consensus']); self.assertEqual(r['status'], 'TRIAGE'); self.assertEqual(r['decision'], 'Triage')
    def test_large_margin_forces_triage_on_agreeing_models(self):
        feats = dict(BORDERLINE, protocol_type='tcp', service='http', flag='SF', src_bytes=232, dst_bytes=8153, logged_in=1)
        low = self.client.post('/api/triage', json={'features': feats, 'margin': 0.02}).get_json()
        high = self.client.post('/api/triage', json={'features': feats, 'margin': 0.49}).get_json()
        self.assertEqual(high['status'], 'TRIAGE')
        if low['consensus']: self.assertEqual(low['status'], 'AUTOMATED')
    def test_demo_csv_has_triage_rows_and_margin_matters(self):
        def run(m):
            return self.client.post('/api/triage/csv', data={'file': (io.BytesIO(DEMO.read_bytes()), 'demo.csv'), 'margin': str(m)}, content_type='multipart/form-data').get_json()
        a, b = run(0.10), run(0.40)
        self.assertGreater(a['triage_count'], 0); self.assertGreaterEqual(b['triage_count'], a['triage_count'])
        self.assertIn('forced_accuracy', a); self.assertLess(a['triage_forced_accuracy'], a['forced_accuracy'])

if __name__=='__main__': unittest.main()
