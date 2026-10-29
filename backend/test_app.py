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

if __name__=='__main__': unittest.main()
