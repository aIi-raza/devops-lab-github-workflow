import unittest

from app import app


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_home_reports_ok(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["status"], "ok")

    def test_results_returns_six_models(self):
        response = self.client.get("/api/results")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.get_json()["models"]), 6)

    def test_every_model_has_all_metrics(self):
        for model in self.client.get("/api/results").get_json()["models"]:
            for key in ("name", "short", "source", "accuracy", "precision", "recall", "f1"):
                self.assertIn(key, model)

    def test_scores_are_valid_percentages(self):
        for model in self.client.get("/api/results").get_json()["models"]:
            for key in ("accuracy", "precision", "recall", "f1"):
                self.assertGreaterEqual(model[key], 0)
                self.assertLessEqual(model[key], 100)

    def test_ours_and_paper_rows_both_present(self):
        sources = {m["source"] for m in self.client.get("/api/results").get_json()["models"]}
        self.assertEqual(sources, {"ours", "paper"})

    def test_cors_header_lets_the_pages_site_read_the_answer(self):
        response = self.client.get("/api/results")
        self.assertEqual(response.headers["Access-Control-Allow-Origin"], "*")

    def test_unknown_route_returns_404(self):
        self.assertEqual(self.client.get("/nope").status_code, 404)


if __name__ == "__main__":
    unittest.main()