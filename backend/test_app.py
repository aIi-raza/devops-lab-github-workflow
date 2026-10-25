import unittest

from app import app


class CalculatorTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def post(self, a, b, operation):
        return self.client.post(
            "/calculate", json={"a": a, "b": b, "operation": operation}
        )

    def test_add(self):
        self.assertEqual(self.post(2, 3, "add").get_json()["result"], 5)

    def test_subtract(self):
        self.assertEqual(self.post(5, 3, "subtract").get_json()["result"], 2)

    def test_multiply(self):
        self.assertEqual(self.post(4, 3, "multiply").get_json()["result"], 12)

    def test_divide(self):
        self.assertEqual(self.post(10, 4, "divide").get_json()["result"], 2.5)

    def test_divide_by_zero(self):
        self.assertEqual(self.post(1, 0, "divide").status_code, 400)

    def test_bad_input(self):
        self.assertEqual(self.post("x", 1, "add").status_code, 400)

    def test_unknown_operation(self):
        self.assertEqual(self.post(1, 1, "power").status_code, 400)


if __name__ == "__main__":
    unittest.main()
