import unittest
import requests

BASE_URL = "http://localhost:8000/api"

class TestRobustness(unittest.TestCase):
    def test_invalid_date_format(self):
        """Test API behavior with invalid date format"""
        response = requests.get(f"{BASE_URL}/tweets?startDate=invalid-date")
        # Should either ignore invalid date or return 422/400
        # FastAPI default for date type is 422 Unprocessable Entity
        self.assertIn(response.status_code, [400, 422, 200]) 

    def test_sql_injection_attempt(self):
        """Test API resilience against SQL injection-like strings"""
        injection_str = "' OR '1'='1"
        response = requests.get(f"{BASE_URL}/tweets?motif={injection_str}")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        # Should return empty list or filtered list, not crash
        self.assertIsInstance(data.get("data"), list)

    def test_large_pagination(self):
        """Test API with large limit"""
        response = requests.get(f"{BASE_URL}/tweets?limit=10000")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertLessEqual(len(data.get("data")), 10000)

    def test_negative_page(self):
        """Test API with negative page number"""
        response = requests.get(f"{BASE_URL}/tweets?page=-1")
        # Should handle gracefully (e.g. treat as page 1 or return error)
        self.assertIn(response.status_code, [200, 422])

if __name__ == "__main__":
    unittest.main()
