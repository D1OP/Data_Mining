import unittest
import requests

BASE_URL = "http://localhost:8000/api"

class TestQuality(unittest.TestCase):
    def test_kpi_consistency(self):
        """Verify that total tweets equals sum of sentiments"""
        response = requests.get(f"{BASE_URL}/kpis")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        total = data["total_tweets"]
        pos = data["positifs"]
        neg = data["negatifs"]
        neu = data["neutres"]
        
        # Allow for small discrepancies if there are other sentiments, but usually should match
        self.assertEqual(total, pos + neg + neu, "Total tweets should equal sum of sentiments")

    def test_churn_rate_validity(self):
        """Verify churn rate is a valid percentage"""
        response = requests.get(f"{BASE_URL}/kpis")
        data = response.json()
        churn_pct = data["churn_pct"]
        self.assertTrue(0 <= churn_pct <= 100, f"Churn pct {churn_pct} should be between 0 and 100")

    def test_data_fields_completeness(self):
        """Verify critical fields are present in tweets"""
        response = requests.get(f"{BASE_URL}/tweets?limit=10")
        data = response.json()
        tweets = data.get("data", [])
        
        if not tweets:
            self.skipTest("No tweets found to test")
            
        for tweet in tweets:
            self.assertIn("motif", tweet)
            self.assertIn("sentiment_norm", tweet)
            self.assertIn("date", tweet)
            self.assertIsNotNone(tweet["motif"])

if __name__ == "__main__":
    unittest.main()
