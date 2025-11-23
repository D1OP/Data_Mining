import unittest
import requests
import time
import concurrent.futures

BASE_URL = "http://localhost:8000/api"

class TestStability(unittest.TestCase):
    def make_request(self, url):
        start = time.time()
        try:
            response = requests.get(url)
            duration = time.time() - start
            return response.status_code, duration
        except Exception as e:
            return 0, 0

    def test_sequential_load(self):
        """Test 50 sequential requests"""
        n_requests = 50
        errors = 0
        total_time = 0
        
        print(f"\nRunning {n_requests} sequential requests...")
        for _ in range(n_requests):
            status, duration = self.make_request(f"{BASE_URL}/kpis")
            total_time += duration
            if status != 200:
                errors += 1
        
        avg_time = total_time / n_requests
        print(f"Average response time: {avg_time:.3f}s")
        self.assertEqual(errors, 0, f"Encountered {errors} errors during load test")
        self.assertLess(avg_time, 0.5, "Average response time should be under 500ms")

    def test_concurrent_load(self):
        """Test 20 concurrent requests"""
        n_requests = 20
        url = f"{BASE_URL}/tweets?limit=50"
        
        print(f"\nRunning {n_requests} concurrent requests...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(self.make_request, url) for _ in range(n_requests)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]
            
        errors = sum(1 for status, _ in results if status != 200)
        self.assertEqual(errors, 0, f"Encountered {errors} errors during concurrent load test")

if __name__ == "__main__":
    unittest.main()
