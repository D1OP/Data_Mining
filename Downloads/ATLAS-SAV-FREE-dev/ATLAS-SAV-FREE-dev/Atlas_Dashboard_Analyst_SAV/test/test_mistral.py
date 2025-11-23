import unittest
from unittest.mock import MagicMock, patch
import json
import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.llm_classification import classify_single_tweet, classify_batch, LLMClassificationError

class TestMistralAnalysis(unittest.TestCase):
    def setUp(self):
        # Mock Mistral client
        self.mock_client = MagicMock()
        self.mock_chat = MagicMock()
        self.mock_client.chat = self.mock_chat
        
        # Sample valid JSON response from LLM
        self.sample_response = {
            "motif": "Problème technique",
            "sentiment": "Négatif",
            "urgence": "Moyenne",
            "intention": "Plainte",
            "risque_churn": "Faible",
            "reponse_suggeree": "Bonjour, désolé pour ce problème."
        }
        
        # Mock response object structure
        self.mock_message = MagicMock()
        self.mock_message.content = json.dumps(self.sample_response)
        self.mock_choice = MagicMock()
        self.mock_choice.message = self.mock_message
        self.mock_api_response = MagicMock()
        self.mock_api_response.choices = [self.mock_choice]

    def test_classify_single_tweet_success(self):
        """Test successful classification of a single tweet"""
        # Setup mock return value
        self.mock_chat.complete.return_value = self.mock_api_response
        
        tweet = "Ma box ne marche plus depuis ce matin, c'est pénible !"
        result = classify_single_tweet(self.mock_client, tweet)
        
        # Verify result structure
        self.assertIsInstance(result, str)
        parsed_result = json.loads(result)
        self.assertEqual(parsed_result["motif"], "Problème technique")
        self.assertEqual(parsed_result["sentiment"], "Négatif")
        
        # Verify API was called
        self.mock_chat.complete.assert_called_once()

    def test_classify_single_tweet_api_error(self):
        """Test handling of API errors"""
        # Setup mock to raise exception
        self.mock_chat.complete.side_effect = Exception("API Error")
        
        tweet = "Test tweet"
        # Should raise LLMClassificationError
        with self.assertRaises(LLMClassificationError):
            classify_single_tweet(self.mock_client, tweet)

    def test_classify_batch_success(self):
        """Test batch classification"""
        # Setup mock
        self.mock_chat.complete.return_value = self.mock_api_response
        
        tweets = ["Tweet 1", "Tweet 2"]
        results = classify_batch(self.mock_client, tweets)
        
        self.assertEqual(len(results), 2)
        
        # classify_batch returns dict with 'raw_response'
        first_result_json = json.loads(results[0]["raw_response"])
        self.assertEqual(first_result_json["motif"], "Problème technique")
        
        # Verify API called twice
        self.assertEqual(self.mock_chat.complete.call_count, 2)

if __name__ == "__main__":
    unittest.main()
