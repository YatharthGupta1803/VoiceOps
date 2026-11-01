import unittest
from unittest.mock import patch
from src.conversation_agent import SupportConversationAgent
from src.voice_agent import normalize_text_for_speech

class TestVoiceOpsFixes(unittest.TestCase):
    def test_normalizer_currency_and_dates(self):
        text = "Your refund of ₹1,200 will be credited on 20 Oct. for order 89782837."
        normalized = normalize_text_for_speech(text)
        self.assertIn("one thousand two hundred rupees", normalized)
        self.assertIn("20th of October 2026", normalized)
        self.assertIn("8978 2837", normalized)
        self.assertNotIn("₹", normalized)

    def test_normalizer_large_number(self):
        text = "Order 12345678 total is ₹15,450."
        normalized = normalize_text_for_speech(text)
        self.assertIn("fifteen thousand four hundred fifty rupees", normalized)

    def test_retry_lockout_prevents_brute_force(self):
        agent = SupportConversationAgent()
        agent.active_order_id = "89782837"
        
        # Turn 1: Wrong guess
        reply, tools = agent.respond("4500")
        self.assertEqual(agent.verification_attempts, 1)
        self.assertFalse(agent.verified)
        self.assertTrue(len(tools) > 0)

        # Turn 2: Second wrong guess (hits cap)
        reply, tools = agent.respond("4501")
        self.assertEqual(agent.verification_attempts, 2)
        self.assertFalse(agent.verified)
        self.assertIn("locked", reply.lower())

        # Turn 3: Correct value on 3rd try (MUST BE LOCKED OUT - NO TOOL CALL)
        with patch("src.conversation_agent.lookup_order") as mock_lookup:
            reply, tools = agent.respond("4533")
            self.assertEqual(tools, [])
            self.assertFalse(agent.verified)
            mock_lookup.assert_not_called()
            self.assertIn("exceeded", reply.lower())

    @patch("src.conversation_agent.ask", return_value="The warranty lasts 12 months.")
    def test_no_false_positive_on_1500_for_compensation(self, mock_ask):
        agent = SupportConversationAgent()
        agent.active_order_id = "89782837"
        agent.verified = True
        # "1500" should not trigger compensation escalation (which previously looked for "500")
        reply, tools = agent.respond("Is the warranty valid for 1500 hours?")
        self.assertFalse(agent.ticket_created)

if __name__ == "__main__":
    unittest.main()
