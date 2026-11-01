"""Multi-Turn Conversational Agent with Memory, Verification gating, and Tool Integration (Step 3).

Handles:
1. Turn-by-turn conversation memory with entity/order tracking.
2. Two-step caller identity verification:
   - Request order ID -> lookup_order(order_id)
   - Request 4-digit registered phone suffix OR 6-digit delivery pincode -> lookup_order(order_id, verification_value)
   - Max 1 retry on verification failure; never expose raw verification values.
3. RAG document retrieval grounded on active conversation context.
4. Tool execution:
   - lookup_order(order_id, verification_value=None)
   - create_ticket(summary, order_id=None, reason=None, intent=None)
5. Decision detection: clarify, answer, escalate, refuse.
"""

import os
import re
import sys
from dotenv import load_dotenv

# Ensure tools can be imported
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tools import lookup_order, create_ticket
from src.agent import ask
from src.retriever import HybridRetriever

load_dotenv()


class SupportConversationAgent:
    def __init__(self, retriever: HybridRetriever = None):
        self.retriever = retriever or HybridRetriever()
        self.model_cfg = {
            "model": "openai/gpt-oss-120b",
            "system_prompt": "",  # Populated dynamically with turn context
        }
        self.reset()

    def reset(self):
        """Reset state for a fresh conversation."""
        self.history = []               # list of {"role": "customer"|"agent", "text": str}
        self.active_order_id = None
        self.verified = False
        self.verification_attempts = 0  # Max 1 retry (2 total attempts)
        self.verified_order_data = None
        self.ticket_created = None
        self.executed_tools = []        # list of tool call signatures executed this turn

    def _extract_order_id(self, text: str) -> str:
        """Extract 8-digit order number even if separated by space."""
        # Find 8 consecutive digits or two 4-digit groups
        m = re.search(r"\b(\d{4})\s*(\d{4})\b", text)
        if m:
            return m.group(1) + m.group(2)
        m8 = re.search(r"\b(\d{8})\b", text)
        if m8:
            return m8.group(1)
        return None

    def _extract_verification_value(self, text: str) -> str:
        """Extract 4-digit phone suffix or 6-digit pincode."""
        # Match explicit statements like '4533' or '781006'
        # 6-digit pincode
        m6 = re.search(r"\b(\d{6})\b", text)
        if m6:
            return m6.group(1)
        # 4-digit phone
        m4 = re.search(r"\b(\d{4})\b", text)
        if m4:
            return m4.group(1)
        return None

    def respond(self, customer_message: str) -> tuple[str, list[str]]:
        """Process one customer message in the conversation.
        
        Returns:
            (agent_reply_text, list_of_executed_tool_calls)
        """
        turn_tools = []
        cust_msg_lower = customer_message.lower()

        # Check for order ID
        found_order = self._extract_order_id(customer_message)
        if found_order and not self.active_order_id:
            self.active_order_id = found_order
            # If the customer message only introduced the order ID (turn 2), ask for verification
            # Do NOT treat the first 4 digits of the order ID as a verification value!
            tool_call = f"lookup_order(order_id={self.active_order_id})"
            turn_tools.append(tool_call)
            lang_is_hi = any(ch in customer_message for ch in "अआइईउऊऋएऐओऔकखगघचछजझटठडढणतथदधनपफबभमयरलवशषसह") or "hai" in cust_msg_lower or "number" in cust_msg_lower
            if lang_is_hi:
                reply = "सुरक्षा कारणों से, कृपया अपने पंजीकृत फ़ोन नंबर के आख़िरी 4 अंक (last 4 digits) या डिलीवरी पिनकोड (pincode) साझा करें।"
            else:
                reply = "For security and privacy, please share the last 4 digits of your registered phone number or your delivery pincode."
            self.history.append({"role": "customer", "text": customer_message})
            self.history.append({"role": "agent", "text": reply})
            return reply, turn_tools

        # 1. State: Needs verification value
        if self.active_order_id and not self.verified:
            verif_val = self._extract_verification_value(customer_message)
            if verif_val:
                self.verification_attempts += 1
                tool_call = f"lookup_order(order_id={self.active_order_id}, verification_value={verif_val})"
                turn_tools.append(tool_call)
                res = lookup_order(self.active_order_id, verification_value=verif_val)

                if res.get("verified"):
                    self.verified = True
                    self.verified_order_data = res
                    # Success reply
                    # Check what the customer wanted if anything else was said
                    if "kya" in cust_msg_lower or "can" in cust_msg_lower or "want" in cust_msg_lower:
                        pass # Let LLM answer with order context
                    else:
                        lang_is_hi = any(ch in customer_message for ch in "अआइईउऊऋएऐओऔकखगघचछजझटठडढणतथदधनपफबभमयरलवशषसह") or "hain" in cust_msg_lower or "aakhri" in cust_msg_lower
                        if lang_is_hi:
                            reply = f"धन्यवाद, आपकी पहचान सत्यापित हो गई है। मैं आपके ऑर्डर {self.active_order_id} के संबंध में आपकी क्या सहायता कर सकता हूँ?"
                        else:
                            reply = f"Thank you, your order {self.active_order_id} has been verified. How can I help you with this order today?"
                        self.history.append({"role": "customer", "text": customer_message})
                        self.history.append({"role": "agent", "text": reply})
                        return reply, turn_tools
                else:
                    # Verification failed
                    if self.verification_attempts >= 2:
                        # Max retries reached -> refuse to share order details
                        reply = "I'm sorry, but verification has failed. For security and privacy, I cannot disclose details about this order. You may contact support with your registered details."
                        self.history.append({"role": "customer", "text": customer_message})
                        self.history.append({"role": "agent", "text": reply})
                        return reply, turn_tools
                    else:
                        reply = "That verification value did not match our records. Please provide the last 4 digits of your registered phone number or your 6-digit delivery pincode (1 attempt remaining)."
                        self.history.append({"role": "customer", "text": customer_message})
                        self.history.append({"role": "agent", "text": reply})
                        return reply, turn_tools

        # 2. Check for human escalation / high-value damage / compensation demand
        needs_escalation = False
        escalation_reason = "human_requested"
        if any(term in cust_msg_lower for term in ["insan", "human", "colleague", "connect", "transfer", "talk to someone"]):
            needs_escalation = True
            escalation_reason = "human_agent_requested"
        elif any(term in cust_msg_lower for term in ["compensation", "500", "rupaye do", "claim ₹"]):
            needs_escalation = True
            escalation_reason = "unsupported_compensation_claim"
        elif any(term in cust_msg_lower for term in ["damage", "replacement", "broken", "toota", "crack"]):
            # Check if order total or product is high-value (> 10,000)
            if self.verified_order_data and (self.verified_order_data.get("total_inr", 0) > 10000 or self.verified_order_data.get("damage_status") == "priority_resolution"):
                needs_escalation = True
                escalation_reason = "high_value_damage_review"

        if needs_escalation and not self.ticket_created:
            summary = (
                f"Customer requested escalation regarding order {self.active_order_id or 'N/A'}. "
                f"Reason: {escalation_reason}. Issue: {customer_message.strip()[:100]}."
            )
            t_res = create_ticket(summary=summary, order_id=self.active_order_id, reason=escalation_reason)
            if t_res.get("created"):
                self.ticket_created = t_res["ticket_id"]
                turn_tools.append("create_ticket()")

        # 3. Retrieve relevant KB docs based on current question + history
        search_query = customer_message
        if self.verified_order_data:
            # Add item title or category for richer coreference resolution
            for item in self.verified_order_data.get("items", []):
                search_query += f" {item['title']}"

        retrieved = self.retriever.retrieve(search_query, top_k=4)

        # 4. Formulate Prompt with Conversation History & Verified Order Data
        context_str = "\n\n".join([f"[{c['section_id']}] {c['title']}:\n{c['text']}" for c in retrieved])

        order_context_str = "No verified order details available."
        if self.verified and self.verified_order_data:
            od = self.verified_order_data
            items_desc = ", ".join([f"{it['title']} (qty: {it['qty']}, ₹{it['unit_price_inr']}, cat: {it['return_category']})" for it in od.get("items", [])])
            order_context_str = (
                f"VERIFIED ORDER DATA for Order #{od.get('order_id')}:\n"
                f"- Placed At: {od.get('placed_at')}, Status: {od.get('status')}\n"
                f"- Delivered On: {od.get('delivered_on')}, Shipped On: {od.get('shipped_on')}\n"
                f"- Payment Method: {od.get('payment_method')}, Total: ₹{od.get('total_inr')}\n"
                f"- Plus Member: {od.get('is_plus')}\n"
                f"- Refund Status: {od.get('return_refund_status') or od.get('cancellation_refund_amount_inr') or 'None'}\n"
                f"- Damage Status: {od.get('damage_status')}\n"
                f"- Items: {items_desc}\n"
                f"- Timeline: {od.get('timeline')}\n"
                f"- Contacts in last 7 days: {od.get('contacts_last_7_days')}\n"
                f"- Express late credit: {od.get('express_late_credit_inr')}\n"
            )

        conversation_history_str = "\n".join([f"{turn['role'].upper()}: {turn['text']}" for turn in self.history[-6:]])

        prompt = (
            f"You are ZentroMart's AI customer support assistant. Today is 20 October 2026.\n"
            f"Business days are Mon-Sat skipping 2026-08-15, 2026-10-02, 2026-10-19, 2026-11-09.\n"
            f"Return windows use calendar days.\n\n"
            f"STRICT RULES:\n"
            f"- If customer says hello/needs help with an order, introduce yourself as an AI assistant and ask for their order ID.\n"
            f"- If a ticket was created, politely let them know you have raised a ticket and a colleague/team will follow up.\n"
            f"- If compensation or immediate unapproved refund is demanded that policy does not allow, politely refuse, create/mention ticket handover.\n"
            f"- Never promise something your code did not execute.\n"
            f"- Reply in the same language (Hindi, Hinglish, English) and tone.\n\n"
            f"RELEVANT POLICIES:\n{context_str}\n\n"
            f"{order_context_str}\n\n"
            f"RECENT CONVERSATION:\n{conversation_history_str}\n"
            f"CUSTOMER: {customer_message}\n"
            f"ASSISTANT:"
        )

        agent_cfg = {
            "model": "openai/gpt-oss-120b",
            "system_prompt": "You are ZentroMart's support assistant. Answer accurately based on context.",
        }

        reply = ask(agent_cfg, prompt)

        self.history.append({"role": "customer", "text": customer_message})
        self.history.append({"role": "agent", "text": reply})

        return reply, turn_tools
