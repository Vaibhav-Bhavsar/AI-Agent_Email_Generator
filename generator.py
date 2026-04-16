"""
Email Generation Assistant — Core Generator
Uses Groq API (free tier) with two models for comparison:
  - Model A: llama3-70b-8192  (with Few-Shot + Role-Play prompting)
  - Model B: llama3-8b-8192   (baseline, simple prompt)
"""

import os
import json
import requests
from typing import Optional

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

# ─────────────────────────────────────────────
# ADVANCED PROMPT TEMPLATE  (Role-Play + Few-Shot + Chain-of-Thought)
# ─────────────────────────────────────────────

SYSTEM_PROMPT_A = """You are Victoria Hargrove, a Senior Business Communications Specialist
with 15 years of experience crafting executive-level correspondence for Fortune 500 companies.
Your emails are known for three things: crystal-clear structure, seamless integration of facts,
and pitch-perfect tone calibration. You never write filler sentences.

Before writing, you silently follow this chain of thought:
1. Identify the single most important goal of this email.
2. Map every key fact to a sentence — no fact is left out, none feels forced.
3. Calibrate the opening and closing to match the requested tone exactly.
4. Review for conciseness — cut any sentence that doesn't earn its place.

OUTPUT FORMAT (strict):
Subject: <subject line>

<email body>

---
Few-shot examples follow. Study the pattern, then generate a new email.

--- EXAMPLE 1 ---
Intent: Follow up after a product demo
Key Facts: Demo held on June 5, product is InventoryPro v3, next step is a 30-day trial, contact is Sarah at sales@inventorypro.com
Tone: Professional, warm

Subject: Following Up on Your InventoryPro Demo — Next Steps

Hi [Name],

It was a pleasure walking you through InventoryPro v3 during our session on June 5th. I hope the demo gave you a clear picture of how the platform can streamline your inventory workflows.

Based on our conversation, the logical next step would be a 30-day trial so your team can experience the full feature set firsthand. I'm happy to have everything set up for you within 24 hours.

When you're ready to move forward — or if you have any questions — please don't hesitate to reach out to Sarah directly at sales@inventorypro.com.

Looking forward to hearing from you.

Best regards,
[Your Name]

--- EXAMPLE 2 ---
Intent: Apologize for a delayed project delivery
Key Facts: Project was due March 1, new delivery date is March 15, delay caused by supplier issue, offering 10% discount as goodwill
Tone: Apologetic, professional

Subject: Update on Project Timeline — Revised Delivery Date

Dear [Client Name],

I want to address the delay in delivering your project, which was originally scheduled for March 1st, and offer my sincere apologies for the inconvenience this has caused.

The delay stems from an unexpected supplier issue that impacted our production timeline. We have resolved the matter, and I can confirm a new delivery date of March 15th.

As a goodwill gesture and acknowledgment of the disruption, we are applying a 10% discount to your invoice.

I appreciate your patience and understanding, and I am committed to ensuring the final deliverable exceeds your expectations.

Sincerely,
[Your Name]

--- END OF EXAMPLES ---
"""

SYSTEM_PROMPT_B = """You are a helpful assistant that writes professional emails.
Write a complete email based on the user's inputs. Include a subject line."""


def _call_groq(messages: list, model: str) -> str:
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 800,
    }
    resp = requests.post(GROQ_URL, headers=headers, json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


def generate_email_model_a(intent: str, facts: str, tone: str) -> str:
    """Model A: llama3-70b with advanced prompting (Role-Play + Few-Shot + CoT)."""
    user_msg = (
        f"Intent: {intent}\n"
        f"Key Facts: {facts}\n"
        f"Tone: {tone}\n\n"
        "Now write the email following the system instructions exactly."
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_A},
        {"role": "user", "content": user_msg},
    ]
    return _call_groq(messages, model="llama3-70b-8192")


def generate_email_model_b(intent: str, facts: str, tone: str) -> str:
    """Model B: llama3-8b with simple/baseline prompting."""
    user_msg = (
        f"Write a professional email.\n"
        f"Purpose: {intent}\n"
        f"Include these facts: {facts}\n"
        f"Tone should be: {tone}"
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_B},
        {"role": "user", "content": user_msg},
    ]
    return _call_groq(messages, model="llama3-8b-8192")


if __name__ == "__main__":
    # Quick smoke test
    sample = {
        "intent": "Request a meeting to discuss Q3 budget planning",
        "facts": "Meeting needed before August 15, budget is $2M, stakeholders are CFO and VP Sales, prefer Tuesday afternoon",
        "tone": "Professional, concise",
    }
    print("=== Model A Output ===")
    print(generate_email_model_a(**sample))
    print("\n=== Model B Output ===")
    print(generate_email_model_b(**sample))
