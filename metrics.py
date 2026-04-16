"""
Evaluation Metrics — Three Custom Metrics for Email Quality
===========================================================

Metric 1: FACT RECALL SCORE (Automated Python)
  Logic: Tokenize key facts into individual fact-nuggets (split by comma).
  For each nugget, check if semantically key words appear in the generated email.
  Score = facts_present / total_facts  → range [0, 1]
  Rationale: An email that omits stated facts fails its core purpose.

Metric 2: TONE ALIGNMENT SCORE (LLM-as-a-Judge via Groq)
  Logic: Ask an LLM judge to rate on a 1–5 scale whether the email's tone
  matches the requested tone. Normalised to [0, 1].
  Rationale: Tone is subjective and nuanced — LLM judgment is more reliable
  than keyword matching for capturing formality, warmth, urgency, etc.

Metric 3: PROFESSIONAL FLUENCY SCORE (Automated Python)
  Logic: Combination of three sub-signals, averaged:
    a) Avg sentence length in [10, 30] words → ideal zone for professional email
    b) Absence of filler phrases (penalty per occurrence)
    c) Subject line presence (binary)
  Score → range [0, 1]
  Rationale: Fluency and professionalism are rule-checkable at a surface level
  without needing expensive models.
"""

import re
import json
import os
import requests

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

FILLER_PHRASES = [
    "i hope this email finds you well",
    "i hope this finds you well",
    "please feel free",
    "do not hesitate to",
    "at your earliest convenience",
    "per my last email",
    "going forward",
    "touching base",
    "circling back",
    "as per",
    "kindly",
    "i am writing to",
]


# ─────────────────────────────────────────────────────────
# METRIC 1: FACT RECALL SCORE
# ─────────────────────────────────────────────────────────

def _extract_keywords(fact_nugget: str) -> list[str]:
    """Extract meaningful keywords from a fact phrase (skip stop words)."""
    stop = {
        "a", "an", "the", "is", "are", "was", "were", "to", "of", "for",
        "and", "or", "in", "on", "at", "by", "be", "it", "its", "from",
        "with", "that", "this", "as", "our", "their", "his", "her", "we",
        "they", "i", "you", "he", "she", "will", "can", "has", "have",
        "had", "not", "but", "so", "if", "about", "up", "out"
    }
    words = re.findall(r"[a-zA-Z0-9\$\%]+", fact_nugget.lower())
    return [w for w in words if w not in stop and len(w) > 2]


def fact_recall_score(email_text: str, key_facts: str) -> dict:
    """
    Returns score in [0, 1] and a breakdown per fact nugget.
    """
    email_lower = email_text.lower()
    nuggets = [n.strip() for n in key_facts.split(",") if n.strip()]
    results = []
    for nugget in nuggets:
        keywords = _extract_keywords(nugget)
        if not keywords:
            continue
        # A nugget is "recalled" if ≥60% of its keywords appear in the email
        hits = sum(1 for kw in keywords if kw in email_lower)
        recalled = (hits / len(keywords)) >= 0.6
        results.append({"nugget": nugget, "recalled": recalled, "keyword_hits": hits, "total_keywords": len(keywords)})
    
    if not results:
        return {"score": 0.0, "details": []}
    
    score = sum(1 for r in results if r["recalled"]) / len(results)
    return {"score": round(score, 3), "details": results}


# ─────────────────────────────────────────────────────────
# METRIC 2: TONE ALIGNMENT SCORE (LLM-as-a-Judge)
# ─────────────────────────────────────────────────────────

def tone_alignment_score(email_text: str, requested_tone: str) -> dict:
    """
    Uses Groq LLM as a judge to rate tone alignment 1–5, normalised to [0, 1].
    """
    judge_prompt = f"""You are an expert email writing coach. Your job is to evaluate how well an email's tone matches the requested tone.

Requested tone: {requested_tone}

Email to evaluate:
---
{email_text}
---

Rate the tone alignment on a scale from 1 to 5:
1 = Completely wrong tone (e.g., casual when formal was requested)
2 = Slightly off — mostly wrong tone
3 = Partially matches — some elements correct, some off
4 = Mostly matches — minor deviations only
5 = Perfect match — the tone is exactly as requested

Respond ONLY with a JSON object in this exact format:
{{"score": <integer 1-5>, "reason": "<one sentence explanation>"}}"""

    try:
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": "llama3-70b-8192",
            "messages": [{"role": "user", "content": judge_prompt}],
            "temperature": 0.1,
            "max_tokens": 150,
        }
        resp = requests.post(GROQ_URL, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"].strip()
        # Strip markdown fences if present
        raw = re.sub(r"```[a-z]*\n?", "", raw).strip("` \n")
        parsed = json.loads(raw)
        raw_score = int(parsed.get("score", 3))
        normalized = round((raw_score - 1) / 4, 3)
        return {"score": normalized, "raw_score": raw_score, "reason": parsed.get("reason", "")}
    except Exception as e:
        return {"score": 0.5, "raw_score": -1, "reason": f"Judge call failed: {str(e)}"}


# ─────────────────────────────────────────────────────────
# METRIC 3: PROFESSIONAL FLUENCY SCORE
# ─────────────────────────────────────────────────────────

def professional_fluency_score(email_text: str) -> dict:
    """
    Automated scoring based on three sub-signals, each [0, 1], averaged.
    """
    # Sub-signal A: Subject line present
    has_subject = 1.0 if re.search(r"(?i)^subject\s*:", email_text, re.MULTILINE) else 0.0

    # Sub-signal B: Sentence length in ideal zone (10–30 words)
    sentences = re.split(r"(?<=[.!?])\s+", email_text.strip())
    sentences = [s for s in sentences if len(s.split()) > 3]  # filter fragments
    if sentences:
        ideal = sum(1 for s in sentences if 10 <= len(s.split()) <= 30)
        length_score = round(ideal / len(sentences), 3)
    else:
        length_score = 0.0

    # Sub-signal C: Filler phrase absence
    email_lower = email_text.lower()
    filler_count = sum(1 for fp in FILLER_PHRASES if fp in email_lower)
    filler_score = max(0.0, round(1.0 - (filler_count * 0.25), 3))  # -0.25 per filler

    overall = round((has_subject + length_score + filler_score) / 3, 3)
    return {
        "score": overall,
        "has_subject_line": bool(has_subject),
        "sentence_length_score": length_score,
        "filler_penalty_score": filler_score,
        "filler_phrases_found": filler_count,
    }


# ─────────────────────────────────────────────────────────
# COMBINED SCORER
# ─────────────────────────────────────────────────────────

def evaluate_email(email_text: str, key_facts: str, requested_tone: str) -> dict:
    """Run all three metrics and return combined results."""
    m1 = fact_recall_score(email_text, key_facts)
    m2 = tone_alignment_score(email_text, requested_tone)
    m3 = professional_fluency_score(email_text)
    
    composite = round((m1["score"] + m2["score"] + m3["score"]) / 3, 3)
    
    return {
        "metric_1_fact_recall": m1,
        "metric_2_tone_alignment": m2,
        "metric_3_professional_fluency": m3,
        "composite_score": composite,
    }


if __name__ == "__main__":
    sample_email = """Subject: Following Up on the Website Redesign Proposal

Dear [Name],

I wanted to follow up on the website redesign proposal I sent over two weeks ago.
The estimated project value is $18,000 and I believe we are well-positioned to 
deliver exceptional results for Harmon & Associates.

If you have any questions, I would be happy to schedule a call at your convenience.

Best regards,
[Your Name]"""
    
    result = evaluate_email(
        email_text=sample_email,
        key_facts="Proposal sent two weeks ago, project is website redesign, estimated value $18,000, client is Harmon & Associates, willing to schedule a call",
        requested_tone="Polite, persistent, professional"
    )
    print(json.dumps(result, indent=2))
