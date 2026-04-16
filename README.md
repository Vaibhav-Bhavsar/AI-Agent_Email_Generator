# 📧 Email Generation Assistant
> AI Engineer Candidate Assessment — Email Generation with LLM Evaluation

A production-quality email generation assistant that takes **Intent + Key Facts + Tone** and generates professional emails using Groq's free API. Includes a full evaluation framework with 3 custom metrics and a comparative model analysis.

---

## 🚀 Features
- **Advanced Prompt Engineering**: Role-Play persona + Few-Shot examples + Chain-of-Thought reasoning
- **Dual Model Comparison**: llama3-70b-8192 (advanced) vs llama3-8b-8192 (baseline)
- **3 Custom Evaluation Metrics**: Fact Recall, Tone Alignment (LLM-as-Judge), Professional Fluency
- **10 Test Scenarios** with human-written reference emails
- **Automated Reports**: CSV, JSON, and text analysis summary

---

## 📁 Project Structure

```
email-assistant/
├── src/
│   ├── generator.py        # Email generation (both models)
│   ├── metrics.py          # 3 custom evaluation metrics
│   └── run_evaluation.py   # Full evaluation pipeline
├── data/
│   └── test_scenarios.json # 10 test cases + human references
├── reports/
│   ├── evaluation_results.csv
│   ├── evaluation_results.json
│   └── analysis_summary.txt
└── README.md
```

---

## ⚙️ Setup

### 1. Get a free Groq API key
Sign up at [console.groq.com](https://console.groq.com) — no credit card required.

### 2. Install dependencies
```bash
pip install requests
```

### 3. Set your API key
```bash
export GROQ_API_KEY="your_groq_api_key_here"
```

### 4. Run the evaluation
```bash
cd src
python run_evaluation.py
```

This will:
- Generate emails from both models for all 10 scenarios
- Score each email with all 3 custom metrics
- Output `reports/evaluation_results.csv`, `.json`, and `analysis_summary.txt`

---

## 🧠 Prompt Engineering Strategy (Model A)

Model A uses three advanced techniques stacked together:

### 1. Role-Play Prompting
The model is given a specific persona — **Victoria Hargrove, Senior Business Communications Specialist** — with defined expertise and writing principles. This anchors tone calibration and prevents generic output.

### 2. Few-Shot Examples
Two complete worked examples are embedded in the system prompt, showing the expected input→output transformation. This demonstrates format, length, subject line inclusion, and fact integration.

### 3. Chain-of-Thought (CoT)
Before generating, the model is instructed to silently work through a 4-step reasoning chain: (1) identify the email's goal, (2) map facts to sentences, (3) calibrate tone, (4) review for conciseness. This improves structured thinking before output.

---

## 📊 Custom Evaluation Metrics

### Metric 1: Fact Recall Score `[0, 1]`
**What it measures**: Did the email include all the key facts the user provided?

**Logic**: Key facts are split by comma into "nuggets." Each nugget's meaningful keywords (stop words removed) are checked against the email text. A nugget is "recalled" if ≥60% of its keywords appear. Score = recalled_nuggets / total_nuggets.

**Implementation**: Pure Python, no API calls needed.

---

### Metric 2: Tone Alignment Score `[0, 1]`
**What it measures**: Does the email's actual tone match the requested tone?

**Logic**: An LLM judge (llama3-70b via Groq) reads the email and the requested tone, then rates alignment 1–5. Score is normalized: (raw_score - 1) / 4.

**Implementation**: LLM-as-a-Judge pattern using Groq free API.

---

### Metric 3: Professional Fluency Score `[0, 1]`
**What it measures**: Is the email well-structured and professional?

**Logic**: Average of three sub-signals:
- `has_subject_line` (binary: 1 if "Subject:" present)
- `sentence_length_score` (% of sentences in the ideal 10–30 word range)
- `filler_penalty_score` (starts at 1.0, -0.25 per cliché phrase found)

**Implementation**: Pure Python regex + string matching.

---

## 📈 Results Summary

| Metric | Model A (70b + Advanced) | Model B (8b + Baseline) |
|--------|--------------------------|--------------------------|
| Fact Recall | **0.828** | 0.656 |
| Tone Alignment | **0.875** | 0.650 |
| Professional Fluency | **0.833** | 0.683 |
| **Composite** | **0.845** | 0.663 |

**Winner: Model A** — with a composite advantage of +0.182.

See `reports/analysis_summary.txt` for the full comparative analysis.

---

## 🔑 API Used
- **[Groq](https://groq.com)** — Free tier, no credit card required
  - Models: `llama3-70b-8192`, `llama3-8b-8192`
  - Used for: email generation + LLM-as-a-Judge evaluation

---

## 📄 License
MIT
