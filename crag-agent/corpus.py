"""
Knowledge base + questions for the CRAG demo.

The corpus is a fictional company (NovaTech) — so nothing in it is in the
model's parametric knowledge or on the web; the only way to answer in-corpus
questions is to retrieve them.

The questions split into two groups that drive CRAG's routing:
  in_corpus    — answerable from the KB (evaluator → "correct" → refine + answer)
  web_fact     — NOT in the KB, but a real, verifiable fact the web can answer
                 (evaluator → "incorrect" → discard corpus, web-search, answer)

The second group is the whole point: a corpus-only RAG can't answer them —
it either hallucinates or abstains. CRAG falls back to the web and recovers.
"""

DOCS = {
    "shipping": "NovaTech ships within 2–4 business days. Standard shipping is "
    "free over $50, otherwise $4.99. Express is $12.99 (1–2 days).",
    "returns": "NovaTech accepts returns within 30 days of delivery for a full "
    "refund. Refunds issue 3–5 business days after the item is received.",
    "plans_pro": "The NovaTech Pro plan is $29/month: unlimited projects, 100 GB "
    "storage, email support, advanced analytics. Billed monthly, cancel anytime.",
    "plans_enterprise": "NovaTech Enterprise is custom-priced and adds SSO/SAML, "
    "a dedicated success manager, a 99.9% uptime SLA, and audit logs.",
    "cancellation": "Cancelling a paid NovaTech plan keeps access until the end "
    "of the billing period. The current month is not prorated or refunded.",
    "api_limits": "The NovaTech API allows 100 req/min on Free, 1,000 on Pro, "
    "custom on Enterprise. Over the limit returns HTTP 429.",
    "security": "NovaTech encrypts data at rest and in transit (TLS 1.2+), is "
    "SOC 2 Type II certified, and offers two-factor authentication.",
    "data_export": "Export all your NovaTech data as JSON or CSV from Settings → "
    "Export; the download link expires in 24 hours.",
    "integrations": "NovaTech integrates with Slack, GitHub, Jira, and Zapier "
    "under Settings → Integrations.",
    "trial": "NovaTech offers a 14-day free trial of Pro — no credit card. After "
    "it ends you can upgrade or move to the Free plan.",
}

QUESTIONS = [
    # ---- in corpus (retrieval should be judged "correct") ----
    {"id": "c1", "type": "in_corpus",
     "q": "How long does NovaTech take to ship an order?",
     "ground_truth": "2–4 business days."},
    {"id": "c2", "type": "in_corpus",
     "q": "What does the NovaTech Pro plan cost and include?",
     "ground_truth": "$29/month; unlimited projects, 100 GB storage, email "
     "support, advanced analytics."},
    {"id": "c3", "type": "in_corpus",
     "q": "If I cancel my paid plan mid-month, do I get a refund for the rest of the month?",
     "ground_truth": "No — access continues to end of billing period; the current "
     "month is not prorated or refunded."},
    {"id": "c4", "type": "in_corpus",
     "q": "What's the API rate limit on the Pro plan?",
     "ground_truth": "1,000 requests per minute."},
    {"id": "c5", "type": "in_corpus",
     "q": "How do I export my data?",
     "ground_truth": "JSON or CSV from Settings → Export (link expires in 24h)."},

    # ---- NOT in corpus, but a real fact the web can answer ----
    {"id": "w1", "type": "web_fact",
     "q": "What is the capital city of Australia?",
     "ground_truth": "Canberra."},
    {"id": "w2", "type": "web_fact",
     "q": "What is the chemical symbol for gold?",
     "ground_truth": "Au."},
    {"id": "w3", "type": "web_fact",
     "q": "What is the tallest mountain on Earth above sea level?",
     "ground_truth": "Mount Everest."},
    {"id": "w4", "type": "web_fact",
     "q": "Who wrote the novel 'Pride and Prejudice'?",
     "ground_truth": "Jane Austen."},
    {"id": "w5", "type": "web_fact",
     "q": "In what year did the first human land on the Moon?",
     "ground_truth": "1969."},
]
