"""
Knowledge base + complexity-labeled questions for the Adaptive-RAG demo.

Adaptive-RAG (Jeong et al., NAACL 2024) routes each query by *complexity* to
one of three pipelines. So the questions are labeled with the route that should
handle them — this label is the ORACLE, used both to score the router and as a
perfect-routing upper bound:

    A  straightforward  → NO retrieval   (model already knows it)
    B  simple           → single-step    (one lookup in the corpus)
    C  complex          → multi-step     (multi-hop: answer spans two docs,
                                           and you must resolve a bridge entity
                                           before you know what to look up)

The NovaTech corpus is a fictional company, so B/C answers can ONLY come from
retrieval; the A questions are real-world facts the model knows without it.
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
    # ---- A: straightforward, model knows it → no retrieval ----
    {"id": "a1", "route": "A", "q": "What is the capital of France?",
     "ground_truth": "Paris."},
    {"id": "a2", "route": "A", "q": "What is 15 multiplied by 12?",
     "ground_truth": "180."},
    {"id": "a3", "route": "A",
     "q": "At sea level, what is the boiling point of water in Celsius?",
     "ground_truth": "100 °C."},
    {"id": "a4", "route": "A", "q": "Who painted the Mona Lisa?",
     "ground_truth": "Leonardo da Vinci."},
    {"id": "a5", "route": "A",
     "q": "What language is primarily spoken in Brazil?",
     "ground_truth": "Portuguese."},

    # ---- B: simple, single lookup in the corpus → single-step ----
    {"id": "b1", "route": "B",
     "q": "How long does NovaTech take to ship an order?",
     "ground_truth": "2–4 business days."},
    {"id": "b2", "route": "B", "q": "What does the NovaTech Pro plan cost?",
     "ground_truth": "$29/month."},
    {"id": "b3", "route": "B", "q": "How do I export my data from NovaTech?",
     "ground_truth": "JSON or CSV from Settings → Export (link expires in 24h)."},
    {"id": "b4", "route": "B", "q": "What is NovaTech's return window?",
     "ground_truth": "30 days from delivery."},
    {"id": "b5", "route": "B", "q": "Which tools does NovaTech integrate with?",
     "ground_truth": "Slack, GitHub, Jira, and Zapier."},

    # ---- C: complex multi-hop → multi-step (bridge entity, then target) ----
    {"id": "c1", "route": "C",
     "q": "What is the API rate limit on the plan that includes a dedicated "
     "success manager?",
     "ground_truth": "Custom (Enterprise) — the dedicated success manager is an "
     "Enterprise feature, and Enterprise API limits are custom."},
    {"id": "c2", "route": "C",
     "q": "If I subscribe to NovaTech's cheapest paid plan, how much storage do "
     "I get?",
     "ground_truth": "100 GB — the cheapest paid plan is Pro ($29/mo), which "
     "includes 100 GB."},
    {"id": "c3", "route": "C",
     "q": "Does the plan that offers SSO/SAML come with a free trial?",
     "ground_truth": "No — SSO/SAML is an Enterprise feature; the 14-day free "
     "trial is for Pro, not Enterprise."},
    {"id": "c4", "route": "C",
     "q": "I'm on the plan with a 99.9% uptime SLA. If I cancel mid-month, am I "
     "refunded for the rest of the month?",
     "ground_truth": "No — the 99.9% SLA is an Enterprise feature, and cancelling "
     "any paid plan does not refund the current month."},
    {"id": "c5", "route": "C",
     "q": "What is the API rate limit for the plan that gives 100 GB of storage?",
     "ground_truth": "1,000 requests/min — 100 GB is the Pro plan, and Pro's API "
     "limit is 1,000 req/min."},
]
