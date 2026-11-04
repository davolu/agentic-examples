"""
NovaTech knowledge base + a ChromaDB retriever over it.

This is the product's factual ground truth — the docs the copilot answers from.
Kept small and local (Chroma, local embeddings, no key) so the tutorial runs for
anyone. In a real deployment this is your help centre / docs, re-indexed on edit.
"""

import chromadb

KB = {
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
    "billing": "NovaTech accepts Visa, Mastercard, and American Express. Invoices "
    "are emailed monthly and available under Settings → Billing.",
    "password_reset": "Reset your NovaTech password from the login page via "
    "'Forgot password'; the reset link is valid for 1 hour.",
}

_client = chromadb.Client()
_col = _client.get_or_create_collection("novatech_kb")
if _col.count() == 0:
    _col.add(documents=list(KB.values()), ids=list(KB.keys()))


def retrieve(query: str, k: int = 3) -> list[dict]:
    res = _col.query(query_texts=[query], n_results=k)
    return [{"id": i, "text": t}
            for i, t in zip(res["ids"][0], res["documents"][0])]
