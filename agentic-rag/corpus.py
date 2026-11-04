"""
The knowledge base + the evaluation questions.

The corpus is a small set of docs for a fictional company (NovaTech). The
questions are chosen to span the cases that separate naive RAG from agentic RAG:

  lookup           — answer sits in one obvious doc (naive RAG handles it)
  poorly_phrased   — vague/informal query; good retrieval needs a rewrite
  multi_hop        — answer spans two docs; one retrieval isn't enough
  out_of_scope     — answer is NOT in the corpus; a good agent abstains
  misleading       — query keywords point at the wrong doc
"""

DOCS = {
    "shipping": "NovaTech ships within 2–4 business days. Standard shipping is "
    "free on orders over $50, otherwise $4.99. Express shipping is $12.99 and "
    "arrives in 1–2 business days. Tracking links are emailed when an order ships.",

    "returns": "NovaTech accepts returns within 30 days of delivery for a full "
    "refund. Items must be in original condition. Start a return from Your "
    "Orders → Return item; a prepaid label is provided. Refunds are issued 3–5 "
    "business days after the item is received.",

    "plans_free": "The NovaTech Free plan includes up to 3 projects, community "
    "support, and 1 GB of storage. No credit card required.",

    "plans_pro": "The NovaTech Pro plan is $29/month and includes unlimited "
    "projects, 100 GB of storage, email support, and advanced analytics. Pro is "
    "billed monthly and can be cancelled anytime.",

    "plans_enterprise": "The NovaTech Enterprise plan has custom pricing and adds "
    "SSO/SAML, a dedicated success manager, a 99.9% uptime SLA, and audit logs. "
    "Contact sales for a quote.",

    "cancellation": "When you cancel a paid NovaTech plan, you keep access until "
    "the end of the current billing period. NovaTech does not prorate or refund "
    "partial months — cancelling stops the next charge but the current month is "
    "not refunded.",

    "billing": "NovaTech accepts Visa, Mastercard, and American Express. You can "
    "update your card under Billing → Payment method. Invoices are available "
    "under Billing → History. Annual billing receives a 20% discount.",

    "security": "NovaTech encrypts data at rest and in transit (TLS 1.2+). The "
    "company is SOC 2 Type II certified. Two-factor authentication is available "
    "to all users and required for Enterprise.",

    "api_limits": "The NovaTech API allows 100 requests per minute on Free, 1,000 "
    "on Pro, and custom limits on Enterprise. Exceeding the limit returns HTTP "
    "429; back off and retry after the Retry-After header.",

    "api_auth": "NovaTech API requests authenticate with a bearer token in the "
    "Authorization header. Generate tokens under Settings → API keys. Rotating a "
    "key immediately invalidates the old one.",

    "data_export": "NovaTech lets you export all your data as JSON or CSV from "
    "Settings → Export. Exports are generated within a few minutes and emailed "
    "as a download link that expires in 24 hours.",

    "gdpr": "NovaTech supports GDPR data-deletion requests. Submit one from "
    "Settings → Privacy → Delete my data; deletion completes within 30 days and "
    "is irreversible.",

    "support_hours": "NovaTech support is available Monday–Friday, 9am–6pm ET. "
    "Pro and Enterprise customers also get priority email support with a 4-hour "
    "first-response target during business hours.",

    "integrations": "NovaTech integrates with Slack, GitHub, Jira, and Zapier. "
    "Connect an integration under Settings → Integrations. Each integration can "
    "be scoped to specific projects.",

    "mobile": "The NovaTech mobile app is available for iOS and Android. It "
    "supports viewing projects, notifications, and approvals, but project "
    "creation and billing changes are web-only.",

    "sla": "The NovaTech Enterprise SLA guarantees 99.9% monthly uptime. If "
    "uptime falls below that, customers receive service credits: 10% for "
    "99.0–99.9%, and 25% below 99.0%.",

    "onboarding": "New NovaTech Enterprise accounts include a guided onboarding: "
    "a kickoff call, data migration assistance, and admin training, typically "
    "completed within the first two weeks.",

    "trial": "NovaTech offers a 14-day free trial of the Pro plan — no credit "
    "card required. At the end of the trial you can upgrade or automatically "
    "move to the Free plan.",
}

QUESTIONS = [
    {"id": "q1", "type": "lookup",
     "q": "How long does NovaTech take to ship an order?",
     "ground_truth": "2–4 business days (standard)."},
    {"id": "q2", "type": "lookup",
     "q": "What is the NovaTech API rate limit on the Pro plan?",
     "ground_truth": "1,000 requests per minute."},
    {"id": "q3", "type": "poorly_phrased",
     "q": "money back how long i have",
     "ground_truth": "30 days from delivery to return for a full refund."},
    {"id": "q4", "type": "poorly_phrased",
     "q": "can i get my stuff out of there",
     "ground_truth": "Yes — export all your data as JSON or CSV from Settings → Export."},
    {"id": "q5", "type": "multi_hop",
     "q": "I'm on the Pro plan and I cancel mid-month — do I get a refund for the rest of the month?",
     "ground_truth": "No — cancelling keeps access until the end of the billing "
     "period, but the current month is not prorated or refunded."},
    {"id": "q6", "type": "multi_hop",
     "q": "Which plan do I need for SSO, and what does it cost?",
     "ground_truth": "Enterprise (it includes SSO/SAML); Enterprise is custom "
     "pricing — contact sales."},
    {"id": "q7", "type": "out_of_scope",
     "q": "What is NovaTech's current stock price?",
     "ground_truth": "Not in the knowledge base — a correct answer abstains / says it doesn't know."},
    {"id": "q8", "type": "out_of_scope",
     "q": "Who is the CEO of NovaTech?",
     "ground_truth": "Not in the knowledge base — a correct answer abstains / says it doesn't know."},
    {"id": "q9", "type": "out_of_scope",
     "q": "Does NovaTech offer a student discount?",
     "ground_truth": "Not in the knowledge base — a correct answer abstains / says it doesn't know."},
    {"id": "q10", "type": "misleading",
     "q": "How do I rotate my payment method?",
     "ground_truth": "Update your card under Billing → Payment method. (Note: 'rotate' "
     "matches API-key docs, but the question is about the payment method.)"},
    {"id": "q11", "type": "lookup",
     "q": "Is two-factor authentication available?",
     "ground_truth": "Yes — 2FA is available to all users and required for Enterprise."},
    {"id": "q12", "type": "multi_hop",
     "q": "If my Enterprise uptime was 98.5% last month, what do I get?",
     "ground_truth": "A 25% service credit (below 99.0% uptime)."},
]
