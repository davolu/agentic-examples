"""
The golden dataset — hand-labeled support emails.

Each case has ground-truth labels an eval can check exactly (category,
urgency, is_refund) plus a short gist of an acceptable reply for the
LLM-judge. 28 cases across 5 categories, with enough spread to show a
per-category breakdown — and to make a regression in ONE category visible.

Categories: billing | technical | sales | spam | other
Urgency: 1 (whenever) .. 5 (drop everything)
"""

CASES = [
    # ---- billing ----
    {"id": "b1", "email": "I was charged twice for my October subscription — can you refund the duplicate?",
     "category": "billing", "urgency": 3, "is_refund": True,
     "reply_gist": "acknowledge the double charge, confirm the duplicate will be refunded"},
    {"id": "b2", "email": "My invoice shows a $49 charge but my plan is supposed to be $29. What's going on?",
     "category": "billing", "urgency": 3, "is_refund": False,
     "reply_gist": "offer to review the invoice and explain or correct the $20 discrepancy"},
    {"id": "b3", "email": "How do I update the credit card on file for my account?",
     "category": "billing", "urgency": 2, "is_refund": False,
     "reply_gist": "explain where to update the card (billing settings) in steps"},
    {"id": "b4", "email": "I cancelled last week but just got billed again. I want my money back.",
     "category": "billing", "urgency": 4, "is_refund": True,
     "reply_gist": "apologize, confirm cancellation, and process the refund for the post-cancel charge"},
    {"id": "b5", "email": "Can I switch from monthly to annual billing and get the discount?",
     "category": "billing", "urgency": 2, "is_refund": False,
     "reply_gist": "confirm annual switch is possible and explain the discount and proration"},
    {"id": "b6", "email": "Why was I charged sales tax? I'm a tax-exempt nonprofit.",
     "category": "billing", "urgency": 3, "is_refund": False,
     "reply_gist": "ask for the exemption certificate and offer to remove/refund the tax"},

    # ---- technical ----
    {"id": "t1", "email": "The API started returning 500 errors on every request about 20 minutes ago. Our checkout is down.",
     "category": "technical", "urgency": 5, "is_refund": False,
     "reply_gist": "treat as urgent outage, acknowledge, and begin investigating immediately"},
    {"id": "t2", "email": "I can't log in — it says my password is wrong but I'm sure it's right.",
     "category": "technical", "urgency": 3, "is_refund": False,
     "reply_gist": "guide through password reset and check for caps lock / account lockout"},
    {"id": "t3", "email": "The CSV export is missing the last column. Is that a known bug?",
     "category": "technical", "urgency": 2, "is_refund": False,
     "reply_gist": "acknowledge the export issue and offer a workaround or fix timeline"},
    {"id": "t4", "email": "Webhooks stopped firing after I rotated my API key this morning.",
     "category": "technical", "urgency": 4, "is_refund": False,
     "reply_gist": "connect the key rotation to webhook auth and walk through re-configuring"},
    {"id": "t5", "email": "Getting 'rate limit exceeded' but I'm only sending 10 requests a minute.",
     "category": "technical", "urgency": 3, "is_refund": False,
     "reply_gist": "ask for details, check the account's actual limit, and explain the discrepancy"},
    {"id": "t6", "email": "The mobile app crashes on launch after the latest update on my iPhone 14.",
     "category": "technical", "urgency": 4, "is_refund": False,
     "reply_gist": "acknowledge the crash, ask for app/OS version, and offer a workaround"},
    {"id": "t7", "email": "How do I set up single sign-on with Okta for my team?",
     "category": "technical", "urgency": 2, "is_refund": False,
     "reply_gist": "point to the SSO/Okta setup docs and offer to help configure it"},

    # ---- sales ----
    {"id": "s1", "email": "We're a 200-person company evaluating your Enterprise plan. Can we get a demo?",
     "category": "sales", "urgency": 3, "is_refund": False,
     "reply_gist": "offer to schedule an Enterprise demo and connect them with sales"},
    {"id": "s2", "email": "Do you offer volume discounts for more than 50 seats?",
     "category": "sales", "urgency": 2, "is_refund": False,
     "reply_gist": "confirm volume discounts exist and offer a custom quote"},
    {"id": "s3", "email": "What's the difference between the Pro and Enterprise tiers?",
     "category": "sales", "urgency": 2, "is_refund": False,
     "reply_gist": "summarize the key Pro vs Enterprise differences and offer details"},
    {"id": "s4", "email": "Is there a free trial of the paid features before we commit?",
     "category": "sales", "urgency": 2, "is_refund": False,
     "reply_gist": "explain the trial availability and how to start it"},
    {"id": "s5", "email": "Can you send pricing for 500 seats with annual billing? Procurement needs it today.",
     "category": "sales", "urgency": 4, "is_refund": False,
     "reply_gist": "treat as time-sensitive, offer a same-day custom quote for 500 annual seats"},

    # ---- spam ----
    {"id": "p1", "email": "CONGRATULATIONS! You've been selected for a $1000 gift card. Click here to claim now!!!",
     "category": "spam", "urgency": 1, "is_refund": False,
     "reply_gist": "no action needed; this is spam"},
    {"id": "p2", "email": "Increase your website traffic 10x with our SEO backlink package. Reply for pricing.",
     "category": "spam", "urgency": 1, "is_refund": False,
     "reply_gist": "no action needed; unsolicited marketing"},
    {"id": "p3", "email": "Hi, I'm reaching out about a guest post opportunity on your blog with a do-follow link.",
     "category": "spam", "urgency": 1, "is_refund": False,
     "reply_gist": "no action needed; link-building outreach"},
    {"id": "p4", "email": "Your package could not be delivered. Confirm your details at this link to reschedule.",
     "category": "spam", "urgency": 1, "is_refund": False,
     "reply_gist": "no action needed; phishing attempt"},

    # ---- other ----
    {"id": "o1", "email": "Just wanted to say your support team was fantastic last week. Thank you!",
     "category": "other", "urgency": 1, "is_refund": False,
     "reply_gist": "warmly thank the customer for the kind feedback"},
    {"id": "o2", "email": "Do you have any open engineering roles? I love your product.",
     "category": "other", "urgency": 1, "is_refund": False,
     "reply_gist": "thank them and point to the careers page"},
    {"id": "o3", "email": "Can I get a copy of your SOC 2 report for our vendor review?",
     "category": "other", "urgency": 2, "is_refund": False,
     "reply_gist": "explain how to request the SOC 2 report (security/trust page or NDA)"},
    {"id": "o4", "email": "Please delete my account and all my data under GDPR.",
     "category": "other", "urgency": 3, "is_refund": False,
     "reply_gist": "acknowledge the GDPR deletion request and explain the process/timeline"},
    {"id": "o5", "email": "Where can I find your API changelog?",
     "category": "other", "urgency": 1, "is_refund": False,
     "reply_gist": "point to the changelog location"},
]

CATEGORIES = ["billing", "technical", "sales", "spam", "other"]
