"""
Corpus for the GraphRAG demo: NovaTech customer support tickets.

The point of GraphRAG is GLOBAL / sensemaking questions — "what are the main
recurring problems?" — where the answer is distributed across the whole corpus
and sits in NO single document. So the tickets are seeded with a known set of
recurring THEMES, spread across many tickets, with no ticket naming them all.
THEMES is the ground truth: we score each approach on how many it recovers.

Vector RAG can only see its top-k tickets, so it structurally misses themes.
GraphRAG clusters the complaint entities into communities, summarizes each, and
aggregates — which is exactly what a global question needs.
"""

# Ground-truth recurring themes (what a perfect global answer should surface).
THEMES = {
    "billing": "Billing confusion — surprise charges, no proration when "
    "cancelling mid-cycle, invoices hard to understand.",
    "export": "Data export is slow or breaks — the 24-hour download link "
    "expires too soon and large exports time out.",
    "integrations": "Integration gaps and flakiness — missing Salesforce/Teams, "
    "Zapier connections drop, GitHub sync lags.",
    "onboarding": "Onboarding friction — confusing initial setup, unclear docs, "
    "hard to invite teammates.",
    "api_limits": "API rate limits too low — Pro's 1,000 req/min isn't enough "
    "and users keep hitting HTTP 429.",
    "support": "Slow support response — email-only support takes days, no phone "
    "or live chat option.",
    "sso": "Security/SSO gaps — SSO locked to Enterprise, and two-factor setup "
    "is confusing.",
}

# Each ticket touches 1–2 themes (noted in `themes`), phrased as a real ticket
# would be — never listing the theme by name. No ticket covers everything.
TICKETS = {
    "t01": {"themes": ["billing"], "text":
        "I cancelled my Pro plan on the 3rd and was still charged the full "
        "month. Nobody told me the current month isn't refunded. Felt like a "
        "surprise charge — can I get that money back?"},
    "t02": {"themes": ["export"], "text":
        "I requested a full CSV export of our data, got the email link, but by "
        "the time the download finished it said the link had expired. The 24-hour "
        "window is way too short for an export this big."},
    "t03": {"themes": ["integrations"], "text":
        "We live in Salesforce and NovaTech still has no Salesforce connector. "
        "We're copying records by hand. When is this coming?"},
    "t04": {"themes": ["onboarding"], "text":
        "Spent my whole first afternoon just trying to figure out where to start. "
        "The setup wizard skips steps and the getting-started doc is out of date."},
    "t05": {"themes": ["api_limits"], "text":
        "Our integration keeps getting 429 errors during business hours. 1,000 "
        "requests a minute on Pro sounds like a lot until you have real traffic."},
    "t06": {"themes": ["support"], "text":
        "Opened a ticket on Monday, it's now Thursday and still no reply. There's "
        "no phone number and no live chat anywhere. This is painful when you're "
        "blocked."},
    "t07": {"themes": ["sso"], "text":
        "We need SSO for our team but it's gated behind Enterprise. We're happy on "
        "Pro otherwise — why is single sign-on an enterprise-only feature?"},
    "t08": {"themes": ["billing", "support"], "text":
        "My invoice has a line item I don't recognize and I can't get anyone to "
        "explain it. Emailed support two days ago, silence. The billing page "
        "itself is also really hard to read."},
    "t09": {"themes": ["export"], "text":
        "Exporting more than a few months of data just spins and eventually "
        "times out. I've tried JSON and CSV, both fail for large ranges."},
    "t10": {"themes": ["integrations"], "text":
        "Our Zapier zap connected to NovaTech keeps disconnecting every couple of "
        "days and we have to re-authorize it. Really unreliable."},
    "t11": {"themes": ["onboarding", "integrations"], "text":
        "Getting started was rough — I couldn't figure out how to connect our "
        "tools. Found the Slack integration eventually but the docs never "
        "mentioned where it lives."},
    "t12": {"themes": ["api_limits"], "text":
        "Please raise the API limits or let us buy more headroom. We hit the rate "
        "cap every afternoon and orders back up behind 429s."},
    "t13": {"themes": ["sso", "onboarding"], "text":
        "Trying to set up two-factor auth for the team and the flow is confusing — "
        "half my colleagues gave up. Onboarding new people securely is harder than "
        "it should be."},
    "t14": {"themes": ["billing"], "text":
        "I was charged again the day after I cancelled. The cancellation clearly "
        "didn't stop the next charge like the help page promised."},
    "t15": {"themes": ["support"], "text":
        "Love the product but support response times are rough. Three days for a "
        "first reply on a blocking issue is too slow for a paid plan."},
    "t16": {"themes": ["integrations"], "text":
        "Any chance of a Microsoft Teams integration? Half our company is on Teams "
        "and we can't pipe NovaTech notifications there."},
    "t17": {"themes": ["export", "support"], "text":
        "Our export failed three times and when I asked support for help it took "
        "days to hear back. By then the link had expired again anyway."},
    "t18": {"themes": ["onboarding"], "text":
        "The docs really need work. I followed the quickstart and three of the "
        "steps didn't match the actual UI. New users are going to bounce."},
    "t19": {"themes": ["api_limits", "integrations"], "text":
        "Our GitHub sync lags by several minutes and sometimes we blow the API "
        "limit when it catches up, throwing 429s. The two problems compound."},
    "t20": {"themes": ["billing"], "text":
        "Can you make invoices clearer? I genuinely can't tell what I'm paying for "
        "month to month — line items are cryptic and there's no summary."},
    "t21": {"themes": ["sso"], "text":
        "SSO being Enterprise-only is a dealbreaker for our security team. We'd "
        "upgrade from Free to Pro today if SAML were included on Pro."},
    "t22": {"themes": ["support", "onboarding"], "text":
        "First week has been frustrating: setup was confusing and when I reached "
        "out for help it took forever to get a response. Nearly gave up."},
    "t23": {"themes": ["export"], "text":
        "Feature request: please email the export as an attachment or keep the "
        "link alive longer than a day. Expiring in 24 hours keeps biting us."},
    "t24": {"themes": ["integrations"], "text":
        "Jira integration is fine but we really need Salesforce and Teams too. The "
        "current integration list is too short for a team our size."},
    "t25": {"themes": ["api_limits"], "text":
        "The 429 responses are our number-one complaint internally. The Pro rate "
        "limit just doesn't fit a production workload."},
}

QUESTIONS = [
    {"id": "global", "type": "global",
     "q": "Based on all the support tickets, what are the main recurring "
     "problems customers are reporting?",
     # ground truth = the full theme set
     "truth_themes": list(THEMES.keys())},
    {"id": "local", "type": "local",
     "q": "What specifically are customers saying about data export?",
     "truth": "The 24-hour export download link expires too soon, and large "
     "exports (JSON or CSV) are slow and time out."},
]
