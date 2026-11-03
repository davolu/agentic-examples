"""
The benchmark — questions where a single confident pass often goes wrong.

These are reasoning traps, multi-step problems, and counterintuitive items:
the kind where a model's first instinct is frequently incorrect but a second
look (or another agent's challenge) catches it. That's exactly the regime
where debate is supposed to help — so it's a fair test of whether it does.

Each question has a canonical answer and a list of accepted equivalents; the
checker normalizes loosely (case, $, commas, trailing words).
"""

QUESTIONS = [
    # ---- cognitive-reflection-test style (intuitive-but-wrong) ----
    {"id": "crt1",
     "q": "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost? Answer in dollars.",
     "answer": "0.05", "accept": ["$0.05", "5 cents", "0.05 dollars", ".05"]},
    {"id": "crt2",
     "q": "If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets? Answer in minutes.",
     "answer": "5", "accept": ["5 minutes", "five"]},
    {"id": "crt3",
     "q": "In a lake there is a patch of lily pads. Every day the patch doubles in size. If it takes 48 days to cover the whole lake, how long would it take to cover half the lake? Answer in days.",
     "answer": "47", "accept": ["47 days", "forty-seven"]},

    # ---- multi-step arithmetic / logic ----
    {"id": "m1",
     "q": "A shirt is on sale for 20% off its original price of $80. You also have a coupon for an additional 10% off the sale price. What is the final price in dollars?",
     "answer": "57.60", "accept": ["$57.60", "57.6", "$57.6", "57.60 dollars"]},
    {"id": "m2",
     "q": "If you overtake the person in second place in a race, what position are you in now?",
     "answer": "second", "accept": ["2nd", "second place", "2"]},
    {"id": "m3",
     "q": "A farmer has 17 sheep. All but 9 die. How many are left?",
     "answer": "9", "accept": ["9 sheep", "nine"]},
    {"id": "m4",
     "q": "Which is heavier: a pound of feathers or a pound of bricks?",
     "answer": "neither", "accept": ["same", "equal", "they weigh the same", "neither, they are equal"]},
    {"id": "m5",
     "q": "How many months have 28 days?",
     "answer": "12", "accept": ["all", "all 12", "twelve", "all of them"]},
    {"id": "m6",
     "q": "Sally has 3 brothers. Each brother has 2 sisters. How many sisters does Sally have?",
     "answer": "1", "accept": ["one", "1 sister"]},

    # ---- comparison / ordering (easy to flip) ----
    {"id": "c1",
     "q": "Which number is larger: 9.9 or 9.11?",
     "answer": "9.9", "accept": ["9.90"]},
    {"id": "c2",
     "q": "How many letter 'r's are in the word 'strawberry'? Answer with a number.",
     "answer": "3", "accept": ["three", "3 rs"]},
    {"id": "c3",
     "q": "If today is Monday, what day will it be 100 days from now?",
     "answer": "wednesday", "accept": ["wed"]},

    # ---- word problems with a twist ----
    {"id": "w1",
     "q": "A bottle and its cork cost $2.20. The bottle costs $2.00 more than the cork. How much is the cork? Answer in dollars.",
     "answer": "0.10", "accept": ["$0.10", "10 cents", ".10", "0.1"]},
    {"id": "w2",
     "q": "There are 6 apples and you take away 4. How many apples do you have?",
     "answer": "4", "accept": ["four", "4 apples"]},
    {"id": "w3",
     "q": "A clock chimes once at 1 o'clock, twice at 2 o'clock, and so on. How many times does it chime in a 12-hour period (1 through 12)?",
     "answer": "78", "accept": ["seventy-eight"]},
    {"id": "w4",
     "q": "If a plane crashes exactly on the border between the US and Canada, where do you bury the survivors?",
     "answer": "nowhere", "accept": ["you don't", "survivors are not buried", "you do not bury survivors", "nowhere, survivors aren't buried"]},
    {"id": "w5",
     "q": "Two fathers and two sons go fishing. Each catches one fish, and they bring home exactly 3 fish total, with none lost. How is this possible? Answer with the number of people.",
     "answer": "3", "accept": ["three", "3 people", "grandfather father son"]},
    {"id": "w6",
     "q": "A snail climbs a 10-foot wall. Each day it climbs 3 feet, and each night it slips back 2 feet. On which day does it reach the top?",
     "answer": "8", "accept": ["day 8", "eighth", "8th"]},
]


def check(answer_text: str, question: dict) -> bool:
    """Loose normalized match against the canonical answer + accepted variants."""
    norm = answer_text.strip().lower().rstrip(".!")
    norm = norm.replace("$", "").replace(",", "").strip()
    targets = [question["answer"]] + question.get("accept", [])
    for t in targets:
        tn = t.strip().lower().replace("$", "").replace(",", "").strip()
        if norm == tn or norm == tn.rstrip("s"):
            return True
        # multi-word accepted phrase → substring match ("they weigh the same"
        # inside "they weigh the same, both are one pound"); single token →
        # whole-word match so "9" doesn't match inside "19".
        if " " in tn:
            if tn in norm:
                return True
        elif tn in norm.split():
            return True
        # numeric equality (e.g. "0.1" vs "0.10")
        try:
            if abs(float(norm) - float(tn)) < 1e-6:
                return True
        except ValueError:
            pass
    return False
