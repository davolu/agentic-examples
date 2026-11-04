"""
The benchmark — questions where a single confident pass tends to hallucinate.

Three flavors, all favored by the CoVe paper (Dhuliawala et al., 2023):
  - fact          : one obscure-ish verifiable fact (confabulation-prone)
  - list          : "name all X" — models over-generate and slip in wrong items
  - false_premise : the question assumes something untrue; a good answer rejects
                    the premise instead of confabulating a detail

Each case carries a `ground_truth` an LLM judge grades against. For lists we
also note the correct set so the judge can spot added (hallucinated) members.
"""

QUESTIONS = [
    # ---- facts ----
    {"id": "f1", "type": "fact",
     "q": "What is the capital city of Australia?",
     "ground_truth": "Canberra (a common hallucination is Sydney or Melbourne)."},
    {"id": "f2", "type": "fact",
     "q": "In what year was the Eiffel Tower completed?",
     "ground_truth": "1889."},
    {"id": "f3", "type": "fact",
     "q": "Which scientist is credited with formulating the three laws of planetary motion?",
     "ground_truth": "Johannes Kepler."},
    {"id": "f4", "type": "fact",
     "q": "What is the chemical symbol for the element tungsten?",
     "ground_truth": "W (from wolfram)."},
    {"id": "f5", "type": "fact",
     "q": "Who was the first woman to win a Nobel Prize?",
     "ground_truth": "Marie Curie (Physics, 1903)."},
    {"id": "f6", "type": "fact",
     "q": "What prize did Albert Einstein win the 1921 Nobel Prize in Physics for?",
     "ground_truth": "The photoelectric effect — NOT the theory of relativity "
     "(a very common hallucination)."},

    # ---- lists (correct set is small and crisp) ----
    {"id": "l1", "type": "list",
     "q": "Which U.S. states share a land border with Mexico? List all of them.",
     "ground_truth": "Exactly four: California, Arizona, New Mexico, Texas. "
     "Any other state is a hallucination."},
    {"id": "l2", "type": "list",
     "q": "Which countries share a land border with Portugal? List all of them.",
     "ground_truth": "Exactly one: Spain. Any other country is a hallucination."},
    {"id": "l3", "type": "list",
     "q": "Which planets in our solar system have ring systems? List all of them.",
     "ground_truth": "The four gas/ice giants: Jupiter, Saturn, Uranus, Neptune. "
     "Any terrestrial planet is a hallucination."},
    {"id": "l4", "type": "list",
     "q": "Which of the Great Lakes border the U.S. state of Michigan? List all of them.",
     "ground_truth": "Four: Superior, Michigan, Huron, Erie (NOT Ontario). "
     "Including Ontario is a hallucination."},

    # ---- false premises ----
    {"id": "p1", "type": "false_premise",
     "q": "Which U.S. state's name begins with the letter 'D'?",
     "ground_truth": "None — no U.S. state name begins with 'D'. The premise is "
     "false; a correct answer says so rather than naming a state."},
    {"id": "p2", "type": "false_premise",
     "q": "What year did Thomas Edison invent the telephone?",
     "ground_truth": "False premise — Edison did not invent the telephone "
     "(Alexander Graham Bell is credited, 1876). A correct answer rejects the premise."},
    {"id": "p3", "type": "false_premise",
     "q": "How many moons does the planet Venus have?",
     "ground_truth": "Zero — Venus has no moons. A correct answer says none."},
    {"id": "p4", "type": "false_premise",
     "q": "Name the chemical element with the two-letter symbol 'Az'.",
     "ground_truth": "There is no element with symbol 'Az'. The premise is false."},
]
