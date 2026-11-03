"""
Fixture support tickets and the (fake) backend the agent investigates.

The orders table and refund policy are what the agent's tools read. The
tickets span the three outcomes we want to demonstrate:
  - a small, in-policy refund that should AUTO-APPROVE
  - a large refund that must PAUSE for a human
  - a policy-edge refund (outside the return window) that must PAUSE
"""

ORDERS = {
    "A1021": {"item": "USB-C cable", "price": 12.99, "days_since_delivery": 5,
              "status": "delivered"},
    "A1044": {"item": "4K monitor", "price": 329.00, "days_since_delivery": 9,
              "status": "delivered"},
    "A1067": {"item": "mechanical keyboard", "price": 89.00, "days_since_delivery": 41,
              "status": "delivered"},
}

# Refund policy the agent checks against.
RETURN_WINDOW_DAYS = 30
AUTO_APPROVE_LIMIT = 50.00   # refunds at/under this, in-policy, need no human

TICKETS = [
    {
        "id": "T-501",
        "order_id": "A1021",
        "message": "The USB-C cable I got is defective — only charges "
        "intermittently. I'd like a refund please.",
    },
    {
        "id": "T-502",
        "order_id": "A1044",
        "message": "My 4K monitor arrived with a cracked panel. This is "
        "unacceptable for a $329 item. I want a full refund immediately.",
    },
    {
        "id": "T-503",
        "order_id": "A1067",
        "message": "The keyboard stopped registering keypresses. I know it's "
        "been about 6 weeks but it's clearly a defect — can I get a refund?",
    },
]


def get_order(order_id: str) -> dict | None:
    return ORDERS.get(order_id)
