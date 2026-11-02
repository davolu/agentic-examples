# Agentic Examples

A collection of small, runnable examples of agentic AI patterns built on the Anthropic Claude API.

## Examples

| Folder | Pattern | Description |
|--------|---------|-------------|
| [`reflectio-agent/`](reflectio-agent/) | Reflection | An agent that critiques and revises its own output in a generate → critique → revise loop until a strict critic approves or an iteration cap is hit. |

More examples coming.

## Running an example

Each folder is self-contained with its own `requirements.txt`:

```bash
cd <example-folder>
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...
python <main script>.py
```
