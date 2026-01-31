Conference PoC: RL decision-maker with sparse LLM context augmentation.

- RL (online tabular Q-learning) chooses actions every step.
- LLM is NOT used for decision making. It only produces context (regime + risk flags).
- LLM call frequency: once per episode (after a short warmup telemetry window).
- Dummy non-stationary environment with hidden regimes.
- Logs per-episode metrics to CSV and generates figures (PNG).

Install:
  python3 -m venv .venv
  source .venv/bin/activate     (macOS/Linux)
  .venv\Scripts\activate        (Windows PowerShell)
  pip install -r requirements.txt

Run (mock LLM):
  python3 main.py --provider mock

Run (real OpenAI LLM):
  export OPENAI_API_KEY="your_key"     (macOS/Linux)
  setx OPENAI_API_KEY "your_key"       (Windows PowerShell, new terminal after)
  python3 main.py --provider openai --model gpt-5-mini

Outputs:
  runs/   : CSV logs
  figures/: PNG figures

Notes:
- If OPENAI_API_KEY is missing, the OpenAI provider will raise a clear error.
- For a cheaper “context-only” call, start with gpt-5-mini or gpt-5-nano.
