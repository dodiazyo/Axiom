# TrendBot

TrendBot is a FastAPI web app for monitoring crypto futures markets and running trend-following / swing-trading logic with simulated, testnet, or real execution modes.

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --host 127.0.0.1 --port 8003
```

Then open:

```text
http://127.0.0.1:8003
```

## Notes

- API credentials are entered from the UI and are not committed to the repository.
- `runtime_state.json` is generated while the bot runs and is intentionally ignored.
- `pattern_memory.json` is included as the current learned pattern-memory dataset.
