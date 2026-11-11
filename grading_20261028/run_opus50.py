"""Opus 5.5 / 4.8, default configuration (temperature unset), N=50 per item.

Tightens the bound from the N=10 run. Writes a NEW file; nothing under
the repository's data/ is touched. Protocol otherwise identical to
repro_v11_extended: same 7 items, same prompt, same extraction regex.
"""
import os, sys, json, collections
from concurrent.futures import ThreadPoolExecutor

SRC = r"C:\Users\hirok\llm-judge-nondeterminism\src"
sys.path.insert(0, SRC)
from repro_v11_extended import ITEMS, _prompt, _extract  # noqa: E402

from anthropic import Anthropic
client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

MODELS = ["claude-opus-5-5", "claude-opus-4-8"]
N = 50


def one(item, model):
    try:
        r = client.messages.create(
            model=model, max_tokens=1024,
            messages=[{"role": "user", "content": _prompt(item)}])
        text = "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
        return {"grade": _extract(text), "raw": text, "resolved": getattr(r, "model", model)}
    except Exception as e:
        return {"grade": "ERROR", "raw": str(e), "resolved": model}


raw = []
summary = []

for model in MODELS:
    tasks = [(i, item) for i, item in enumerate(ITEMS, 1) for _ in range(N)]
    with ThreadPoolExecutor(max_workers=8) as ex:
        out = list(ex.map(lambda t: (t[0], one(t[1], model)), tasks))
    per = collections.defaultdict(collections.Counter)
    for i, res in out:
        per[i][res["grade"]] += 1
        raw.append({"label": f"{model} default N={N}", "model": model, "item": i,
                    "temperature": None, "grade": res["grade"], "raw": res["raw"]})
    flips = sum(1 for c in per.values() if len([g for g in c if g in ("C", "I")]) > 1)
    errs = sum(c.get("ERROR", 0) for c in per.values())
    summary.append({"model": model, "condition": f"default (temperature unset), N={N}",
                    "calls": len(out), "errors": errs,
                    "nonreproducible_items": flips, "total_items": len(ITEMS),
                    "per_item": {i: dict(c) for i, c in sorted(per.items())}})
    print(f"{model}: {flips}/{len(ITEMS)} non-reproducible, {errs} errors, {len(out)} calls")
    for i, c in sorted(per.items()):
        print(f"   item{i}: {dict(c)}")

d = os.path.dirname(os.path.abspath(__file__))
json.dump(raw, open(os.path.join(d, "opus_raw_n50_20261028.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
json.dump(summary, open(os.path.join(d, "opus_summary_n50_20261028.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print("\nTOTAL calls:", len(raw))
