"""Opus 4.6 / 4.7 / 4.8, 7 items x N=50, same protocol as src/v3/run_opus50.py.

Conditions: 4.6 default, 4.6 temperature=0, 4.7 default, 4.8 default.
Same 7 items, prompt and extraction regex (imported, read-only, no bytecode written).
anthropic-version: SDK default. Writes one JSON line per call to a NEW file.
Never prints or stores key values.
"""
import sys
sys.dont_write_bytecode = True
import collections, datetime, json, threading
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, r"C:\Users\hirok\llm-judge-nondeterminism\src")
from repro_v11_extended import ITEMS, _prompt, _extract  # noqa: E402

import anthropic
from anth_common import get_key

OUT, SUMMARY = sys.argv[1], sys.argv[2]
N = int(sys.argv[3]) if len(sys.argv) > 3 else 50
KEY_NAME = "ANTHROPIC_API_KEY_NOV"
CONDS = [("claude-opus-4-6", "default", None), ("claude-opus-4-6", "temp=0", 0),
         ("claude-opus-4-7", "default", None), ("claude-opus-4-8", "default", None)]
PRICE_IN, PRICE_OUT = 5.0 / 1e6, 25.0 / 1e6   # USD per token, as given by the user
COST_CAP = 15.0

key, _ = get_key(KEY_NAME)
client = anthropic.Anthropic(api_key=key)
VERSION = client.default_headers.get("anthropic-version")
lock = threading.Lock()
state = {"cost": 0.0, "stop": False}


def one(f, model, cond, temp, i, item):
    if state["stop"]:
        return i, "SKIPPED"
    rec = {"sent_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
           "key_env_var": KEY_NAME, "sdk": "anthropic " + anthropic.__version__,
           "anthropic_version_header_sent": VERSION, "model": model, "condition": cond,
           "temperature": temp, "max_tokens": 1024, "item": i}
    kw = {} if temp is None else {"temperature": temp}
    try:
        raw = client.messages.with_raw_response.create(
            model=model, max_tokens=1024,
            messages=[{"role": "user", "content": _prompt(item)}], **kw)
        r = raw.parse()
        text = "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
        u = r.usage
        rec.update(http_status=raw.http_response.status_code,
                   request_id=raw.http_response.headers.get("request-id"),
                   grade=_extract(text), raw=text, resolved=r.model, stop_reason=r.stop_reason,
                   input_tokens=u.input_tokens, output_tokens=u.output_tokens)
        cost = u.input_tokens * PRICE_IN + u.output_tokens * PRICE_OUT
    except anthropic.APIStatusError as e:
        rec.update(http_status=e.status_code, request_id=e.response.headers.get("request-id"),
                   grade="ERROR", raw=e.response.text)
        cost = 0.0
    except Exception as e:  # connection errors etc.
        rec.update(http_status=None, request_id=None, grade="ERROR", raw=repr(e))
        cost = 0.0
    with lock:
        state["cost"] += cost
        if state["cost"] > COST_CAP:
            state["stop"] = True
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        f.flush()
    return i, rec["grade"]


summary = []
with open(OUT, "x", encoding="utf-8", newline="\n") as f:   # "x": fail if the file exists
    for model, cond, temp in CONDS:
        tasks = [(i, item) for i, item in enumerate(ITEMS, 1) for _ in range(N)]
        with ThreadPoolExecutor(max_workers=8) as ex:
            out = list(ex.map(lambda t: one(f, model, cond, temp, t[0], t[1]), tasks))
        per = collections.defaultdict(collections.Counter)
        for i, g in out:
            per[i][g] += 1
        flips = sum(1 for c in per.values() if len([g for g in c if g in ("C", "I")]) > 1)
        errs = sum(c.get("ERROR", 0) for c in per.values())
        summary.append({"model": model, "condition": f"{cond}, N={N}", "temperature": temp,
                        "calls": len(out), "errors": errs,
                        "skipped": sum(c.get("SKIPPED", 0) for c in per.values()),
                        "nonreproducible_items": flips, "total_items": len(ITEMS),
                        "per_item": {i: dict(c) for i, c in sorted(per.items())}})
        print(f"{model} {cond}: {flips}/{len(ITEMS)} non-reproducible, {errs} errors, "
              f"{len(out)} calls, est. cost so far ${state['cost']:.2f}", flush=True)
        for i, c in sorted(per.items()):
            print(f"   item{i}: {dict(c)}", flush=True)
        if state["stop"]:
            print("STOP: cost cap reached", flush=True)
            break
with open(SUMMARY, "x", encoding="utf-8", newline="\n") as f:
    json.dump({"sdk": "anthropic " + anthropic.__version__, "anthropic_version_header_sent": VERSION,
               "est_cost_usd": round(state["cost"], 4), "conditions": summary}, f,
              ensure_ascii=False, indent=1)
print(f"done. est. cost ${state['cost']:.2f}", flush=True)
