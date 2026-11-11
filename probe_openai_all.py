"""Where did OpenAI withdraw temperature=0 and logprobs?"""
import os, json
from openai import OpenAI

c = OpenAI()
msg = [{"role": "user", "content": "Reply with the single letter C."}]
MODELS = ["gpt-4o", "gpt-4.1", "gpt-4.1-mini", "o1", "o3", "o3-mini", "o4-mini",
          "gpt-5", "gpt-5-mini", "gpt-5.1", "gpt-5.2", "gpt-5.4", "gpt-5.5",
          "gpt-5.6-sol", "gpt-6-sol", "gpt-6.1-sol", "gpt-6.1-terra"]
rows = []


def probe(model, **kw):
    for tok in ({"max_tokens": 5}, {"max_completion_tokens": 32}):
        try:
            c.chat.completions.create(model=model, messages=msg, **tok, **kw)
            return "accepted"
        except Exception as e:
            s = str(e)
            if "max_tokens" in s and "max_completion_tokens" in s and "nsupported" in s:
                continue
            if "does not support" in s and "temperature" in s:
                return "400 pinned to default"
            if "logprobs' is not supported" in s:
                return "400 unsupported"
            return "error: " + s[:90]
    return "error: token param"


for m in MODELS:
    t = probe(m, temperature=0)
    lp = probe(m, logprobs=True, top_logprobs=5)
    rows.append({"model": m, "temperature": t, "logprobs": lp})
    print(f"{m:16} temperature={t:24} logprobs={lp}")

d = os.path.dirname(os.path.abspath(__file__))
json.dump(rows, open(os.path.join(d, "probe_openai.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
