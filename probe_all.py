"""Map where each vendor withdrew the sampler and the readout."""
import os, json
from anthropic import Anthropic

c = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
msg = [{"role": "user", "content": "Reply with the single letter C."}]
rows = []

for m in [x.id for x in c.models.list(limit=100)]:
    r = {"model": m}
    try:
        c.messages.create(model=m, max_tokens=16, messages=msg, temperature=0)
        r["temperature"] = "accepted"
    except Exception as e:
        s = str(e)
        r["temperature"] = "400 deprecated" if "deprecated" in s else "error"
        r["temperature_msg"] = s[:160]
    try:
        c.messages.create(model=m, max_tokens=16, messages=msg,
                          extra_body={"logprobs": True})
        r["logprobs"] = "accepted"
    except Exception as e:
        s = str(e)
        r["logprobs"] = "400 not permitted" if "not permitted" in s else "error"
        r["logprobs_msg"] = s[:160]
    rows.append(r)
    print(f"{m:28} temperature={r['temperature']:16} logprobs={r['logprobs']}")

d = os.path.dirname(os.path.abspath(__file__))
json.dump(rows, open(os.path.join(d, "probe_anthropic.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
