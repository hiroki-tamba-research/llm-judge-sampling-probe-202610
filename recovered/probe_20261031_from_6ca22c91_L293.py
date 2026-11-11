"""Second dated observation of the withdrawal map (Table tab:instruments).

Probes both vendors in one run and writes dated JSON next to this file:

  probe_anthropic_<YYYYMMDD>.json   temperature / logprobs on every listed model
  probe_openai_<YYYYMMDD>.json      chat completions: temperature / logprobs
                                    Responses API: temperature, temperature with
                                    reasoning effort "minimal" and "none"
  probe_summary_<YYYYMMDD>.md       one table per vendor, ready to compare with
                                    the October rows

Each probe is one short call ("Reply with the single letter C."), so the whole
run costs a few cents.  Keys come from ANTHROPIC_API_KEY / OPENAI_API_KEY.

  python probe_nov.py            # both vendors
  python probe_nov.py anthropic  # one vendor
  python probe_nov.py openai
"""
import datetime
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DATE = datetime.date.today().strftime("%Y%m%d")
PROMPT = "Reply with the single letter C."

OPENAI_MODELS = [
    "gpt-4o", "gpt-4.1", "gpt-4.1-mini", "o1", "o3", "o3-mini", "o4-mini",
    "gpt-5", "gpt-5-mini", "gpt-5.1", "gpt-5.2", "gpt-5.4", "gpt-5.5",
    "gpt-5.6-sol", "gpt-6-sol", "gpt-6.1-sol", "gpt-6.1-terra",
]


def short(e, n=200):
    return str(e).replace("\n", " ")[:n]


# ---------------------------------------------------------------- Anthropic
def probe_anthropic():
    from anthropic import Anthropic
    c = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    msg = [{"role": "user", "content": PROMPT}]
    rows = []
    for m in [x.id for x in c.models.list(limit=100)]:
        r = {"model": m, "date": DATE}
        try:
            c.messages.create(model=m, max_tokens=16, messages=msg, temperature=0)
            r["temperature"] = "accepted"
        except Exception as e:
            s = str(e)
            r["temperature"] = "400 deprecated" if "deprecated" in s else "error"
            r["temperature_msg"] = short(e)
        try:
            c.messages.create(model=m, max_tokens=16, messages=msg,
                              extra_body={"logprobs": True})
            r["logprobs"] = "accepted"
        except Exception as e:
            s = str(e)
            r["logprobs"] = "400 not permitted" if "not permitted" in s else "error"
            r["logprobs_msg"] = short(e)
        rows.append(r)
        print(f"{m:28} temperature={r['temperature']:16} logprobs={r['logprobs']}",
              flush=True)
    out = os.path.join(HERE, f"probe_anthropic_{DATE}.json")
    json.dump(rows, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return rows


# ------------------------------------------------------------------ OpenAI
def probe_openai():
    from openai import OpenAI
    c = OpenAI()
    msg = [{"role": "user", "content": PROMPT}]

    def chat(model, **kw):
        for tok in ({"max_tokens": 5}, {"max_completion_tokens": 32}):
            try:
                c.chat.completions.create(model=model, messages=msg, **tok, **kw)
                return "accepted", ""
            except Exception as e:
                s = str(e)
                if "max_tokens" in s and "max_completion_tokens" in s and "nsupported" in s:
                    continue
                if "does not support" in s and "temperature" in s:
                    return "400 pinned to default", short(e)
                if "logprobs' is not supported" in s:
                    return "400 unsupported", short(e)
                if "403" in s and "not allowed" in s:
                    return "403 not allowed", short(e)
                return "error", short(e)
        return "error", "token param"

    def responses(model, **kw):
        try:
            c.responses.create(model=model, input=PROMPT, max_output_tokens=16, **kw)
            return "accepted", ""
        except Exception as e:
            s = str(e)
            if "does not support" in s and "temperature" in s:
                return "400 pinned to default", short(e)
            if "reasoning" in s and "effort" in s:
                return "400 effort rejected", short(e)
            return "error", short(e)

    rows = []
    for m in OPENAI_MODELS:
        r = {"model": m, "date": DATE}
        r["chat_temperature"], r["chat_temperature_msg"] = chat(m, temperature=0)
        r["chat_logprobs"], r["chat_logprobs_msg"] = chat(m, logprobs=True, top_logprobs=5)
        r["resp_temperature"], r["resp_temperature_msg"] = responses(m, temperature=0)
        r["resp_temp_effort_minimal"], r["resp_temp_effort_minimal_msg"] = responses(
            m, temperature=0, reasoning={"effort": "minimal"})
        r["resp_temp_effort_none"], r["resp_temp_effort_none_msg"] = responses(
            m, temperature=0, reasoning={"effort": "none"})
        rows.append(r)
        print(f"{m:14} chat: T={r['chat_temperature']:22} lp={r['chat_logprobs']:16} "
              f"| responses: T={r['resp_temperature']:22} "
              f"minimal={r['resp_temp_effort_minimal']:22} "
              f"none={r['resp_temp_effort_none']}", flush=True)
    out = os.path.join(HERE, f"probe_openai_{DATE}.json")
    json.dump(rows, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return rows


# ----------------------------------------------------------------- summary
def write_summary(anth, oai):
    lines = [f"# Withdrawal map, probed {DATE}", ""]
    if anth:
        lines += ["## Anthropic (Messages API)", "",
                  "| model | temperature=0 | logprobs |", "|---|---|---|"]
        lines += [f"| {r['model']} | {r['temperature']} | {r['logprobs']} |" for r in anth]
        lines.append("")
    if oai:
        lines += ["## OpenAI", "",
                  "| model | chat temperature=0 | chat logprobs | responses temperature=0 "
                  "| responses T=0 + effort minimal | responses T=0 + effort none |",
                  "|---|---|---|---|---|---|"]
        lines += [f"| {r['model']} | {r['chat_temperature']} | {r['chat_logprobs']} | "
                  f"{r['resp_temperature']} | {r['resp_temp_effort_minimal']} | "
                  f"{r['resp_temp_effort_none']} |" for r in oai]
        lines.append("")
    out = os.path.join(HERE, f"probe_summary_{DATE}.md")
    open(out, "w", encoding="utf-8").write("\n".join(lines))
    print("wrote", out)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    anth = probe_anthropic() if which in ("both", "anthropic") else []
    oai = probe_openai() if which in ("both", "openai") else []
    write_summary(anth, oai)
