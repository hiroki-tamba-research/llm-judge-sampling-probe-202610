"""Does the Responses API behave differently from chat completions?

Inspect uses the Responses API by default for GPT-5 and later, so the
documentation claim has to be checked on that path, not only on
/chat/completions.
"""
from openai import OpenAI

c = OpenAI()
MODELS = ["gpt-4o", "gpt-5", "gpt-5.4", "gpt-5.5", "gpt-5.6-sol", "gpt-6.1-sol"]


def short(e):
    s = str(e).replace("\n", " ")
    return s[:150]


for m in MODELS:
    out = []
    # plain temperature
    try:
        c.responses.create(model=m, input="Reply with the single letter C.",
                           temperature=0, max_output_tokens=16)
        out.append("temp=0 ACCEPTED")
    except Exception as e:
        out.append("temp=0 -> " + short(e))
    # temperature with reasoning turned down, the path the docs suggest
    try:
        c.responses.create(model=m, input="Reply with the single letter C.",
                           temperature=0, max_output_tokens=16,
                           reasoning={"effort": "minimal"})
        out.append("temp=0 + effort=minimal ACCEPTED")
    except Exception as e:
        out.append("temp=0 + effort=minimal -> " + short(e))
    print(f"--- {m}")
    for o in out:
        print("   ", o)
