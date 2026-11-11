import os
from anthropic import Anthropic

c = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
msg = [{"role": "user", "content": "Reply with the single letter C."}]

for name, kw in (("logprobs", {"logprobs": True}),
                 ("top_logprobs", {"top_logprobs": 5})):
    try:
        c.messages.create(model="claude-opus-5-5", max_tokens=16, messages=msg, **kw)
        print(name, "ACCEPTED")
    except TypeError as e:
        print(name, "SDK rejects:", str(e)[:120])
    except Exception as e:
        print(name, "API:", str(e)[:180])

# and via extra_body, which bypasses the SDK's typed signature
try:
    c.messages.create(model="claude-opus-5-5", max_tokens=16, messages=msg,
                      extra_body={"logprobs": True})
    print("extra_body logprobs ACCEPTED")
except Exception as e:
    print("extra_body logprobs:", str(e)[:200])
