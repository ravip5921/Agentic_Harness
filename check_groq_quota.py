#!/usr/bin/env python3
"""Inspect Groq quota headers without generating tokens."""

import os
import sys
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

# Optional: use the existing project's .env.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def main():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("Set GROQ_API_KEY in your environment or .env.", file=sys.stderr)
        return 1

    request = Request(
        "https://api.groq.com/openai/v1/models",
        headers={"Authorization": f"Bearer {api_key}"},
        method="GET",
    )

    try:
        with urlopen(request, timeout=20) as response:
            headers = dict(
                (name.lower(), value)
                for name, value in response.headers.items()
            )
    except HTTPError as exc:
        print(f"Groq returned HTTP {exc.code}. No inference was requested.")
        headers = {
            name.lower(): value for name, value in exc.headers.items()
        }
    except (URLError, TimeoutError):
        print("Could not connect to Groq.", file=sys.stderr)
        return 1

    # Day-specific headers are not guaranteed by Groq's public documentation.
    daily_remaining = headers.get("x-ratelimit-remaining-tokens-day")
    daily_limit = headers.get("x-ratelimit-limit-tokens-day")

    if daily_remaining is not None:
        print(f"Daily tokens remaining: {daily_remaining}")
        if daily_limit is not None:
            print(f"Daily token limit:     {daily_limit}")
    else:
        print("Daily tokens remaining: NOT EXPOSED by this response.")
        print("Check today's usage and your model's daily limit in Groq Console:")
        print("  https://console.groq.com/dashboard/usage")
        print("  https://console.groq.com/settings/limits")

    print("\nOther quota headers, if available:")
    fields = {
        "x-ratelimit-remaining-tokens": "Tokens remaining this MINUTE",
        "x-ratelimit-limit-tokens": "Tokens-per-minute limit",
        "x-ratelimit-remaining-requests": "Requests remaining per DAY",
        "x-ratelimit-reset-tokens": "Minute token reset",
        "x-ratelimit-reset-requests": "Daily request reset",
    }

    found = False
    for header, label in fields.items():
        if header in headers:
            print(f"  {label}: {headers[header]}")
            found = True

    if not found:
        print("  No documented quota headers returned by the models endpoint.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
