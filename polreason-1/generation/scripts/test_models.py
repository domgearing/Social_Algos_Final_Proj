#!/usr/bin/env python3
"""
test_gpt5_where_answer.py

Goal: Ask ONE simple numeric question to openai/gpt-5 via OpenRouter
and print exactly where the answer shows up in choices[0].message.
"""

import os
import json
import requests
from pprint import pprint

API_URL = "https://openrouter.ai/api/v1/chat/completions"
API_KEY = os.getenv("OPENROUTER_API_KEY")

if not API_KEY:
    raise RuntimeError("Set OPENROUTER_API_KEY in your environment first.")


def call_gpt5_with_string_content():
    """Call GPT-5 using a plain string 'content' (like your main script)."""
    print("\n=== CALL 1: content as plain string ===\n")

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    prompt = (
        "You are taking a survey.\n"
        "Question: Choose option 3.\n"
        "Options: 1, 2, 3, 4.\n\n"
        "Respond with ONLY the number 3. No words, no punctuation."
    )

    payload = {
    "model": "openai/gpt-5",
    "messages": [{"role": "user", "content": prompt}],
    }

    resp = requests.post(API_URL, headers=headers, json=payload, timeout=30)
    print("Status code:", resp.status_code)
    print()

    try:
        data = resp.json()
    except Exception as e:
        print("Could not parse JSON:", e)
        print("Raw text:\n", resp.text)
        return

    print("=== Full JSON (truncated) ===")
    print(json.dumps(data, indent=2)[:2000])
    print()

    if "error" in data:
        print("ERROR FIELD PRESENT:")
        pprint(data["error"])
        return

    try:
        msg = data["choices"][0]["message"]
    except (KeyError, IndexError) as e:
        print("No choices[0].message found:", e)
        return

    print("=== message object (exact) ===")
    print(json.dumps(msg, indent=2))
    print()

    print("message keys:", list(msg.keys()))
    print("type(message['content']):", type(msg.get("content")))
    print("content:", repr(msg.get("content")))
    print("reasoning:", repr(msg.get("reasoning")))
    print("reasoning_details:", repr(msg.get("reasoning_details")))
    print()


def call_gpt5_with_list_content():
    """Call GPT-5 using the OpenRouter-style list content (for comparison)."""
    print("\n=== CALL 2: content as list of {type: text} ===\n")

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": "openai/gpt-5",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "You are taking a survey.\n"
                            "Question: Choose option 3.\n"
                            "Options: 1, 2, 3, 4.\n\n"
                            "Respond with ONLY the number 3. No words, no punctuation."
                        ),
                    }
                ],
            }
        ],
        "max_tokens": 50,
        "temperature": 0.0,
    }

    resp = requests.post(API_URL, headers=headers, json=payload, timeout=30)
    print("Status code:", resp.status_code)
    print()

    try:
        data = resp.json()
    except Exception as e:
        print("Could not parse JSON:", e)
        print("Raw text:\n", resp.text)
        return

    print("=== Full JSON (truncated) ===")
    print(json.dumps(data, indent=2)[:2000])
    print()

    if "error" in data:
        print("ERROR FIELD PRESENT:")
        pprint(data["error"])
        return

    try:
        msg = data["choices"][0]["message"]
    except (KeyError, IndexError) as e:
        print("No choices[0].message found:", e)
        return

    print("=== message object (exact) ===")
    print(json.dumps(msg, indent=2))
    print()

    print("message keys:", list(msg.keys()))
    print("type(message['content']):", type(msg.get("content")))
    print("content:", repr(msg.get("content")))
    print("reasoning:", repr(msg.get("reasoning")))
    print("reasoning_details:", repr(msg.get("reasoning_details")))
    print()


def main():
    call_gpt5_with_string_content()
    call_gpt5_with_list_content()


if __name__ == "__main__":
    main()
