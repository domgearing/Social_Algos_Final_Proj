#!/usr/bin/env python3
"""
01_generate_synthetic_GSS_gpt5safe.py

Query LLMs with COMPREHENSIVE attitudinal battery including:
- All 30 original culture-war items (abortion, guns, immigration, etc.)
- 22 new non-culture-war items (spending, trust, economic outlook)

Total: 52 attitudinal items to test whether constraint/inflation patterns
extend beyond hot-button identity politics.

Features:
- ✅ Proper resume functionality (skips already-completed persona/question/run combinations)
- ✅ Retry logic for failed API calls with exponential backoff
- ✅ Year-specific persona loading
- ✅ Batch saving with progress tracking
- ✅ GPT-5-compatible parsing (handles reasoning fields, JSON-ish outputs, etc.)

Usage:
    python 01_generate_synthetic_GSS_gpt5safe.py --year 2024 --all-models --personas 600
    python 01_generate_synthetic_GSS_gpt5safe.py --year 2016 --models openai/gpt-4o-mini --personas 300

Set OPENROUTER_API_KEY environment variable before running.
"""

import os
import json
import time
import random
import re
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from pathlib import Path
import argparse
from typing import List, Dict, Optional, Set, Tuple
import requests
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configuration
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent  # generation/ directory (one level up from scripts/)
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

# === COMPREHENSIVE GSS QUESTIONS ===
# Part 1: Original 30 culture-war items
GSS_QUESTIONS_CULTUREWAR = {
    # Abortion (4)
    "abhlth": {
        "text": "Please tell me whether or not you think it should be possible for a pregnant woman to obtain a legal abortion if the woman's own health is seriously endangered by the pregnancy?",
        "options": {1: "Yes", 2: "No"}
    },
    "abrape": {
        "text": "Please tell me whether or not you think it should be possible for a pregnant woman to obtain a legal abortion if she becomes pregnant as a result of rape?",
        "options": {1: "Yes", 2: "No"}
    },
    "abpoor": {
        "text": "Please tell me whether or not you think it should be possible for a pregnant woman to obtain a legal abortion if the family has a very low income and cannot afford any more children?",
        "options": {1: "Yes", 2: "No"}
    },
    "abdefect": {
        "text": "Please tell me whether or not you think it should be possible for a pregnant woman to obtain a legal abortion if there is a strong chance of serious defect in the baby?",
        "options": {1: "Yes", 2: "No"}
    },
    # Guns/Crime (2)
    "gunlaw": {
        "text": "Would you favor or oppose a law which would require a person to obtain a police permit before he or she could buy a gun?",
        "options": {1: "Favor", 2: "Oppose"}
    },
    "cappun": {
        "text": "Do you favor or oppose the death penalty for persons convicted of murder?",
        "options": {1: "Favor", 2: "Oppose"}
    },
    # Sexual morality (4)
    "homosex": {
        "text": "What is your opinion about sexual relations between two adults of the same sex--do you think it is always wrong, almost always wrong, wrong only sometimes, or not wrong at all?",
        "options": {1: "Always wrong", 2: "Almost always wrong", 3: "Wrong only sometimes", 4: "Not wrong at all"}
    },
    "premarsx": {
        "text": "If a man and woman have sexual relations before marriage, do you think it is always wrong, almost always wrong, wrong only sometimes, or not wrong at all?",
        "options": {1: "Always wrong", 2: "Almost always wrong", 3: "Wrong only sometimes", 4: "Not wrong at all"}
    },
    "xmarsex": {
        "text": "What is your opinion about a married person having sexual relations with someone other than the marriage partner--is it always wrong, almost always wrong, wrong only sometimes, or not wrong at all?",
        "options": {1: "Always wrong", 2: "Almost always wrong", 3: "Wrong only sometimes", 4: "Not wrong at all"}
    },
    "pornlaw": {
        "text": "Which of these statements comes closest to your feelings about pornography laws? 1) There should be laws against the distribution of pornography whatever the age. 2) There should be laws against the distribution of pornography to persons under 18. 3) There should be no laws forbidding the distribution of pornography.",
        "options": {1: "There should be laws against the distribution of pornography whatever the age", 2: "There should be laws against the distribution of pornography to persons under 18.", 3: "There should be no laws forbidding the distribution of pornography"}
    },
    # Drugs
    "grass": {
        "text": "Do you think the use of marijuana should be made legal or not?",
        "options": {1: "Should be legal", 2: "Should not be legal"}
    },
    # Gender roles (3)
    "fechld": {
        "text": "To what extent do you you agree or disagree that: A working mother can establish just as warm and secure a relationship with her children as a mother who does not work.",
        "options": {1: "Strongly agree", 2: "Agree", 3: "Neither agree nor disagree", 4: "Disagree"}
    },
    "fefam": {
        "text": "Please read the following statement and indicate whether you strongly agree, agree, disagree, or strongly disagree: It is much better for everyone involved if the man is the achiever outside the home and the woman takes care of the home and family.",
        "options": {1: "Strongly agree", 2: "Agree", 3: "Disagree", 4: "Strongly disagree"}
    },
    "fepresch": {
        "text": "Please read the following statement and indicate whether you strongly agree, agree, disagree, or strongly disagree: A preschool child is likely to suffer if his or her mother works.",
        "options": {1: "Strongly agree", 2: "Agree", 3: "Disagree", 4: "Strongly disagree"}
    },
    # Economic (2)
    "affrmact": {
        "text": "Some people say that because of past discrimination, Black people should be given preference in hiring and promotion. Others say that such preference in hiring and promotion of Black people is wrong because it discriminates against whites. What about your opinion--are you for or against preferential hiring and promotion of Black people?",
        "options": {1: "Strongly favor", 2: "Not strongly favor", 3: "Not strongly oppose", 4: "Strongly oppose"}
    },
    "eqwlth": {
        "text": "Some people think that the government in Washington ought to reduce the income differences between the rich and the poor, perhaps by raising the taxes of wealthy families or by giving income assistance to the poor. Others think that the government should not concern itself with reducing this income difference between the rich and the poor. Here is a card with a scale from 1 to 7. Think of a score of 1 as meaning that the government ought to reduce the income differences between rich and poor, and a score of 7 meaning that the government should not concern itself with reducing income differences. What score between 1 and 7 comes closest to the way you feel?",
        "options": {
            1: "The Government should reduce differences",
            2: "2",
            3: "3",
            4: "4",
            5: "5",
            6: "6",
            7: "The government should not concern itself with reducing income differences"
        }
    },
    # Immigration (3)
    "immjobs": {
        "text": "There are different opinions about immigrants from other countries living in America. (By immigrants we mean people who come to settle in America.) How much do you agree or disagree with each of the following statement: Immigrants take jobs away from people who were born in America.",
        "options": {1: "Agree strongly", 2: "Agree", 3: "Neither agree nor disagree", 4: "Disagree", 5: "Disagree strongly"}
    },
    "immcrime": {
        "text": "There are different opinions about immigrants from other countries living in America. (By immigrants we mean people who come to settle in America.) How much do you agree or disagree with each of the following statement: Immigrants increase crime rates in America?",
        "options": {1: "Agree strongly", 2: "Agree", 3: "Neither agree nor disagree", 4: "Disagree", 5: "Disagree strongly"}
    },
    "immameco": {
        "text": "There are different opinions about immigrants from other countries living in America. (By immigrants we mean people who come to settle in America.) How much do you agree or disagree with each of the following statement: Immigrants are generally good for America's economy?",
        "options": {1: "Agree strongly", 2: "Agree", 3: "Neither agree nor disagree", 4: "Disagree", 5: "Disagree strongly"}
    },
    # Free speech (6)
    "spkath": {
        "text": "There are always some people whose ideas are considered bad or dangerous by other people. For instance, somebody who is against all churches and religion. If such a person wanted to make a speech in your (city/town/community) against churches and religion, should he be allowed to speak, or not?",
        "options": {1: "Yes, allowed to speak", 2: "Not allowed"}
    },
    "colath": {
        "text": "There are always some people whose ideas are considered bad or dangerous by other people. For instance, somebody who is against all churches and religion. Should such a person be allowed to teach in a college or university, or not?",
        "options": {1: "Yes, allowed to speak", 2: "Not allowed"}
    },
    "spkrac": {
        "text": "Consider a person who believes that Black people are genetically inferior. If such a person wanted to make a speech in your community claiming that Black people are inferior, should he be allowed to speak, or not?",
        "options": {1: "Yes, allowed to speak", 2: "Not allowed"}
    },
    "colrac": {
        "text": "Consider a person who believes that Black people are genetically inferior. Should such a person be allowed to teach in a college or university, or not?",
        "options": {1: "Yes, allowed to teach", 2: "Not allowed"}
    },
    "spkmslm": {
        "text": "Consider a Muslim clergyman who preaches hatred of the United States. If such a person wanted to make a speech in your community preaching hatred of the United States, should he be allowed to speak, or not?",
        "options": {1: "Yes, allowed", 2: "Not allowed"}
    },
    "colmslm": {
        "text": "Consider a Muslim clergyman who preaches hatred of the United States. Should such a person be allowed to teach in a college or university, or not?",
        "options": {1: "Yes, allowed to teach", 2: "Not allowed"}
    },
    # Religion (2)
    "god": {
        "text": "Tell me which statement comes closest to expressing what you believe about God. I don't believe in God; I don't know whether there is a God and I don't believe there is any way to find out; I don't believe in a personal God, but I do believe in a Higher Power of some kind; I find myself believing in God some of the time, but not at others; While I have doubts, I feel that I do believe in God; I know God really exists and I have no doubts about it.",
        "options": {
            1: "I don't believe in God",
            2: "I don't know whether there is a God and I don't believe there is any way to find out",
            3: "I don't believe in a personal God, but I do believe in a Higher Power of some kind",
            4: "I find myself believing in God some of the time, but not at others",
            5: "While I have doubts, I feel that I do believe in God",
            6: "I know God really exists and I have no doubts about it"
        }
    },
    "bible": {
        "text": "Which of these statements comes closest to describing your feelings about the Bible? 1) The Bible is the actual word of God and is to be taken literally, word for word. 2) The Bible is the inspired word of God but not everything in it should be taken literally, word for word. 3) The Bible is an ancient book of fables, legends, history, and moral precepts recorded by men.",
        "options": {
            1: "The Bible is the actual word of God and is to be taken literally, word for word",
            2: "The Bible is the inspired word of God but not everything in it should be taken literally, word for word",
            3: "The Bible is an ancient book of fables, legends, history, and moral precepts recorded by men"
        }
    },
    # Confidence in institutions (3)
    "confed": {
        "text": "I am going to name some institutions in this country. As far as the people running these institutions are concerned, would you say you have a great deal of confidence, only some confidence, or hardly any confidence at all in them?: Executive branch of the federal government.",
        "options": {1: "A great deal", 2: "Only some", 3: "Hardly any"}
    },
    "conpress": {
        "text": "I am going to name some institutions in this country. As far as the people running these institutions are concerned, would you say you have a great deal of confidence, only some confidence, or hardly any confidence at all in them?: The Press.",
        "options": {1: "A great deal", 2: "Only some", 3: "Hardly any"}
    },
    "consci": {
        "text": "I am going to name some institutions in this country. As far as the people running these institutions are concerned, would you say you have a great deal of confidence, only some confidence, or hardly any confidence at all in them?: The Scientific Community.",
        "options": {1: "A great deal", 2: "Only some", 3: "Hardly any"}
    },
}

# Part 2: Non-culture-war items (spending, trust, economic)
GSS_QUESTIONS_NONCULTUREWAR = {
    # === PUBLIC SPENDING PRIORITIES (NAT* series) ===
    "nataid": {
        "text": "We are faced with many problems in this country, none of which can be solved easily or inexpensively. I'm going to name some of these problems, and for each one I'd like you to tell me whether you think we're spending too much money on it, too little money, or about the right amount. Are we spending too much, too little, or about the right amount on: Foreign aid?",
        "options": {1: "Too little", 2: "About right", 3: "Too much"}
    },
    "natspac": {
        "text": "We are faced with many problems in this country, none of which can be solved easily or inexpensively. I'm going to name some of these problems, and for each one I'd like you to tell me whether you think we're spending too much money on it, too little money, or about the right amount. Are we spending too much, too little, or about the right amount on: The space exploration program?",
        "options": {1: "Too little", 2: "About right", 3: "Too much"}
    },
    "natsci": {
        "text": "We are faced with many problems in this country, none of which can be solved easily or inexpensively. I'm going to name some of these problems, and for each one I'd like you to tell me whether you think we're spending too much money on it, too little money, or about the right amount. Are we spending too much, too little, or about the right amount on: Supporting scientific research?",
        "options": {1: "Too little", 2: "About right", 3: "Too much"}
    },
    "natenvir": {
        "text": "We are faced with many problems in this country, none of which can be solved easily or inexpensively. I'm going to name some of these problems, and for each one I'd like you to tell me whether you think we're spending too much money on it, too little money, or about the right amount. Are we spending too much, too little, or about the right amount on: Improving and protecting the environment?",
        "options": {1: "Too little", 2: "About right", 3: "Too much"}
    },
    "nateduc": {
        "text": "We are faced with many problems in this country, none of which can be solved easily or inexpensively. I'm going to name some of these problems, and for each one I'd like you to tell me whether you think we're spending too much money on it, too little money, or about the right amount. Are we spending too much, too little, or about the right amount on: Improving the nation's education system?",
        "options": {1: "Too little", 2: "About right", 3: "Too much"}
    },
    "natheal": {
        "text": "We are faced with many problems in this country, none of which can be solved easily or inexpensively. I'm going to name some of these problems, and for each one I'd like you to tell me whether you think we're spending too much money on it, too little money, or about the right amount. Are we spending too much, too little, or about the right amount on: Improving and protecting the nation's health?",
        "options": {1: "Too little", 2: "About right", 3: "Too much"}
    },
    "natroad": {
        "text": "We are faced with many problems in this country, none of which can be solved easily or inexpensively. I'm going to name some of these problems, and for each one I'd like you to tell me whether you think we're spending too much money on it, too little money, or about the right amount. Are we spending too much, too little, or about the right amount on: Highways and bridges?",
        "options": {1: "Too little", 2: "About right", 3: "Too much"}
    },
    "natsoc": {
        "text": "We are faced with many problems in this country, none of which can be solved easily or inexpensively. I'm going to name some of these problems, and for each one I'd like you to tell me whether you think we're spending too much money on it, too little money, or about the right amount. Are we spending too much, too little, or about the right amount on: Social Security?",
        "options": {1: "Too little", 2: "About right", 3: "Too much"}
    },

    # === INSTITUTIONAL CONFIDENCE (CON* series) ===
    "coneduc": {
        "text": "I am going to name some institutions in this country. As far as the people running these institutions are concerned, would you say you have a great deal of confidence, only some confidence, or hardly any confidence at all in them: Education?",
        "options": {1: "A great deal", 2: "Only some", 3: "Hardly any"}
    },
    "conbus": {
        "text": "I am going to name some institutions in this country. As far as the people running these institutions are concerned, would you say you have a great deal of confidence, only some confidence, or hardly any confidence at all in them: Major companies?",
        "options": {1: "A great deal", 2: "Only some", 3: "Hardly any"}
    },
    "confinan": {
        "text": "I am going to name some institutions in this country. As far as the people running these institutions are concerned, would you say you have a great deal of confidence, only some confidence, or hardly any confidence at all in them: Banks and financial institutions?",
        "options": {1: "A great deal", 2: "Only some", 3: "Hardly any"}
    },
    "conlegis": {
        "text": "I am going to name some institutions in this country. As far as the people running these institutions are concerned, would you say you have a great deal of confidence, only some confidence, or hardly any confidence at all in them: Congress?",
        "options": {1: "A great deal", 2: "Only some", 3: "Hardly any"}
    },
    "conarmy": {
        "text": "I am going to name some institutions in this country. As far as the people running these institutions are concerned, would you say you have a great deal of confidence, only some confidence, or hardly any confidence at all in them: Military?",
        "options": {1: "A great deal", 2: "Only some", 3: "Hardly any"}
    },
    "conmedic": {
        "text": "I am going to name some institutions in this country. As far as the people running these institutions are concerned, would you say you have a great deal of confidence, only some confidence, or hardly any confidence at all in them: Medicine?",
        "options": {1: "A great deal", 2: "Only some", 3: "Hardly any"}
    },

    # === SOCIAL TRUST ===
    "trust": {
        "text": "Generally speaking, would you say that most people can be trusted or that you can't be too careful in dealing with people?",
        "options": {1: "Can trust", 2: "Can't be too careful", 3: "Depends"}
    },
    "fair": {
        "text": "Do you think most people would try to take advantage of you if they got a chance, or would they try to be fair?",
        "options": {1: "Would take advantage of you", 2: "Would try to be fair", 3: "Depends"}
    },
    "helpful": {
        "text": "Would you say that most of the time people try to be helpful, or that they are mostly just looking out for themselves?",
        "options": {1: "Try to be helpful", 2: "Looking out for themselves", 3: "Depends"}
    },

    # === ECONOMIC OUTLOOK & CLASS ===
    "satfin": {
        "text": "We are interested in how people are getting along financially these days. So far as you and your family are concerned, would you say that you are pretty well satisfied with your present financial situation, more or less satisfied, or not satisfied at all?",
        "options": {1: "Pretty well satisfied", 2: "More or less satisfied", 3: "Not satisfied at all"}
    },
    "finalter": {
        "text": "During the last few years, has your financial situation been getting better, worse, or has it stayed the same?",
        "options": {1: "Getting better", 2: "Stayed the same", 3: "Getting worse"}
    },
    "getahead": {
        "text": "Some people say that people get ahead by their own hard work; others say that lucky breaks or help from other people are more important. Which do you think is most important?",
        "options": {1: "Hard work most important", 2: "Hard work, luck equally important", 3: "Luck most important"}
    },

    # === WORK CENTRALITY ===
    "richwork": {
        "text": "If you were to get enough money to live as comfortably as you would like for the rest of your life, would you continue to work or would you stop working?",
        "options": {1: "Continue to work", 2: "Stop working"}
    },
    "satjob": {
        "text": "On the whole, how satisfied are you with the work you do--would you say you are very satisfied, moderately satisfied, a little dissatisfied, or very dissatisfied?",
        "options": {1: "Very satisfied", 2: "Moderately satisfied", 3: "A little dissatisfied", 4: "Very dissatisfied"}
    },
}

# Combine both dictionaries into comprehensive battery
GSS_QUESTIONS_COMPREHENSIVE = {**GSS_QUESTIONS_CULTUREWAR, **GSS_QUESTIONS_NONCULTUREWAR}

# Popular models on OpenRouter (subset + GPT-5)
POPULAR_MODELS = [
    # # some existing models
    # "mistralai/mistral-small-3.2-24b-instruct",
    # "mistralai/mistral-medium-3.1",
    # "mistralai/mistral-nemo",
    # "z-ai/glm-4.6",
    # "qwen/qwen-2.5-72b-instruct",
    # "arcee-ai/trinity-mini",
    # "anthropic/claude-3.7-sonnet",
    # "anthropic/claude-sonnet-4.5",
    # "x-ai/grok-4-fast",

    # --- OpenAI 5-series ---
    "openai/gpt-5-mini",
    "openai/gpt-5",

    # # Gemini tiers
    # "google/gemini-2.5-flash-lite",

    # # Newer DeepSeek chat models
    # "deepseek/deepseek-chat-v3-0324",
    # "deepseek/deepseek-v3.2-20251201",

    # # New Mistral flagship
    # "mistralai/mistral-large-2512",

    # # New Qwen3 large instruct
    # "qwen/qwen3-235b-a22b-2507",

    # # Popular non-US giant
    # "minimax/minimax-m2",

    # # non-mainstream additions
    # "moonshotai/kimi-k2",
    # "ai21/jamba-large-1.7",
    # "cohere/command-r-plus-08-2024",
    # "allenai/olmo-3-7b-instruct",
    # "deepcogito/cogito-v2-preview-llama-405b",
]


def extract_answer_text(result: Dict) -> str:
    """
    Extract raw textual answer from an OpenRouter chat completion result.

    Handles:
    - Standard models: message['content'] is a simple string like "3"
    - GPT-5-style: message['content'] == "" and answer lives in message['reasoning']
    - Possible list-based content: content is a list of {type: "text", text: "..."}
    - Possible future: answer in reasoning_details of type 'reasoning.summary'
    """
    choices = result.get("choices") or []
    if not choices:
        return ""

    msg = choices[0].get("message", {}) or {}

    # 1) Normal content as plain string
    content = msg.get("content")
    if isinstance(content, str) and content.strip():
        return content.strip()

    # 2) Content as list of parts (e.g. [{"type": "text", "text": "..."}])
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                parts.append(part["text"])
        if parts:
            return "\n".join(parts).strip()

    # 3) GPT-5: answer text is in `reasoning` string
    reasoning = msg.get("reasoning")
    if isinstance(reasoning, str) and reasoning.strip():
        return reasoning.strip()

    # 4) Fallback: summary in reasoning_details
    reasoning_details = msg.get("reasoning_details") or []
    for rd in reasoning_details:
        if rd.get("type") == "reasoning.summary":
            summary = rd.get("summary")
            if isinstance(summary, str) and summary.strip():
                return summary.strip()

    # Nothing usable
    return ""


def query_openrouter(
    model: str,
    persona: str,
    question: str,
    options: Dict[int, str],
    api_key: str,
    year: int,
    timeout: int = 30,
    max_retries: int = 3
) -> Dict:
    """
    Query OpenRouter API for a single question with retry logic.

    Returns:
        Dict with 'answer', 'error', token counts, and 'raw_response'
    """

    options_text = "\n".join([f"{k}. {v}" for k, v in options.items()])

    prompt = f"""It is now {year}. You are answering survey questions as the following person, who is living in the United States:

{persona}

Question: {question}

Options:
{options_text}

Respond with ONLY the number of your answer (e.g., "1" or "2"). Do not explain your reasoning."""

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    data = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }

    last_error = None

    for attempt in range(max_retries):
        try:
            response = requests.post(
                OPENROUTER_API_URL,
                headers=headers,
                json=data,
                timeout=timeout
            )
            response.raise_for_status()

            result = response.json()

            # Extract raw text using robust helper
            raw_text = extract_answer_text(result)
            raw_response = raw_text

            usage = result.get("usage", {}) or {}
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)

            if not raw_text:
                return {
                    "answer": None,
                    "error": "Empty response from model",
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "raw_response": raw_response,
                }

            clean = raw_text.strip()
            answer: Optional[int] = None

            # 1) Try JSON parsing: handles "[1.0, 1.0]", "1.0", "3", etc.
            try:
                parsed = json.loads(clean)
                candidate = None

                if isinstance(parsed, list) and parsed:
                    candidate = parsed[0]
                else:
                    candidate = parsed

                if isinstance(candidate, (int, float)):
                    answer = int(round(candidate))
                elif isinstance(candidate, str) and candidate.strip().isdigit():
                    answer = int(candidate.strip())
            except Exception:
                # Not JSON-like; fall through to regex
                pass

            # 2) Fallback: find first integer in the text
            if answer is None:
                m = re.search(r"\d+", clean)
                if m:
                    answer = int(m.group())

            # If we still don't have an answer, surface a parse error
            if answer is None:
                return {
                    "answer": None,
                    "error": f"Could not parse answer: {clean}",
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "raw_response": raw_response,
                }

            # Validate answer against options
            if answer not in options:
                return {
                    "answer": None,
                    "error": f"Invalid answer: {clean}",
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "raw_response": raw_response,
                }

            # Success
            return {
                "answer": answer,
                "error": None,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "raw_response": raw_response,
            }

        except requests.exceptions.Timeout:
            last_error = "Request timeout"
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
                continue

        except requests.exceptions.RequestException as e:
            last_error = str(e)
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue

        except Exception as e:
            last_error = f"Unexpected error: {str(e)}"
            break

    return {
        "answer": None,
        "error": last_error,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "raw_response": "",
    }


def load_completed_tasks(output_file: Path) -> Set[Tuple[int, str, int]]:
    """
    Load already-completed tasks from existing output file.

    Returns:
        Set of (persona_id, variable, run) tuples
    """
    if not output_file.exists():
        return set()

    try:
        df = pd.read_csv(output_file)
        completed = set(
            df[df['answer'].notna()][['persona_id', 'variable', 'run']]
            .apply(tuple, axis=1)
        )
        return completed
    except Exception as e:
        print(f"Warning: Could not load existing results: {e}")
        return set()


def main():
    parser = argparse.ArgumentParser(
        description="Query LLMs with comprehensive GSS attitudinal items"
    )
    parser.add_argument(
        "--year",
        type=int,
        required=True,
        choices=[2024, 2016, 2008, 2000],
        help="Survey year to simulate (2024, 2016, 2008, or 2000)"
    )
    parser.add_argument(
        "--models",
        type=str,
        help="Comma-separated list of model names"
    )
    parser.add_argument(
        "--all-models",
        action="store_true",
        help="Use all popular models"
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of runs per persona-question pair (default: 1)"
    )
    parser.add_argument(
        "--personas",
        type=int,
        default=600,
        help="Number of personas to query (default: 600)"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="Number of parallel workers (default: 8)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Number of results to accumulate before saving (default: 100)"
    )

    args = parser.parse_args()
    year = args.year

    # Get API key
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENROUTER_API_KEY environment variable not set. "
            "Get your API key from https://openrouter.ai/keys"
        )

    # Determine models to use
    if args.all_models:
        models = POPULAR_MODELS
    elif args.models:
        models = [m.strip() for m in args.models.split(",")]
    else:
        raise ValueError("Must specify either --models or --all-models")

    # Create output directory
    output_base = PROJECT_DIR / "synthetic_data"
    output_dir = output_base / f"year_{year}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"Comprehensive GSS Survey - Year {year}")
    print("=" * 70)
    print()
    print(f"Models: {len(models)}")
    for model in models:
        print(f"  - {model}")
    print(f"\nPersonas: {args.personas}")
    print(f"Questions: {len(GSS_QUESTIONS_COMPREHENSIVE)}")
    print(f"  - Culture-war items: {len(GSS_QUESTIONS_CULTUREWAR)}")
    print(f"  - Non-culture-war items: {len(GSS_QUESTIONS_NONCULTUREWAR)}")
    print(f"Runs per persona-question: {args.runs}")
    print(f"Max workers: {args.max_workers}")
    print(f"Batch size: {args.batch_size}")
    print(f"Output directory: {output_dir}")
    print()

    # Load personas - USE YEAR-SPECIFIC PERSONAS
    personas_file = PROJECT_DIR / "data" / f"gss{year}_personas.csv"
    if not personas_file.exists():
        # Fallback to 2024 if year-specific doesn't exist
        print(f"Warning: {personas_file} not found, falling back to 2024 personas")
        personas_file = PROJECT_DIR / "data" / "gss2024_personas.csv"

    if not personas_file.exists():
        raise FileNotFoundError(f"Personas file not found: {personas_file}")

    personas_df = pd.read_csv(personas_file)

    # Sample personas
    if args.personas and args.personas > 0 and len(personas_df) > args.personas:
        random.seed(42)
        personas_df = personas_df.sample(n=args.personas, random_state=42)

    print(f"Loaded {len(personas_df)} personas from {personas_file.name}")
    print()

    # Calculate total requests
    total_requests = len(models) * len(personas_df) * len(GSS_QUESTIONS_COMPREHENSIVE) * args.runs
    print(f"Total requests: {total_requests:,}")
    print()

    # Process each model
    for model in models:
        print("=" * 70)
        print(f"Processing: {model}")
        print("=" * 70)
        print()

        # Clean model name for filename
        model_filename = model.replace("/", "_")
        output_file = output_dir / f"{model_filename}.csv"

        # Load already-completed tasks for resume functionality
        completed_tasks = load_completed_tasks(output_file)

        if completed_tasks:
            print(f"Found {len(completed_tasks)} already-completed tasks")
            print("Resuming from previous run...")

        # Build task list (excluding already-completed)
        tasks = []
        for _, persona_row in personas_df.iterrows():
            persona_id = persona_row['respondent_id']
            persona_text = persona_row['persona']

            for var_name, question_data in GSS_QUESTIONS_COMPREHENSIVE.items():
                for run in range(1, args.runs + 1):
                    task_key = (persona_id, var_name, run)

                    # Skip if already completed
                    if task_key in completed_tasks:
                        continue

                    tasks.append({
                        'persona_id': persona_id,
                        'persona_text': persona_text,
                        'variable': var_name,
                        'question': question_data['text'],
                        'options': question_data['options'],
                        'run': run
                    })

        total_tasks = len(tasks) + len(completed_tasks)

        if not tasks:
            print(f"All tasks already completed for {model}")
            print(f"Output file: {output_file}\n")
            continue

        print(f"Tasks: {len(tasks)} remaining, {len(completed_tasks)} already done ({total_tasks} total)")
        print()

        results = []

        # Process tasks in parallel
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            future_to_task = {
                executor.submit(
                    query_openrouter,
                    model,
                    task['persona_text'],
                    task['question'],
                    task['options'],
                    api_key,
                    year
                ): task for task in tasks
            }

            pbar = tqdm(total=len(tasks), desc=model.split('/')[-1])

            for future in as_completed(future_to_task):
                task = future_to_task[future]
                result = future.result()

                results.append({
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'model': model,
                    'persona_id': task['persona_id'],
                    'variable': task['variable'],
                    'question_short': task['question'][:50] + '...',
                    'run': task['run'],
                    'answer': result.get('answer'),
                    'prompt_tokens': result.get('prompt_tokens', 0),
                    'completion_tokens': result.get('completion_tokens', 0),
                    'total_tokens': result.get('prompt_tokens', 0) + result.get('completion_tokens', 0),
                    'error': result.get('error', ''),
                    'raw_response': result.get('raw_response', '')
                })

                pbar.update(1)

                # Batch save
                if len(results) >= args.batch_size:
                    results_df = pd.DataFrame(results)
                    if not output_file.exists():
                        results_df.to_csv(output_file, index=False)
                    else:
                        results_df.to_csv(output_file, mode='a', header=False, index=False)
                    results = []

            pbar.close()

        # Save any remaining results
        if results:
            results_df = pd.DataFrame(results)
            if not output_file.exists():
                results_df.to_csv(output_file, index=False)
            else:
                results_df.to_csv(output_file, mode='a', header=False, index=False)

        # Read full results for summary
        results_df = pd.read_csv(output_file)

        # Print summary
        success_rate = (results_df['answer'].notna().sum() / len(results_df)) * 100
        total_tokens = results_df['total_tokens'].sum()

        print(f"\nResults saved: {output_file}")
        print(f"  Total responses: {len(results_df):,}")
        print(f"  Success rate: {success_rate:.1f}%")
        print(f"  Total tokens: {total_tokens:,}")
        print()

    print("=" * 70)
    print("All models complete!")
    print("=" * 70)


if __name__ == '__main__':
    main()
