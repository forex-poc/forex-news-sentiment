from datetime import datetime
import math
import re
from news_weights import KEYWORD_BOOST
from news_weights import CURRENCY_IMPACTS
from news_weights import SOURCE_WEIGHTS




def compute_time_weight(published_str):
    try:
        dt = datetime.fromisoformat(published_str.replace("Z", ""))
        delta_hours = (datetime.utcnow() - dt).total_seconds() / 3600
        return 1 / (1 + delta_hours)
    except Exception:
        return 0.5


def compute_source_weight(source):
    return SOURCE_WEIGHTS.get(source, 1.0)


def compute_keyword_boost(text):
    score = 1.0
    for word, boost in KEYWORD_BOOST.items():
        if re.search(rf"\b{re.escape(word)}\b", text, re.IGNORECASE):
            score *= boost
    return score


def normalize_currency(term):
    aliases = {
        "dollar": "USD",
        "euro": "EUR",
        "yen": "JPY",
        "pound": "GBP",
        "aussie": "AUD",
        "loonie": "CAD",
        "franc": "CHF",
        "kiwi": "NZD",
        "real": "BRL",
        "reais": "BRL",
        "brazil": "BRL",
        "brasil": "BRL",
    }
    upper = term.upper()
    return aliases.get(term.lower(), upper)


def detect_currencies(text):
    keywords = [
        "USD",
        "EUR",
        "JPY",
        "GBP",
        "AUD",
        "CAD",
        "CHF",
        "NZD",
        "BRL",
        "dollar",
        "euro",
        "yen",
        "pound",
        "aussie",
        "loonie",
        "franc",
        "kiwi",
        "real",
        "reais",
        "brazil",
        "brasil",
    ]
    currencies = set()

    for word in keywords:
        if word.lower() in text.lower():
            currencies.add(normalize_currency(word))

    for word, related in CURRENCY_IMPACTS.items():
        if word.lower() in text.lower():
            currencies.update(related)

    return list(currencies)
