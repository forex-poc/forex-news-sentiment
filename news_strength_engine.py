import os
import json
from datetime import datetime, timedelta
from collections import defaultdict

from transformers import (
    pipeline,
    AutoModelForSequenceClassification,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from utils import (
    compute_time_weight,
    compute_keyword_boost,
    compute_source_weight,
    detect_currencies,
)

# === CONFIG ===
NEWS_DIR = "../oraculum/src/app/data/news"
OUTPUT_STRENGTH = "../oraculum/src/app/data/news/news_strength.json"
OUTPUT_RECOMMENDATIONS = "../oraculum/src/app/data/news/news_trade_recommendations.json"
DAYS = 3

# === MODELOS ===
MODEL_DIR = "models"
FINBERT_DIR = os.path.join(MODEL_DIR, "finbert")
BART_DIR = os.path.join(MODEL_DIR, "bart")
SBERT_DIR = os.path.join(MODEL_DIR, "sbert")
SBERT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
SBERT_PATH = os.path.join(SBERT_DIR, SBERT_MODEL_NAME.replace("/", "_"))

os.makedirs(FINBERT_DIR, exist_ok=True)
os.makedirs(BART_DIR, exist_ok=True)
os.makedirs(SBERT_DIR, exist_ok=True)
os.makedirs("data", exist_ok=True)

# === FINBERT ===
if not os.path.exists(os.path.join(FINBERT_DIR, "pytorch_model.bin")):
    print("⬇️ sellndo FinBERT...")
    finbert_model = AutoModelForSequenceClassification.from_pretrained(
        "ProsusAI/finbert"
    )
    finbert_tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    finbert_model.save_pretrained(FINBERT_DIR)
    finbert_tokenizer.save_pretrained(FINBERT_DIR)
else:
    finbert_model = AutoModelForSequenceClassification.from_pretrained(FINBERT_DIR)
    finbert_tokenizer = AutoTokenizer.from_pretrained(FINBERT_DIR)

classifier = pipeline(
    "sentiment-analysis", model=finbert_model, tokenizer=finbert_tokenizer
)

# === BART SUMMARIZER ===
if not os.path.exists(os.path.join(BART_DIR, "pytorch_model.bin")):
    print("⬇️ sellndo BART summarizer...")
    bart_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")
    bart_tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
    bart_model.save_pretrained(BART_DIR)
    bart_tokenizer.save_pretrained(BART_DIR)
else:
    bart_model = AutoModelForSeq2SeqLM.from_pretrained(BART_DIR)
    bart_tokenizer = AutoTokenizer.from_pretrained(BART_DIR)

summarizer = pipeline("summarization", model=bart_model, tokenizer=bart_tokenizer)

# === SBERT ===
if not os.path.exists(SBERT_PATH):
    print("⬇️ sellndo SBERT...")
    model = SentenceTransformer(SBERT_MODEL_NAME)
    model.save(SBERT_PATH)

embedder = SentenceTransformer(SBERT_PATH)

# === FUNÇÕES ===


def load_news():
    all_news = []
    for i in range(DAYS):
        day = (datetime.utcnow() - timedelta(days=i)).strftime("%Y-%m-%d")
        path = os.path.join(NEWS_DIR, f"{day}.json")
        if os.path.exists(path):
            with open(path) as f:
                all_news.extend(json.load(f))
    return all_news


def deduplicate(news):
    texts = [n["title"] + " " + n.get("summary", "") for n in news]
    embeddings = embedder.encode(texts)
    used = set()
    unique_news = []

    for i in range(len(news)):
        if i in used:
            continue
        for j in range(i + 1, len(news)):
            if j in used:
                continue
            sim = cosine_similarity([embeddings[i]], [embeddings[j]])[0][0]
            if sim > 0.87:
                used.add(j)
        unique_news.append(news[i])
    return unique_news


def summarize_if_needed(text, threshold=1500):
    if len(text) <= threshold:
        return text
    try:
        summary = summarizer(
            text, max_length=180, min_length=60, do_sample=False, truncation=True
        )[0]["summary_text"]
        print(f"🧠 Sumarizado: {text[:80]}... → {summary[:80]}")
        return summary
    except Exception as e:
        print(f"⚠️ Erro ao resumir: {e}")
        return text[:threshold]


def analyze(news):
    result = defaultdict(
        lambda: {
            "score": 0.0,
            "positive": 0,
            "negative": 0,
            "neutral": 0,
            "articles": [],
        }
    )

    for item in news:
        raw_text = item["title"] + " " + item.get("summary", "")
        clean_text = summarize_if_needed(raw_text)

        try:
            sent = classifier(clean_text)[0]
        except Exception as e:
            print(f"⚠️ Erro no modelo de sentimento: {e}")
            continue

        sentiment = sent["label"].lower()
        score_val = (
            1 if sentiment == "positive" else -1 if sentiment == "negative" else 0
        )

        weight = (
            compute_time_weight(item.get("published", ""))
            * compute_keyword_boost(clean_text)
            * compute_source_weight(item.get("source", ""))
        )

        currencies = detect_currencies(clean_text)
        for c in currencies:
            result[c]["score"] += score_val * weight
            result[c][sentiment] += 1
            result[c]["articles"].append(
                {
                    "title": item.get("title"),
                    "link": item.get("link"),
                    "sentiment": sentiment,
                    "published": item.get("published", ""),
                    "source": item.get("source", ""),
                }
            )

    for c in result:
        s = result[c]["score"]
        result[c]["direction"] = (
            "buy" if s > 0.75 else "sell" if s < -0.75 else "neutral"
        )
        result[c]["score"] = round(result[c]["score"], 3)

    return dict(sorted(result.items(), key=lambda x: -abs(x[1]["score"])))


def generate_trade_recommendations(scores):
    pairs = [
        "EURUSD",
        "GBPUSD",
        "AUDUSD",
        "NZDUSD",
        "USDJPY",
        "USDCAD",
        "USDCHF",
        "EURJPY",
        "EURGBP",
        "EURCHF",
        "EURAUD",
        "GBPJPY",
        "GBPCHF",
        "GBPAUD",
        "AUDJPY",
        "AUDCHF",
        "NZDJPY",
        "NZDCHF",
        "CADJPY",
        "CADCHF",
        "CHFJPY",
    ]

    recs = {}
    for pair in pairs:
        base = pair[:3]
        quote = pair[3:]
        base_score = scores.get(base, {}).get("score", 0)
        quote_score = scores.get(quote, {}).get("score", 0)
        diff = base_score - quote_score

        if abs(diff) < 0.5:
            direction = "neutral"
        elif diff > 0:
            direction = "BUY"
        else:
            direction = "SELL"

        recs[pair] = {
            "base": base,
            "quote": quote,
            "base_score": base_score,
            "quote_score": quote_score,
            "recommendation": direction,
        }

    return recs


def run():
    news = load_news()
    if not news:
        print("❌ Nenhuma notícia encontrada.")
        return

    print(f"🧠 {len(news)} notícias carregadas.")
    news = deduplicate(news)
    print(f"🧹 {len(news)} notícias únicas após deduplicação.")

    result = analyze(news)

    with open(OUTPUT_STRENGTH, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"✅ Resultado salvo em: {OUTPUT_STRENGTH}")

    recommendations = generate_trade_recommendations(result)
    with open(OUTPUT_RECOMMENDATIONS, "w") as f:
        json.dump(recommendations, f, indent=2, ensure_ascii=False)

    print(f"📈 Recomendações salvas em: {OUTPUT_RECOMMENDATIONS}")
    print("\n🏆 Recomendações por par:")
    for pair, data in recommendations.items():
        print(
            f"{pair}: {data['recommendation']} ({data['base_score']} vs {data['quote_score']})"
        )


if __name__ == "__main__":
    run()
