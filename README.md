
# 🧠 AI-Powered News Analysis for Trading: A Self-Hosted Engine That Scores, Filters, and Recommends

In financial markets, news is fuel — but knowing which headlines actually matter is the difference between signal and noise. This article introduces a **self-hosted, private and free** system powered by artificial intelligence that collects, filters, and analyzes economic news to generate **daily trading signals for currency pairs**.

No paid APIs. No cloud. No data leaks.

---

## ⚙️ Purpose

The system was built to:

1. **Collect** hundreds of economic headlines from trusted sources  
2. **Deduplicate** semantically similar headlines using AI  
3. **Analyze sentiment** using FinBERT  
4. **Weight impact** by keywords and affected currency  
5. **Aggregate scores** for each currency  
6. **Generate trade recommendations** for currency pairs  

All this runs locally in Python 3.11.

---

## 📁 Project Structure

There are four main files:

| File | Role |
|------|------|
| `news_collector.py` | Fetches and saves daily economic news |
| `news_strength_engine.py` | Deduplicates, scores, classifies and generates recommendations |
| `news_weights.py` | Defines keywords and weights per currency |
| `utils.py` | Helper functions: normalization, vectorization, currency detection |

---

## 📥 Step 1: Collecting Economic News

Run:

```bash
$ python3.11 news_collector.py
```

The script scrapes over 20 trusted sources, including:

- **Media**: Reuters, Bloomberg, FXStreet, CNBC, Yahoo Finance  
- **Institutions**: Federal Reserve, IMF, World Bank, OECD  
- **Central banks**: ECB, BoE, RBA, BoC  
- **Reports**: DailyFX, ABC AU, Financial Post, CFTC  

Output example:

```
✅ 256 headlines saved at ../oraculum/src/app/data/news/2025-06-05.json
```

---

## 🧠 AI Engine #1: Semantic Deduplication

Identical news often appear on multiple outlets — but text comparison isn't enough.

We use `SentenceTransformer` to embed each headline and remove semantically duplicated entries:

```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")
vectors = model.encode(headlines)
```

Clustering + thresholding eliminates noise.

Real-world result:
```text
🧠 256 headlines loaded.  
🧹 243 unique after deduplication.
```

---

## 🧠 AI Engine #2: Sentiment Analysis with FinBERT

Each headline is scored using FinBERT, a BERT model trained on financial texts.

```python
def get_sentiment(text):
    tokens = tokenizer(text, return_tensors='pt')
    logits = model(**tokens).logits
    score = logits.softmax(dim=1)[0]
    return score[1] - score[0]  # positive - negative
```

Score ranges from -1 (bearish) to +1 (bullish). Neutral news gets close to 0.

---

## 🧠 AI Engine #3: Impact Weighting by Keywords

Not all news are equal. A “Fed rate hike” is more impactful than “consumer sentiment survey”.

In `news_weights.py`, we define keyword weights per currency:

```python
WEIGHTS = {
    "USD": {
        "interest": 0.9,
        "inflation": 0.85,
        "jobs": 0.8,
        ...
    },
    "EUR": {
        "ecb": 0.95,
        "gdp": 0.8,
        ...
    }
}
```

Each headline is tagged with a currency by keyword detection.  
The final **strength score** for the headline is:

```
sentiment × keyword_weight × confidence
```

---

## 📊 Final Output: Currency Pair Recommendations

Once all news are processed, we aggregate strength scores by currency:

```json
{
  "USD": -2.473,
  "EUR": -3.786,
  "JPY": -4.795,
  ...
}
```

Then generate pairwise trade signals:

```
EURUSD: SELL (-3.786 vs -2.473)
USDJPY: BUY  (-2.473 vs -4.795)
GBPCHF: SELL (-3.059 vs 1.22)
...
```

Buy = left currency is stronger  
Sell = left currency is weaker  
Neutral = similar scores

---

## 📤 Outputs Generated

Running the engine:

```bash
$ python3.11 news_strength_engine.py
```

Yields:

### 1. `news_strength.json` – full headline scoring

```json
{
  "headline": "Federal Reserve raises interest rates again",
  "currency": "USD",
  "sentiment": 0.84,
  "impact": 0.9,
  "strength": 0.756,
  "recommendation": "SELL"
}
```

### 2. `news_trade_recommendations.json` – trading signals

```json
{
  "EURUSD": "SELL",
  "USDJPY": "BUY",
  "GBPCHF": "SELL",
  ...
}
```

---

## 🔒 Why Self-Hosted?

- ✅ 100% offline  
- ✅ No API usage or call limits  
- ✅ Fully customizable  
- ✅ Ideal for sensitive or private data  
- ✅ Works on any Linux/macOS system

---

## 🧪 Use Cases

- 📈 Manual trading: daily bias per currency pair  
- 🤖 Bots: feed signal logic  
- 📊 Backtesting: check past headlines vs price action  
- 🗺️ Dashboards: generate heatmaps or trends  
- 📰 Internal curation: custom economic monitoring

---

## 🛣️ Roadmap (Next Steps)

- Multilingual translation (auto-detect and convert to English)  
- Live streaming support (real-time news via websockets)  
- React frontend with interactive heatmap  
- Historical backtest module  
- Optional LLM fine-tuning on proprietary datasets  

---

## 👨‍💻 About the Author

**Rafael Goulart Pedroso**  
Security researcher & AI developer  
📧 forex@codeartisan.cloud  
📱 [WhatsApp](https://wa.me/5511934251920)

---

If you believe AI + News = Better Trading Decisions, fork it, use it, or help us make it smarter.
