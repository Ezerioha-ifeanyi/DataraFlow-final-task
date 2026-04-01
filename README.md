# 🛒 E-Commerce Review Intelligence System
### A 5-Agent Generative AI Pipeline for Amazon Product Review Analysis

> *Final Internship Project — DataraFlow | Ezerioha Ifeanyi Emmanuel*

---
[366b37c1-f88e-41ea-a9ce-c0f37aad0b6b.webm](https://github.com/user-attachments/assets/7507542a-8057-4f12-91f8-14d5d41c925b)

## 📌 Overview

This project is a **Multi-Agent AI system** that transforms raw Amazon product review data into verified, actionable business intelligence — automatically. It combines a **Sequential Pipeline** with a **Reflection Pattern** to ensure every output is grounded in data, not AI guesswork.

Five specialised GPT-4o agents work in sequence: one validates the data, one analyses it statistically, one runs machine learning models, one drafts a business report, and a final critic agent reviews and corrects that draft before it ever reaches the user.

The system is deployed as an interactive **Streamlit web application**, publicly accessible via LocalTunnel directly from Google Colab.

---

## 🏗️ System Architecture

```
[CSV Dataset]
     │
     ▼
┌──────────────────────────────────────┐
│  Agent 1: DataValidator  (QA)        │  ← validate_dataset() tool
│  Checks: nulls, duplicates, schema   │
└──────────────────────────────────────┘
     │  Health report (PASS/FAIL + null counts)
     ▼
┌──────────────────────────────────────┐
│  Agent 2: DataLoader  (Analyst)      │  ← summarize_dataset() tool
│  Produces: row counts, avg rating    │
└──────────────────────────────────────┘
     │  Statistical summary
     ▼
┌──────────────────────────────────────┐
│  Agent 3: NLPSpecialist  (Scientist) │  ← perform_nlp_and_visualize() tool
│  Produces: sentiment, topics, charts │
└──────────────────────────────────────┘
     │  NLP findings
     ▼
┌──────────────────────────────────────┐
│  Agent 4: ReportGenerator  (PM)      │  ← No tools; synthesises into JSON
│  Produces: draft business report     │
└──────────────────────────────────────┘
     │  Draft JSON report
     ▼
┌──────────────────────────────────────┐
│  Agent 5: FeedbackAgent  (Director)  │  ← No tools; validates + corrects
│  Removes hallucinations, fixes schema│
└──────────────────────────────────────┘
     │
     ▼
[Final Verified JSON Business Report]
```

---

## ✨ Features

- **5-Agent Sequential Pipeline** with Reflection Pattern
- **3 Custom Tools** via OpenAI Function Calling API (validation, statistics, NLP)
- **Sentiment Analysis** using TextBlob (polarity scoring per review)
- **Topic Modelling** using LDA (Latent Dirichlet Allocation) via Scikit-learn
- **Data Visualisations** — rating distribution bar chart + keyword word cloud
- **Hallucination Guard** — Feedback Agent cross-checks report against raw NLP data
- **Data Quality Gate** — Validator flags nulls and duplicates before any analysis runs
- **Structured JSON Output** — schema-enforced business report with headline stats, themes, and prioritised interventions
- **Interactive Chat Interface** — post-analysis Q&A powered by GPT-4o with full report context
- **Streamlit Web App** — deployed live via LocalTunnel from Google Colab

---

## 🤖 Agent Roster

| # | Agent Name | Role | Tools | Responsibility |
|---|---|---|---|---|
| 1 | `DataValidator` | QA Engineer | `validate_dataset` | Checks for nulls, duplicates, and required columns |
| 2 | `DataLoader` | Data Analyst | `summarize_dataset` | Computes row counts, average rating, column inventory |
| 3 | `NLPSpecialist` | Data Scientist | `perform_nlp_and_visualize` | Sentiment analysis, LDA topic modelling, chart generation |
| 4 | `ReportGenerator` | Product Manager | None | Synthesises all findings into a draft JSON business report |
| 5 | `FeedbackAgent` | Senior Director | None | Reviews draft, removes hallucinations, enforces JSON schema |

---

## 🛠️ Custom Tools (Function Calling)

### `validate_dataset(file_path)`
Checks dataset health before any analysis runs. Reports total rows, duplicate count, missing required columns, and null value counts per critical column. Returns `"status": "PASS"` or `"FAIL"`.

### `summarize_dataset(file_path)`
Returns total review count, average star rating, and full column list as JSON. Used by the Data Analyst agent.

### `perform_nlp_and_visualize(file_path)`
The core ML tool. Runs three operations:
- **TextBlob sentiment analysis** — polarity score per review, aggregated to average and negative %
- **LDA topic modelling** — TF-IDF vectorisation (1,000 features) + 3-component LDA to extract product themes
- **Visualisations** — saves `rating_distribution.png` and `word_cloud.png` to disk

---

## 📊 Sample Output

```json
{
  "headline_stats": {
    "total_reviews": 1597,
    "average_rating": 4.36,
    "columns_found": 26,
    "null_values": { "reviews.rating": 420, "reviews.text": 0 },
    "average_sentiment_polarity": 0.278,
    "percent_negative_sentiment": 4.8
  },
  "extracted_themes": [
    { "theme": "Headphones and Sound Quality", "keywords": ["headphones", "sound", "like", "don", "ears"] },
    { "theme": "Kindle and Amazon Devices",    "keywords": ["kindle", "amazon", "tv", "prime", "screen"] },
    { "theme": "General Product Satisfaction", "keywords": ["great", "product", "quality", "price", "buy"] }
  ],
  "business_interventions": [
    { "priority": "high",   "action": "Address audio comfort complaints", "rationale": "Ear-related keywords dominate Theme 1 despite high overall ratings" },
    { "priority": "medium", "action": "Highlight Kindle battery and screen in marketing", "rationale": "These features are frequently evaluated by reviewers" },
    { "priority": "low",    "action": "Maintain current pricing strategy", "rationale": "Price-to-quality ratio is positively received" }
  ],
  "executive_summary": "Customer sentiment is overwhelmingly positive with a 4.36 average rating and only 4.8% negative reviews. Key strategic priority is addressing headphone ergonomics while reinforcing Kindle screen and battery messaging."
}
```

---

## 🚀 Getting Started

### Prerequisites
```bash
pip install openai pandas matplotlib seaborn textblob wordcloud scikit-learn streamlit python-dotenv
```

### Running the Notebook
1. Open `Ezerioha_Ifeanyi_Final_Task.ipynb` in Google Colab
2. Mount your Google Drive and place `amazon_reviews.csv` at `/content/drive/MyDrive/`
3. Add your Azure OpenAI credentials in Cell 5:
   ```python
   AZURE_ENDPOINT = "your-endpoint"
   API_KEY = "your-api-key"
   DEPLOYMENT_NAME = "gpt-4o"
   ```
4. Run all cells — the 5-agent pipeline executes automatically

### Running the Streamlit App (LocalTunnel Deployment)
The notebook's final cells handle deployment automatically:
```python
# Cell 13: Writes app.py to disk via %%writefile
# Cell 15: Launches Streamlit + LocalTunnel
!npm install localtunnel
!streamlit run app.py & npx localtunnel --port 8501
```
A public `loca.lt` URL is generated. Use the printed IP address as the LocalTunnel password to access the live dashboard.

### Running Locally
```bash
# Set environment variables
export AZURE_OPENAI_ENDPOINT="your-endpoint"
export AZURE_OPENAI_API_KEY="your-api-key"
export AZURE_OPENAI_DEPLOYMENT_NAME="gpt-4o"

# Launch the app
streamlit run app.py
```

---

## 📁 Project Structure

```
├── Ezerioha_Ifeanyi_Final_Task.ipynb   # Main notebook — pipeline + deployment
├── app.py                               # Streamlit app (auto-generated by notebook)
├── rating_distribution.png             # Generated visualisation
├── word_cloud.png                       # Generated visualisation
└── README.md
```

---

## 🧱 Tech Stack

| Category | Technologies |
|---|---|
| **LLM** | Azure OpenAI GPT-4o |
| **Agent Framework** | Custom `Agent` class + OpenAI Function Calling |
| **NLP / ML** | TextBlob, Scikit-learn (TF-IDF + LDA), WordCloud |
| **Data** | Pandas, Seaborn, Matplotlib |
| **App** | Streamlit |
| **Deployment** | Google Colab + LocalTunnel |
| **Language** | Python 3.12 |

---

## 🧠 Key Design Decisions

**Why 5 agents instead of 1 prompt?**
Each agent is scoped to a single responsibility, making failures easy to isolate. A monolithic prompt would mix data validation, NLP, and business logic into one uncontrollable output.

**Why a Feedback Agent?**
LLMs can synthesise claims that sound plausible but aren't supported by the actual data. The Feedback Agent holds the Report Generator accountable by cross-referencing its output against the raw NLP findings — a pattern called **Reflection**.

**Why deterministic tools for numerical outputs?**
LLMs are not calculators. Sentiment scores, null counts, and topic keywords are computed by Python libraries and passed to the LLM as verified facts. The LLM interprets; it never calculates.

---

## 📈 Results

Running the full pipeline on 1,597 Amazon reviews produced:
- Data quality gate: **420 null rating values** detected (26% of dataset) before analysis
- Average sentiment polarity: **0.278** (mildly positive)
- Negative sentiment rate: **4.8%**
- **3 product conversation themes** discovered via LDA
- **3 prioritised business interventions** verified against NLP findings
- Live Streamlit dashboard deployed in under 3 minutes from a Colab notebook

---

## 🔮 Future Improvements

- Replace LDA with **BERTopic** for transformer-based topic extraction
- Add an **Imputation Agent** to handle missing values before analysis
- Deploy on **Azure Functions** with persistent per-user report storage
- Implement a **re-run loop**: if the Feedback Agent flags unresolvable hallucinations, trigger Agent 4 to regenerate
- Extend to **multi-product comparison** across different categories

---

## 📄 Publication

Full technical write-up available here: *[https://godwithus.hashnode.dev/i-built-a-multi-agent-ai-system-that-analyses-amazon-product-reviews]*

---

## 👤 Author

**Ezerioha Ifeanyi Emmanuel**
DataraFlow Intern | Data Science & AI Engineering

---

## 📜 License

This project was built as part of an internship assessment. Feel free to fork, adapt, and build on it.
