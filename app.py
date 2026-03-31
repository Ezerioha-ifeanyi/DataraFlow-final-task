import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import traceback
import textwrap
from textblob import TextBlob
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from openai import AzureOpenAI

# ==========================================
# 1. CUSTOM TOOLS (Data Science & NLP)
# ==========================================
def summarize_dataset(file_path: str) -> str:
    try:
        df = pd.read_csv(file_path)
        summary = {
            "total_reviews": len(df),
            "average_rating": round(df['reviews.rating'].mean(), 2) if 'reviews.rating' in df.columns else "N/A",
            "columns_found": list(df.columns)
        }
        return json.dumps(summary)
    except Exception as e:
        return json.dumps({"error": str(e)})

def perform_nlp_and_visualize(file_path: str) -> str:
    try:
        df = pd.read_csv(file_path)
        results = {"status": "success", "visualizations_saved": []}

        # 1. Rating Distribution Plot
        if 'reviews.rating' in df.columns:
            plt.figure(figsize=(6, 4))
            sns.countplot(data=df, x='reviews.rating', palette='viridis')
            plt.title('Distribution of Customer Ratings')
            plt.savefig('rating_distribution.png', bbox_inches='tight')
            plt.close()
            results["visualizations_saved"].append("rating_distribution.png")

        if 'reviews.text' in df.columns:
            text_data = df['reviews.text'].dropna().astype(str)
            
            # 2. Sentiment Analysis
            df['polarity'] = text_data.apply(lambda x: TextBlob(x).sentiment.polarity)
            results["average_sentiment_polarity"] = round(df['polarity'].mean(), 3)
            results["percent_negative_sentiment"] = round((df['polarity'] < 0).mean() * 100, 1)

            # 3. Word Cloud
            text_combined = ' '.join(text_data)
            wordcloud = WordCloud(width=600, height=300, background_color='white').generate(text_combined)
            plt.figure(figsize=(8, 4))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title('Common Keywords in Reviews')
            plt.savefig('word_cloud.png', bbox_inches='tight')
            plt.close()
            results["visualizations_saved"].append("word_cloud.png")

            # 4. Topic Modeling
            vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
            tfidf = vectorizer.fit_transform(text_data)
            lda = LatentDirichletAllocation(n_components=3, random_state=42)
            lda.fit(tfidf)
            
            topics = {}
            feature_names = vectorizer.get_feature_names_out()
            for topic_idx, topic in enumerate(lda.components_):
                topics[f"Theme_{topic_idx+1}"] = [feature_names[i] for i in topic.argsort()[:-6:-1]]
            results["extracted_topics"] = topics

        return json.dumps(results)
    except Exception as e:
        return json.dumps({"error": str(e), "traceback": traceback.format_exc()})

AVAILABLE_TOOLS = {
    "summarize_dataset": summarize_dataset,
    "perform_nlp_and_visualize": perform_nlp_and_visualize
}

tools_schema = [
    {
        "type": "function",
        "function": {
            "name": "summarize_dataset",
            "description": "Get basic stats from the dataset.",
            "parameters": {
                "type": "object",
                "properties": {"file_path": {"type": "string"}},
                "required": ["file_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "perform_nlp_and_visualize",
            "description": "Runs NLP algorithms to extract topics, calculates sentiment, and saves visualization images.",
            "parameters": {
                "type": "object",
                "properties": {"file_path": {"type": "string"}},
                "required": ["file_path"]
            }
        }
    }
]

# ==========================================
# 2. BASE AGENT CLASS (with Logging)
# ==========================================
class Agent:
    def __init__(self, name: str, role: str, system_prompt: str, client: AzureOpenAI, deployment: str, tools: list = None):
        self.name = name
        self.role = role
        self.system_prompt = system_prompt
        self.client = client
        self.deployment = deployment
        self.tools = tools or []
        self.max_iterations = 5
        self.logs = []

    def _log(self, msg: str):
        self.logs.append(msg)

    def run(self, user_message: str) -> str:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_message}
        ]
        
        self._log(f"[{self.name}] Starting task...")
        
        for iteration in range(self.max_iterations):
            try:
                response = self.client.chat.completions.create(
                    model=self.deployment,
                    messages=messages,
                    tools=self.tools if self.tools else None,
                    tool_choice="auto" if self.tools else None,
                    temperature=0.2
                )
                
                response_message = response.choices[0].message
                
                if response_message.tool_calls:
                    messages.append(response_message)
                    for tool_call in response_message.tool_calls:
                        function_name = tool_call.function.name
                        function_args = json.loads(tool_call.function.arguments)
                        
                        self._log(f"[{self.name}] Calling tool: {function_name}")
                        function_to_call = AVAILABLE_TOOLS.get(function_name)
                        
                        if function_to_call:
                            function_response = function_to_call(**function_args)
                            messages.append({
                                "tool_call_id": tool_call.id,
                                "role": "tool",
                                "name": function_name,
                                "content": str(function_response),
                            })
                    continue 
                else:
                    self._log(f"[{self.name}] Task complete.")
                    return response_message.content

            except Exception as e:
                error_msg = f"[{self.name}] Error: {str(e)}"
                self._log(error_msg)
                return error_msg
        
        self._log(f"[{self.name}] Reached max iterations.")
        return f"Agent '{self.name}' reached maximum iterations."

# ==========================================
# 3. ORCHESTRATOR CLASS
# ==========================================
class Orchestrator:
    def __init__(self, file_path: str, client: AzureOpenAI, deployment: str):
        self.file_path = file_path
        self.client = client
        self.deployment = deployment
        self.logs = []
        self.report = {}

    def _safe_parse(self, text: str, agent_name: str) -> dict:
        """Safely extracts JSON from LLM output, stripping markdown fences."""
        try:
            text = text.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            return json.loads(text)
        except json.JSONDecodeError:
            self.logs.append(f"[Orchestrator] WARNING: {agent_name} returned non-JSON.")
            return {}

    def run(self, progress_callback=None) -> dict:
        def _progress(step: int, total: int, label: str):
            self.logs.append(f"[Orchestrator] Step {step}/{total}: {label}")
            if progress_callback:
                progress_callback(step / total, label)

        # Agent 1: Data Loader
        _progress(1, 3, "Data Analyst is summarizing the dataset...")
        data_agent = Agent(
            "DataLoader", "Data Analyst",
            "You are a Data Analyst. Use tools to load the dataset and provide a brief statistical summary.",
            self.client, self.deployment, [tools_schema[0]]
        )
        data_summary = data_agent.run(f"Summarize the dataset at '{self.file_path}'.")
        self.logs.extend(data_agent.logs)

        # Agent 2: NLP Specialist
        _progress(2, 3, "NLP Specialist is running ML models and generating charts...")
        nlp_agent = Agent(
            "NLPSpecialist", "Data Scientist",
            "You are an NLP Specialist. Use the 'perform_nlp_and_visualize' tool to analyze sentiments and extract topics. Summarize the findings clearly.",
            self.client, self.deployment, [tools_schema[1]]
        )
        nlp_output = nlp_agent.run(f"Run NLP analysis for '{self.file_path}'. Context: {data_summary}")
        self.logs.extend(nlp_agent.logs)

        # Agent 3: Report Generator (Strict JSON schema)
        _progress(3, 3, "Product Manager is compiling the final JSON report...")
        report_agent = Agent(
            "ReportGenerator", "Product Manager",
            textwrap.dedent("""
            You are a Product Manager. Based on the data summary and NLP analysis provided, 
            generate a final strategic report. You MUST return ONLY a JSON object matching this exact schema:
            {
              "headline_stats": {"total_reviews": <int>, "average_rating": <float>, "negative_sentiment_pct": <float>},
              "extracted_themes": [{"theme_name": <str>, "keywords": [<str>, ...]}, ...],
              "business_interventions": [
                {"priority": "high"|"medium"|"low", "action": <str>, "rationale": <str>}
              ],
              "executive_summary": <2-sentence plain-English summary>
            }
            Do not include any text outside the JSON block.
            """),
            self.client, self.deployment, None
        )
        report_raw = report_agent.run(f"Data:\n{data_summary}\n\nNLP:\n{nlp_output}")
        self.logs.extend(report_agent.logs)
        
        self.report = self._safe_parse(report_raw, "ReportGenerator")
        return self.report

# ==========================================
# 4. STREAMLIT UI (Dashboard & Tabs)
# ==========================================
def main():
    st.set_page_config(page_title="Review Intelligence", layout="wide", page_icon="🛒")

    # --- Sidebar Configuration ---
    st.sidebar.title("⚙️ Azure Configuration")
    azure_endpoint = st.sidebar.text_input("Azure Endpoint", value=os.getenv("AZURE_OPENAI_ENDPOINT", ""))
    azure_key = st.sidebar.text_input("API Key", value=os.getenv("AZURE_OPENAI_API_KEY", ""), type="password")
    deployment = st.sidebar.text_input("Deployment Name", value=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4"))
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Pipeline Architecture:**\n1. DataLoader\n2. NLPSpecialist\n3. BusinessInsightAgent")

    # --- Main Page ---
    st.title("🛒 E-Commerce Review Intelligence")
    st.markdown("Multi-Agent ML Pipeline with NLP Topic Modeling & Sentiment Analysis")

    uploaded_file = st.file_uploader("Upload Amazon Reviews CSV", type=["csv"])

    if uploaded_file is not None:
        temp_path = "temp_dataset.csv"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        df_preview = pd.read_csv(temp_path)
        st.write(f"**Dataset Loaded:** {len(df_preview):,} rows")
        st.dataframe(df_preview.head(3), use_container_width=True)
        
        if st.button("🚀 Run Analysis Pipeline"):
            if not azure_key or not azure_endpoint:
                st.error("Please provide Azure credentials in the sidebar.")
                return

            client = AzureOpenAI(
                azure_endpoint=azure_endpoint,
                api_key=azure_key,
                api_version="2024-02-15-preview"
            )

            progress_bar = st.progress(0, text="Initializing...")
            
            def _update_progress(fraction, label):
                progress_bar.progress(fraction, text=label)

            orchestrator = Orchestrator(temp_path, client, deployment)
            report = orchestrator.run(progress_callback=_update_progress)
            
            progress_bar.progress(1.0, text="✅ Pipeline complete!")
            
            # --- Render Dashboard ---
            if report:
                st.markdown("---")
                
                # 1. Executive Summary & Metrics
                st.info(report.get("executive_summary", "Analysis complete."))
                
                stats = report.get("headline_stats", {})
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Reviews Analyzed", f"{stats.get('total_reviews', len(df_preview)):,}")
                col2.metric("Average Rating", f"{stats.get('average_rating', 'N/A')} / 5.0")
                col3.metric("Negative Sentiment", f"{stats.get('negative_sentiment_pct', 'N/A')}%")
                
                st.markdown("---")
                
                # 2. Tabbed Interface
                tab1, tab2, tab3, tab4 = st.tabs(["📊 Visualizations", "🎯 Action Plan", "📄 JSON Report", "🖥️ Agent Logs"])
                
                with tab1:
                    c1, c2 = st.columns(2)
                    if os.path.exists("rating_distribution.png"):
                        c1.image("rating_distribution.png", use_column_width=True)
                    if os.path.exists("word_cloud.png"):
                        c2.image("word_cloud.png", use_column_width=True)
                        
                    st.subheader("Extracted Themes (LDA Modeling)")
                    themes = report.get("extracted_themes", [])
                    for t in themes:
                        st.markdown(f"**{t.get('theme_name', 'Theme')}**: {', '.join(t.get('keywords', []))}")

                with tab2:
                    st.subheader("Recommended Business Interventions")
                    for item in report.get("business_interventions", []):
                        priority = item.get("priority", "medium").lower()
                        icon = {"high": "🔴", "medium": "🟡", "low": "🔵"}.get(priority, "⚪")
                        st.markdown(f"### {icon} {priority.upper()} PRIORITY")
                        st.markdown(f"**Action:** {item.get('action', '')}")
                        st.markdown(f"**Rationale:** {item.get('rationale', '')}")
                        st.markdown("---")

                with tab3:
                    st.json(report)

                with tab4:
                    for log in orchestrator.logs:
                        if "Error" in log or "WARNING" in log:
                            st.error(log)
                        elif "complete" in log:
                            st.success(log)
                        else:
                            st.text(log)

if __name__ == "__main__":
    main()