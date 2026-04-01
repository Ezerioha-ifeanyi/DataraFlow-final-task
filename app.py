%%writefile app.py
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
# 1. CUSTOM TOOLS (Validation, Data Science & NLP)
# ==========================================
def validate_dataset(file_path: str) -> str:
    """Tool 1: Validates the health of the dataset (missing values, duplicates)."""
    try:
        df = pd.read_csv(file_path)
        required_cols = ['reviews.rating', 'reviews.text']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        health_report = {
            "total_rows": len(df),
            "duplicate_rows": int(df.duplicated().sum()),
            "missing_required_columns": missing_cols,
            "null_values": {col: int(df[col].isnull().sum()) for col in required_cols if col in df.columns},
            "status": "FAIL" if missing_cols else "PASS"
        }
        return json.dumps(health_report)
    except Exception as e:
        return json.dumps({"error": str(e)})

def summarize_dataset(file_path: str) -> str:
    """Tool 2: Reads the dataset and returns a statistical summary."""
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
    """Tool 3: Performs sentiment analysis, topic modeling, and saves visualizations."""
    try:
        df = pd.read_csv(file_path)
        results = {"status": "success", "visualizations_saved": []}

        if 'reviews.rating' in df.columns:
            plt.figure(figsize=(6, 4))
            sns.countplot(data=df, x='reviews.rating', hue='reviews.rating', legend=False, palette='viridis')
            plt.title('Distribution of Customer Ratings')
            plt.savefig('rating_distribution.png', bbox_inches='tight')
            plt.close()
            results["visualizations_saved"].append("rating_distribution.png")

        if 'reviews.text' in df.columns:
            text_data = df['reviews.text'].dropna().astype(str)
            
            df['polarity'] = text_data.apply(lambda x: TextBlob(x).sentiment.polarity)
            results["average_sentiment_polarity"] = round(df['polarity'].mean(), 3)
            results["percent_negative_sentiment"] = round((df['polarity'] < 0).mean() * 100, 1)

            text_combined = ' '.join(text_data)
            wordcloud = WordCloud(width=600, height=300, background_color='white').generate(text_combined)
            plt.figure(figsize=(8, 4))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title('Common Keywords in Reviews')
            plt.savefig('word_cloud.png', bbox_inches='tight')
            plt.close()
            results["visualizations_saved"].append("word_cloud.png")

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
    "validate_dataset": validate_dataset,
    "summarize_dataset": summarize_dataset,
    "perform_nlp_and_visualize": perform_nlp_and_visualize
}

tools_schema = [
    {
        "type": "function",
        "function": {
            "name": "validate_dataset",
            "description": "Checks the dataset for missing values, duplicates, and schema health.",
            "parameters": {"type": "object", "properties": {"file_path": {"type": "string"}}, "required": ["file_path"]}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "summarize_dataset",
            "description": "Get basic stats from the dataset.",
            "parameters": {"type": "object", "properties": {"file_path": {"type": "string"}}, "required": ["file_path"]}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "perform_nlp_and_visualize",
            "description": "Runs NLP algorithms to extract topics, calculates sentiment, and saves visualization images.",
            "parameters": {"type": "object", "properties": {"file_path": {"type": "string"}}, "required": ["file_path"]}
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
# 3. ORCHESTRATOR CLASS (The 5-Agent Pipeline)
# ==========================================
class Orchestrator:
    def __init__(self, file_path: str, client: AzureOpenAI, deployment: str):
        self.file_path = file_path
        self.client = client
        self.deployment = deployment
        self.logs = []
        self.report = {}
        self.raw_nlp_context = ""

    def _safe_parse(self, text: str, agent_name: str) -> dict:
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

        # Agent 1: Data Validator
        _progress(1, 5, "QA Engineer is validating dataset health...")
        validator_agent = Agent("DataValidator", "QA Engineer", "You are a Data Quality Engineer. Check the dataset for missing values and duplicates. Report the health status.", self.client, self.deployment, [tools_schema[0]])
        validation_status = validator_agent.run(f"Validate the dataset at '{self.file_path}'.")
        self.logs.extend(validator_agent.logs)

        # Agent 2: Data Analyst
        _progress(2, 5, "Data Analyst is summarizing the dataset...")
        data_agent = Agent("DataLoader", "Data Analyst", "You are a Data Analyst. Read the dataset and provide a brief statistical summary.", self.client, self.deployment, [tools_schema[1]])
        data_summary = data_agent.run(f"Summarize the dataset at '{self.file_path}'. Here is the validation context: {validation_status}")
        self.logs.extend(data_agent.logs)

        # Agent 3: NLP Specialist
        _progress(3, 5, "NLP Specialist is extracting themes and sentiments...")
        nlp_agent = Agent("NLPSpecialist", "Data Scientist", "You are an NLP Specialist. Use the 'perform_nlp_and_visualize' tool to analyze sentiments and extract topics.", self.client, self.deployment, [tools_schema[2]])
        nlp_output = nlp_agent.run(f"Run NLP analysis for '{self.file_path}'. Context: {data_summary}")
        self.logs.extend(nlp_agent.logs)
        self.raw_nlp_context = nlp_output 

        # Agent 4: Report Generator (Draft)
        _progress(4, 5, "Product Manager is drafting the initial report...")
        report_agent = Agent(
            "ReportGenerator", "Product Manager",
            textwrap.dedent("""
            You are a Product Manager. Based on the data summary and NLP analysis provided, 
            generate a draft strategic report. You MUST return ONLY a JSON object matching this exact schema:
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
        draft_report = report_agent.run(f"Validation:\n{validation_status}\n\nData:\n{data_summary}\n\nNLP:\n{nlp_output}")
        self.logs.extend(report_agent.logs)

        # Agent 5: Feedback & Review Agent (Final Output)
        _progress(5, 5, "Senior Strategy Director is reviewing and finalizing the report...")
        feedback_agent = Agent(
            "FeedbackAgent", "Senior Director",
            textwrap.dedent("""
            You are a Senior Strategy Director. Review the draft JSON report below.
            1. Ensure the interventions are directly supported by the NLP themes.
            2. Remove any 'hallucinations' (claims not supported by the data).
            3. Ensure the JSON schema is perfectly intact.
            Return ONLY the final, polished JSON object. No markdown fences.
            """),
            self.client, self.deployment, None
        )
        final_report_raw = feedback_agent.run(f"Original Context:\n{nlp_output}\n\nDraft Report:\n{draft_report}")
        self.logs.extend(feedback_agent.logs)
        
        self.report = self._safe_parse(final_report_raw, "FeedbackAgent")
        return self.report

# ==========================================
# 4. STREAMLIT UI & CHAT INTERFACE
# ==========================================
def main():
    st.set_page_config(page_title="Review Intelligence", layout="wide", page_icon="🛒")

    if "report" not in st.session_state: st.session_state.report = None
    if "nlp_context" not in st.session_state: st.session_state.nlp_context = ""
    if "chat_history" not in st.session_state: st.session_state.chat_history = []
    if "client" not in st.session_state: st.session_state.client = None
    if "deployment" not in st.session_state: st.session_state.deployment = "gpt-4o"
    if "logs" not in st.session_state: st.session_state.logs = []

    with st.sidebar:
        st.title("⚙️ Azure Configuration")
        azure_endpoint = st.text_input("Azure Endpoint", value=os.getenv("AZURE_OPENAI_ENDPOINT", ""))
        azure_key = st.text_input("API Key", value=os.getenv("AZURE_OPENAI_API_KEY", ""), type="password")
        deployment = st.text_input("Deployment Name", value=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o"))
        st.markdown("---")
        st.markdown("**Agents Deployed:**\n1. DataValidator (QA)\n2. DataLoader\n3. NLPSpecialist\n4. BusinessInsightAgent\n5. FeedbackAgent (Reviewer)")

    st.title("🛒 E-Commerce Review Intelligence")
    st.markdown("Multi-Agent ML Pipeline with Validation, Reflection, and NLP Topic Modeling.")

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
            
            st.session_state.client = client
            st.session_state.deployment = deployment

            progress_bar = st.progress(0, text="Initializing...")
            def _update_progress(fraction, label): progress_bar.progress(fraction, text=label)

            orchestrator = Orchestrator(temp_path, client, deployment)
            report = orchestrator.run(progress_callback=_update_progress)
            
            st.session_state.report = report
            st.session_state.nlp_context = orchestrator.raw_nlp_context
            st.session_state.logs = orchestrator.logs
            st.session_state.chat_history = [] 
            
            progress_bar.progress(1.0, text="✅ Pipeline complete!")

    if st.session_state.report:
        report = st.session_state.report
        st.markdown("---")
        st.info(report.get("executive_summary", "Analysis complete."))
        
        stats = report.get("headline_stats", {})
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Reviews Analyzed", f"{stats.get('total_reviews', 'N/A'):,}")
        col2.metric("Average Rating", f"{stats.get('average_rating', 'N/A')} / 5.0")
        col3.metric("Negative Sentiment", f"{stats.get('negative_sentiment_pct', 'N/A')}%")
        
        st.markdown("---")
        
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Visualizations", "🎯 Action Plan", "📄 JSON Report", "🖥️ Agent Logs"])
        
        with tab1:
            c1, c2 = st.columns(2)
            if os.path.exists("rating_distribution.png"): c1.image("rating_distribution.png", use_column_width=True)
            if os.path.exists("word_cloud.png"): c2.image("word_cloud.png", use_column_width=True)
            st.subheader("Extracted Themes (LDA Modeling)")
            for t in report.get("extracted_themes", []):
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

        with tab3: st.json(report)

        with tab4:
            for log in st.session_state.logs:
                if "Error" in log or "WARNING" in log: st.error(log)
                elif "complete" in log: st.success(log)
                else: st.text(log)
        
        st.markdown("---")
        st.subheader("💬 Ask the Insights Agent")
        st.markdown("Ask anything about the customer sentiments, themes, or recommended actions.")

        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        user_input = st.chat_input("e.g. Why is the negative sentiment percentage so high?")

        if user_input:
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.write(user_input)

            with st.chat_message("assistant"):
                with st.spinner("Analyzing context..."):
                    system_prompt = f"""You are a brilliant E-commerce Product Strategist. 
Use the following context from our multi-agent analysis to answer the user's questions.

NLP Analysis Context:
{st.session_state.nlp_context}

Final Structured Report:
{json.dumps(st.session_state.report, indent=2)}

Be concise, strategic, and practical. Reference the data directly."""

                    messages = [{"role": "system", "content": system_prompt}]
                    for msg in st.session_state.chat_history:
                        messages.append({"role": msg["role"], "content": msg["content"]})

                    try:
                        resp = st.session_state.client.chat.completions.create(
                            model=st.session_state.deployment,
                            messages=messages,
                            max_tokens=500,
                            temperature=0.4
                        )
                        answer = resp.choices[0].message.content
                    except Exception as e:
                        answer = f"Error connecting to AI: {e}"

                    st.write(answer)
                    st.session_state.chat_history.append({"role": "assistant", "content": answer})

    elif st.session_state.logs and not st.session_state.report:
        st.error("🚨 The pipeline finished, but the final report was empty. Please check the logs below to see where the API failed.")
        st.subheader("🖥️ Error Logs")
        for log in st.session_state.logs:
            if "Error" in log or "WARNING" in log:
                st.error(log)
            else:
                st.text(log)

if __name__ == "__main__":
    main()