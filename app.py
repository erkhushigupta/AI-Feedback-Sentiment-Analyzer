# dashboard/streamlit_app.py - Simplified: Emotion components removed, shows derived sentiment/tone

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import time
from collections import Counter
import os
from dotenv import load_dotenv
import uuid
from typing import Optional

load_dotenv()

# --- Configuration ---
FLASK_API_URL_BASE = os.getenv('FLASK_API_URL', "http://127.0.0.1:5000") # Flask URL
FEEDBACK_ENDPOINT = f"{FLASK_API_URL_BASE}/submit_feedback"
RESULTS_ENDPOINT = f"{FLASK_API_URL_BASE}/get_results"
HEALTH_ENDPOINT = f"{FLASK_API_URL_BASE}/health"

TOPIC_LIST = [
    "Product Design", "Product Performance", "Software/App",
    "Customer Support", "Website/Ordering", "Documentation", "Other"
]
DEFAULT_TOPIC_OPTION = "All Topics"
TOP_N_ASPECTS = 5

# --- Page Setup ---
st.set_page_config(
    layout="wide",
    page_title="Philips Pulse | Simplified Feedback Analysis",
    page_icon="📊"
)

# --- API Health Check ---
@st.cache_data(ttl=30)
def check_api_health():
    # (Function remains the same)
    try:
        response = requests.get(HEALTH_ENDPOINT, timeout=5)
        if response.status_code == 200: return response.json()
        else:
            detail = f"API Error {response.status_code}"
            try: detail = response.json().get('message', detail)
            except Exception: pass
            return {"api_status": "error", "message": detail}
    except requests.exceptions.RequestException as e:
        return {"api_status": "connection_error", "message": f"Connection failed: {e}"}

# --- Display API Status ---
# (Code remains the same)
api_health = check_api_health()
overall_api_status = api_health.get("api_status", "unknown")
status_color = "green" if overall_api_status == "ok" else ("orange" if overall_api_status == "degraded" else "red")
status_text = overall_api_status.capitalize() if overall_api_status != "connection_error" else "Connection Error"
st.sidebar.markdown(f"**Backend API Status:** <span style='color:{status_color};'>● {status_text}</span>", unsafe_allow_html=True)
if overall_api_status != "ok":
    st.sidebar.caption(f"Details: {api_health.get('message', 'No details')}")
    # Updated to reflect only aspect model status
    st.sidebar.caption(f"DB: {api_health.get('database_status','N/A')} | Aspect Model: {api_health.get('analysis_models_status','N/A')}")

# --- Main Title ---
st.title("📊 Philips Pulse: Simplified Feedback Analysis")
st.markdown("Submit feedback by topic and view live aspect-based sentiment analysis.")

# --- Feedback Form ---
# (Code remains the same)
st.subheader("Submit New Feedback")
with st.form("feedback_form", clear_on_submit=True):
    selected_topic_submit = st.selectbox("Select Feedback Topic:", options=TOPIC_LIST, index=None, placeholder="Choose a topic...")
    feedback_text = st.text_area("Enter feedback:", height=100, key="feedback_text_input")
    submitted = st.form_submit_button("Analyze & Submit")
    if submitted:
        if selected_topic_submit and feedback_text and feedback_text.strip():
            try:
                with st.spinner(f"Submitting feedback..."):
                    payload = {"feedback_text": feedback_text, "topic": selected_topic_submit}
                    response = requests.post(FEEDBACK_ENDPOINT, json=payload, timeout=30)
                    response.raise_for_status()
                result = response.json()
                if result.get("status") == "success": st.success(f"Feedback submitted! (ID: {result.get('id')}).")
                else: st.error(f"Submission failed: {result.get('message', 'Unknown error')}")
            except Exception as e: st.error(f"Submission error: {e}") # Simplified error handling
        elif not selected_topic_submit: st.warning("Please select a topic.")
        else: st.warning("Please enter feedback text.")

# --- Dashboard Filters ---
# (Code remains the same)
st.sidebar.markdown("---"); st.sidebar.subheader("Dashboard Filters")
if 'selected_filter_topic' not in st.session_state: st.session_state['selected_filter_topic'] = DEFAULT_TOPIC_OPTION
selected_filter_topic = st.sidebar.selectbox(
    "Filter by Topic:", options=[DEFAULT_TOPIC_OPTION] + TOPIC_LIST, key='selected_filter_topic',
    index=([DEFAULT_TOPIC_OPTION] + TOPIC_LIST).index(st.session_state['selected_filter_topic'])
)

# --- Dashboard Placeholder ---
dashboard_placeholder = st.empty()

# --- Data Fetching Function ---
def fetch_data(filter_topic: Optional[str] = None) -> pd.DataFrame:
    """Fetches and prepares feedback data (aspects, derived sentiment/tone)."""
    api_params = {}
    request_url = RESULTS_ENDPOINT
    if filter_topic and filter_topic != DEFAULT_TOPIC_OPTION: api_params["topic"] = filter_topic
    try:
        response = requests.get(request_url, params=api_params, timeout=15)
        response.raise_for_status(); data = response.json()
        if isinstance(data, list):
            if not data: return pd.DataFrame()
            df = pd.DataFrame(data)
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce', utc=True)
            # Ensure 'aspects' exists, remove 'emotions' handling
            if 'aspects' not in df.columns: df['aspects'] = [[] for _ in range(len(df))]
            df['aspects'] = df['aspects'].apply(lambda x: x if isinstance(x, list) else [])
            # Ensure new derived fields exist (with defaults)
            if 'overall_sentiment_label' not in df.columns: df['overall_sentiment_label'] = "Neutral"
            if 'derived_tone' not in df.columns: df['derived_tone'] = "Neutral"
            if 'overall_sentiment_score' not in df.columns: df['overall_sentiment_score'] = 0

            if 'topic' not in df.columns: df['topic'] = "Unknown"
            # Map _id to entry_id if fetched (Flask version fetches it)
            if '_id' in df.columns: df.rename(columns={'_id': 'entry_id'}, inplace=True)
            return df
        else: st.error(f"API Error: Expected list, got {type(data)}."); return pd.DataFrame()
    except Exception as e: st.error(f"Data Fetch/Processing Error: {e}"); return pd.DataFrame()


# --- Efficient Aspect Aggregation Function ---
# (Function remains identical)
def get_top_aspects_summary(df, top_n=5):
    # ... (get_top_aspects_summary function is the same) ...
    all_aspects = []
    for aspects_list in df['aspects']: # Logic relies only on 'aspects' column
        if isinstance(aspects_list, list):
            all_aspects.extend([(item.get('aspect', '').strip(), item.get('sentiment', '').strip())
                                for item in aspects_list if isinstance(item, dict) and 'aspect' in item and 'sentiment' in item])
    all_aspects = [(aspect, sentiment) for aspect, sentiment in all_aspects if aspect]
    counts = Counter(all_aspects)
    if not counts: return pd.DataFrame(columns=['Aspect', '✅ Pos.', '➖ Neut.', '❌ Neg.', 'Total'])
    df_counts = pd.DataFrame([(a, s, c) for (a, s), c in counts.items()], columns=['Aspect', 'Sentiment', 'Count'])
    try: pivot = pd.pivot_table(df_counts, index='Aspect', columns='Sentiment', values='Count', fill_value=0)
    except Exception as e: st.error(f"Pivot Error: {e}"); return pd.DataFrame(...)
    s_map = {'Positive': '✅ Pos.', 'Neutral': '➖ Neut.', 'Negative': '❌ Neg.'}; cols = ['Aspect', *s_map.values(), 'Total']
    for orig, disp in s_map.items():
        if orig not in pivot.columns: pivot[orig] = 0
        pivot = pivot.rename(columns={orig: disp})
    cols_keep = [c for c in pivot.columns if c in s_map.values()]; pivot = pivot[cols_keep]
    pivot['Total'] = pivot.sum(axis=1)
    if 'Total' in pivot.columns: summary_df = pivot.nlargest(top_n, 'Total').reset_index()
    else: summary_df = pivot.reset_index().head(top_n)
    return summary_df[[c for c in cols if c in summary_df.columns]]


# --- Dashboard Update Loop ---
REFRESH_INTERVAL = 10
while True:
    df = fetch_data(filter_topic=st.session_state['selected_filter_topic'])
    update_id = str(uuid.uuid4()) # Unique ID for keys

    with dashboard_placeholder.container():
        filter_title = st.session_state['selected_filter_topic']
        st.subheader(f"Showing Analysis for Topic: {filter_title}")

        if not df.empty:
            update_time_str = pd.Timestamp.now(tz='UTC').strftime('%Y-%m-%d %H:%M:%S %Z')
            st.success(f"Dashboard updated: {update_time_str}")

            # --- Calculate Summaries ---
            top_aspects_summary_df = get_top_aspects_summary(df, top_n=TOP_N_ASPECTS)

            # --- Display KPIs (Updated) ---
            col_kpi1, col_kpi2, col_kpi3 = st.columns(3) # Use 3 columns for KPIs
            with col_kpi1:
                st.metric("Feedback Entries Shown", len(df))
            with col_kpi2:
                # Overall Sentiment Distribution KPI
                overall_sentiment_counts = df['overall_sentiment_label'].value_counts(normalize=True).mul(100)
                pos_perc = overall_sentiment_counts.get('Positive', 0)
                st.metric("Overall Sentiment", f"{pos_perc:.1f}% Positive")
            with col_kpi3:
                 # Derived Tone Distribution KPI
                derived_tone_counts = df['derived_tone'].value_counts()
                top_tone = derived_tone_counts.index[0] if not derived_tone_counts.empty else "N/A"
                st.metric("Predominant Tone", top_tone)


            # --- Display Top Aspects Table ---
            st.markdown("---")
            st.markdown(f"##### Top {TOP_N_ASPECTS} Mentioned Aspects Summary")
            if not top_aspects_summary_df.empty:
                st.dataframe(top_aspects_summary_df, hide_index=True, use_container_width=True)
            else:
                st.info("No aspects detected or summarized for this selection.")


            # --- Visualizations (Simplified) ---
            st.markdown("---")
            viz_col1, viz_col2 = st.columns(2)

            # --- VISUALIZATION COLUMN 1: Overall Sentiment/Tone & Trend ---
            with viz_col1:
                # --- Overall Sentiment Pie Chart ---
                st.markdown("###### Overall Sentiment Distribution")
                overall_sent_counts = df['overall_sentiment_label'].value_counts()
                if not overall_sent_counts.empty:
                     fig_overall_sent = px.pie(overall_sent_counts, values=overall_sent_counts.values, names=overall_sent_counts.index,
                                              title="Overall Feedback Sentiment", hole=0.3,
                                              color_discrete_map={'Positive':'mediumseagreen', 'Negative':'indianred', 'Neutral':'lightslategrey'})
                     fig_overall_sent.update_traces(textposition='inside', textinfo='percent+label')
                     st.plotly_chart(fig_overall_sent, use_container_width=True, key=f"overall_sentiment_pie_{update_id}") # Unique key
                else: st.info("No overall sentiment data.")

                # --- Feedback Volume Trend Chart ---
                st.markdown("###### Feedback Volume Trend")
                # (Code remains the same)
                df_time = df.dropna(subset=['timestamp']).copy()
                if not df_time.empty:
                    df_time = df_time.set_index('timestamp')
                    feedback_counts_time = df_time.resample('h').size()[lambda x: x > 0]
                    if not feedback_counts_time.empty:
                         fig_time = px.line(x=feedback_counts_time.index, y=feedback_counts_time.values, title="Feedback Volume (Hourly)", labels={'y': 'Entries', 'x': 'Time'})
                         fig_time.update_layout(xaxis_title="Time", yaxis_title="Number of Entries")
                         st.plotly_chart(fig_time, use_container_width=True, key=f"volume_trend_chart_{update_id}") # Unique key
                    else: st.info("No data for hourly trend.")
                else: st.info("No timestamp data for trend.")

            # --- VISUALIZATION COLUMN 2: Aspect Detail ---
            with viz_col2:
                # --- Overall Aspect Sentiment Bar Chart ---
                st.markdown("##### Aspect Sentiment Breakdown Chart")
                # (Code remains the same)
                all_aspects_data_viz = [item for sublist in df['aspects'] if isinstance(sublist, list) for item in sublist if isinstance(item, dict) and 'aspect' in item and 'sentiment' in item]
                if all_aspects_data_viz:
                    df_aspects_viz = pd.DataFrame(all_aspects_data_viz)
                    if not df_aspects_viz.empty:
                        aspect_counts_viz = df_aspects_viz.groupby(['aspect', 'sentiment']).size().reset_index(name='count')
                        if not aspect_counts_viz.empty:
                            fig_aspect = px.bar(aspect_counts_viz, x='aspect', y='count', color='sentiment', title="Sentiment Count per Detected Aspect", barmode='group', color_discrete_map={'Positive': 'mediumseagreen', 'Negative': 'indianred', 'Neutral': 'lightslategrey'}, category_orders={"sentiment": ["Positive", "Neutral", "Negative"]})
                            fig_aspect.update_layout(xaxis_title="Aspect", yaxis_title="Count", legend_title="Sentiment")
                            st.plotly_chart(fig_aspect, use_container_width=True, key=f"aspect_sentiment_chart_{update_id}") # Unique key
                        else: st.info("No aspect summary for bar chart.")
                    else: st.info("Aspect data format issue.")
                else: st.info("No aspects detected for bar chart.")

            # --- Recent Feedback Table (Updated Columns) ---
            st.markdown("---")
            st.subheader("Recent Feedback Entries")
            # Show derived fields instead of raw emotions
            display_columns = ['timestamp', 'topic', 'text', 'aspects', 'overall_sentiment_label', 'derived_tone']
            display_df = df.copy()[[col for col in display_columns if col in df.columns]].head(20)

            def format_aspects(aspects_list): # Function is the same
                 if not isinstance(aspects_list, list) or not aspects_list: return 'None'
                 return ', '.join([f"{item.get('aspect','?')} ({item.get('sentiment','?')})" for item in aspects_list if isinstance(item, dict)])
            display_df['aspects'] = display_df['aspects'].apply(format_aspects)

            display_df['timestamp'] = display_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S %Z')
            # Rename columns for display
            display_df = display_df.rename(columns={
                'timestamp':'Time (UTC)', 'topic':'Topic', 'text':'Feedback',
                'aspects':'Aspects (Sentiment)',
                'overall_sentiment_label': 'Overall Sent.', 'derived_tone': 'Derived Tone'
            })
            st.dataframe(display_df, use_container_width=True, height=500)

        else: # No data in df
            if overall_api_status != "ok": st.warning(f"Could not fetch data. API status: {status_text}.")
            elif st.session_state['selected_filter_topic'] == DEFAULT_TOPIC_OPTION: st.info("No feedback submitted yet.")
            else: st.info(f"No feedback found for topic: '{st.session_state['selected_filter_topic']}'.")

    # Pause before next refresh
    time.sleep(REFRESH_INTERVAL)
