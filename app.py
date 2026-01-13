import streamlit as st
from transformers import pipeline
import pandas as pd
import numpy as np

# 1. Page Configuration (The "Frontend" Polish)
st.set_page_config(page_title="SocialPulse - GenAI Analytics", layout="wide")

st.title("SocialPulse: AI-Powered Social Media Analyzer")
st.markdown("""
**Objective:** Analyze social media text for Sentiment and Entities using LLM-based pipelines.
*Built for the JioStar Data Science PLI Application.*
""")

# 2. Loading Models (Caching prevents reloading on every click)
@st.cache_resource
def load_models():
    # Sentiment Analysis Pipeline (DistilBERT - faster/lighter LLM)
    sentiment_pipe = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    
    # NER (Named Entity Recognition) Pipeline (BERT - identifies Person, Org, Location)
    ner_pipe = pipeline("ner", grouped_entities=True)
    
    return sentiment_pipe, ner_pipe

# Load the AI models
with st.spinner("Loading GenAI Models..."):
    sentiment_pipeline, ner_pipeline = load_models()

# 3. Sidebar - Simulating "Data Source" selection
st.sidebar.header("Data Configuration")
source_type = st.sidebar.selectbox("Select Data Source", ["Direct Input", "Sample Tweet 1", "Sample Tweet 2"])

sample_texts = {
    "Sample Tweet 1": "I absolutely love the new features in JioCinema! The streaming quality is fantastic and the app is so smooth. Kudos to the team!",
    "Sample Tweet 2": "Really frustrated with the server downtime yesterday. I couldn't watch the match and the support team was unresponsive. #Angry",
}

# 4. Input Area
if source_type == "Direct Input":
    user_text = st.text_area("Enter Text / Tweet to Analyze:", height=100)
else:
    user_text = st.text_area("Enter Text / Tweet to Analyze:", value=sample_texts[source_type], height=100)

# 5. The Analysis Logic
if st.button("Analyze Content"):
    if user_text:
        # --- A. Sentiment Analysis ---
        result = sentiment_pipeline(user_text)[0]
        sentiment_label = result['label']
        confidence = result['score']

        # Display Metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Sentiment", sentiment_label, delta=f"{confidence:.2%}")
        
        # --- B. Named Entity Recognition (NER) ---
        # This addresses the JD requirement: "Analyze social media data... NER"
        st.subheader("Named Entity Recognition (NER)")
        ner_results = ner_pipeline(user_text)
        
        if ner_results:
            # Create a clean dataframe for the entities
            entities = []
            for entity in ner_results:
                entities.append({
                    "Entity Name": entity['word'],
                    "Category": entity['entity_group'],
                    "Confidence": f"{entity['score']:.2f}"
                })
            df_ner = pd.DataFrame(entities)
            st.table(df_ner)
        else:
            st.info("No specific entities (People, Orgs, Locations) detected.")

        # --- C. Simulated "Digital Analytics" Dashboard ---
        # This addresses the JD requirement: "metric tracking"
        st.markdown("---")
        st.subheader("Analytics Overview (Simulated)")
        
        # Fake historical data to show you know Data Visualization
        chart_data = pd.DataFrame(
            np.random.randn(20, 2),
            columns=['Positive Trend', 'Negative Trend']
        )
        st.line_chart(chart_data)

    else:
        st.warning("Please enter some text to analyze.")