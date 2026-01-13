# SocialPulse - GenAI Social Media Analytics

**SocialPulse** is a real-time analytics dashboard that leverages Large Language Models (LLMs) to perform **Sentiment Analysis** and **Named Entity Recognition (NER)** on social media text. 

Built to simulate enterprise-level customer feedback analysis, this tool helps identify brand sentiment and key entities (People, Organizations, Locations) from unstructured text.

## 🚀 Features
* **Sentiment Analysis Pipeline:** Uses `distilbert-base-uncased` to classify text as Positive or Negative with confidence scores.
* **Named Entity Recognition (NER):** Uses BERT-based models to extract and categorize entities (e.g., "Elon Musk" -> Person, "Twitter" -> Organization).
* **Real-time Dashboard:** Interactive UI built with Streamlit for instant data visualization.
* **Optimized Performance:** Implements caching to handle heavy GenAI model loading efficiently.

## 🛠 Tech Stack
* **Language:** Python 3.10+
* **GenAI/ML:** Hugging Face Transformers, PyTorch
* **Frontend:** Streamlit
* **Data Processing:** Pandas, NumPy

## ⚙️ Installation & Run
1. Clone the repository:
   ```bash
   git clone [https://github.com/YOUR_USERNAME/SocialPulse.git](https://github.com/YOUR_USERNAME/SocialPulse.git)
