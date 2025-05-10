# AI-Feedback-Sentiment-Analyzer
 

A robust web-based sentiment analysis tool designed to process customer feedback, extract aspect-based sentiments, and derive overall sentiment labels and emotional tones using advanced NLP techniques.

## 🔧 Features

- Accepts customer feedback via a RESTful API
- Performs **Aspect-Based Sentiment Analysis (ABSA)** using `pyabsa`
- Derives overall sentiment and emotional tone (e.g., Joyful, Frustrated)
- Stores and indexes feedback in MongoDB Atlas
- Provides an endpoint to retrieve and filter analyzed feedback
- Built using Flask, MongoDB, HuggingFace Transformers, and Streamlit (optional UI layer)

## 📁 Project Structure

- `main.py`: Flask API for submitting and retrieving feedback
- `analysis.py`: Aspect extraction and sentiment analysis logic using `pyabsa`
- `app.py`: Streamlit frontend (optional)
- `requirements.txt`: All dependencies needed for this project
- `checkpoints*.json`: Configurations for fine-tuned or selected ABSA models

## 🚀 Getting Started

### 1. Clone the Repo
```bash
git clone https://github.com/<your-username>/feedback-sentiment-analyzer.git
cd feedback-sentiment-analyzer
