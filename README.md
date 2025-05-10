# AI-Feedback-Sentiment-Analyzer
 

 

 

A full-stack sentiment analysis system designed to extract actionable insights from customer feedback. Using advanced NLP techniques, it identifies specific product aspects mentioned in reviews and analyzes their associated sentiment. Built with Flask, PyABSA, VADER, MongoDB, and Streamlit, it enables real-time submission, analysis, storage, and visualization of feedback for smart devices like Philips Smart Bulbs.

---

## 🚀 Features

- 🔍 **Aspect-Based Sentiment Analysis** (ABSA) using PyABSA (BERT-based)
- 💡 **Rule-based fallback** sentiment analysis using VADER
- 🗂️ Topic categorization for contextual understanding
- 🧾 Derives **overall sentiment** and **emotional tone**
- ☁️ Stores feedback in MongoDB Atlas
- 📈 Live visualization dashboard via Streamlit

---

## 🗂️ Project Structure

```

feedback-sentiment-analyzer/
├── main.py                 # Flask API backend
├── app.py                  # Streamlit frontend
├── analysis.py             # Core sentiment analysis logic (ABSA + VADER)
├── checkpoints.json        # PyABSA model checkpoint definitions
├── checkpoints-v1.16.json  # Alternate checkpoint configuration
├── requirements.txt        # Dependency list
└── .env                    # Environment variables

````

---

## ⚙️ Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/feedback-sentiment-analyzer.git
cd feedback-sentiment-analyzer
````

### 2. Create a `.env` File

```env
MONGO_URI=mongodb+srv://<username>:<password>@<cluster>.mongodb.net/?retryWrites=true&w=majority
FLASK_API_URL=http://127.0.0.1:5000
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Start the Flask Backend

```bash
python main.py
```

### 5. Launch the Streamlit Dashboard (Optional)

```bash
streamlit run app.py
```

---

## 🔌 API Overview

### `POST /submit_feedback`

Submit feedback for analysis.

```json
{
  "topic": "Product Design",
  "feedback_text": "The design looks sleek, but the app crashes often."
}
```

### `GET /get_results`

Fetch analyzed feedback with aspects, sentiments, and derived tones.

### `GET /health`

Returns backend and model status.

---

## 📈 Streamlit Dashboard

The optional frontend dashboard allows:

* Submitting feedback via form
* Filtering by topic
* Viewing aspect sentiment summaries
* Monitoring overall sentiment and trends

---

## 🧠 Technologies Used

* **Flask** – RESTful backend
* **Streamlit** – Interactive frontend
* **PyABSA** – Aspect-Based Sentiment Analysis (BERT-based)
* **VADER** – Rule-based sentiment fallback
* **MongoDB Atlas** – Cloud-hosted document storage
* **pandas**, **plotly**, **dotenv**, **requests**, **transformers** – Support libraries

---

## 🙌 Acknowledgements

* 🧠 [PyABSA](https://github.com/yangheng95/PyABSA) – ABSA models by Dr. Heng Yang
* 🤗 [Hugging Face Transformers](https://huggingface.co/) – NLP backbone models
* 🗣️ [VADER Sentiment](https://github.com/cjhutto/vaderSentiment) – Rule-based sentiment engine
* 🎨 [Streamlit](https://streamlit.io/) – Rapid UI for data apps
* ☁️ [MongoDB Atlas](https://www.mongodb.com/cloud/atlas) – Cloud database service

---

## ✍️ Author

Built by **Khushi Gupta** — Final-year Computer Science student passionate about AI, NLP, and human-centric tech innovation.

---

## 📄 License

This project is licensed under the MIT License.

 
