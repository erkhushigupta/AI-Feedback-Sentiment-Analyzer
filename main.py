# app/flask_api.py - Simplified: No emotion analysis, derives overall sentiment

import os
from flask import Flask, request, jsonify
from pymongo import MongoClient, DESCENDING, errors as pymongo_errors
from datetime import datetime
from bson import ObjectId
from dotenv import load_dotenv
import analysis # Import updated analysis functions
from enum import Enum
import time

load_dotenv()

# --- Topics Enum ---
class FeedbackTopic(str, Enum):
    PRODUCT_DESIGN = "Product Design"; PRODUCT_PERFORMANCE = "Product Performance"
    SOFTWARE_APP = "Software/App"; CUSTOMER_SUPPORT = "Customer Support"
    WEBSITE_ORDERING = "Website/Ordering"; DOCUMENTATION = "Documentation"; OTHER = "Other"
VALID_TOPIC_VALUES = [topic.value for topic in FeedbackTopic]

# --- Flask App ---
app = Flask(__name__)

# --- Configuration & Global State ---
# MongoDB Atlas: MONGO_URI=mongodb+srv://<username>:<password>@<cluster-name>.mongodb.net/?retryWrites=true&w=majority
# MONGO_URI=mongodb+srv://khushigupta:!Password1@Cluster0.4ylqifz.mongodb.net'

MONGO_URI = os.getenv('MONGO_URI', 'mongodb+srv://khushigupta:!Password1@Cluster0.4ylqifz.mongodb.net/?retryWrites=true&w=majority')
DB_NAME = 'philips_feedback_db_flask_simple' # New DB name
COLLECTION_NAME = 'feedback_entries'
db_status = "startup_pending"
analysis_status = "startup_pending" # Renamed from models_loaded_status
aspect_model_available = False # Track specific model
mongo_client = None
collection = None

# --- Connect to MongoDB & Indexes ---
try:
    print(f"Connecting to MongoDB at {MONGO_URI}...")
    mongo_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    mongo_client.admin.command('ping')
    db = mongo_client[DB_NAME]; collection = db[COLLECTION_NAME]
    print(f"MongoDB Connected to DB: '{DB_NAME}'.")
    try:
        print("Ensuring indexes..."); collection.create_index([("timestamp", DESCENDING)], name="ts_idx")
        collection.create_index([("topic", 1)], name="topic_idx")
        print("Indexes ensured."); db_status = "connected"
    except Exception as e: print(f"ERROR ensuring indexes: {e}"); db_status = "index_error"
except Exception as e: print(f"CRITICAL ERROR connecting MongoDB: {e}"); db_status = "connection_failed"

# --- Load Aspect Model ---
# Called when module loads/app starts
# Note: In production, consider lazy loading or ensuring loading before first request if startup is slow
aspect_model_available = False # Initialize

# --- Helper: Serialize BSON types ---
def serialize_doc(doc):
    if isinstance(doc, list): return [serialize_doc(item) for item in doc]
    if isinstance(doc, dict): return {k: serialize_doc(v) for k, v in doc.items()}
    if isinstance(doc, ObjectId): return str(doc)
    if isinstance(doc, datetime): return doc.isoformat()
    return doc

# --- Helper: Calculate Overall Sentiment & Tone from Aspects ---
def derive_overall_sentiment(aspects: list):
    if not aspects: # No aspects found
        return 0, "Neutral", "Neutral"

    score = 0
    pos_count = 0
    neg_count = 0
    neu_count = 0

    for item in aspects:
        sentiment = item.get("sentiment", "Neutral")
        if sentiment == "Positive":
            score += 1
            pos_count += 1
        elif sentiment == "Negative":
            score -= 1
            neg_count += 1
        else:
            neu_count += 1

    total_aspects = len(aspects)

    # Determine Overall Label
    if score > 0: overall_label = "Positive"
    elif score < 0: overall_label = "Negative"
    else: overall_label = "Neutral"

    # Determine Derived Tone (Simplified)
    if score >= 2 or (pos_count / total_aspects > 0.6): # Strongly positive
        derived_tone = "Positive/Joyful"
    elif score <= -2 or (neg_count / total_aspects > 0.6): # Strongly negative
        derived_tone = "Negative/Frustrated"
    elif pos_count > 0 and neg_count > 0: # Mixed
         derived_tone = "Mixed"
    else: # Mostly neutral or balanced
        derived_tone = "Neutral"

    return score, overall_label, derived_tone


# --- API Endpoints ---
@app.route('/submit_feedback', methods=['POST'])
def submit_feedback():
    """Receives feedback, performs ASPECT analysis, derives overall sentiment, stores."""
    if not request.is_json: return jsonify({"message": "Request must be JSON"}), 400
    data = request.get_json()
    feedback_text = data.get('feedback_text'); topic_str = data.get('topic')

    # Validation
    if not feedback_text or not isinstance(feedback_text, str) or not feedback_text.strip():
        return jsonify({"message": "feedback_text required"}), 400
    if not topic_str or topic_str not in VALID_TOPIC_VALUES:
        return jsonify({"message": f"Valid topic required"}), 400

    print(f"Processing submission for topic '{topic_str}'...")

    # Check model status (only aspect model matters now)
    if not aspect_model_available:
         print("WARN: Submission attempt, but aspect model not loaded.")
         # Still process rules if VADER exists, but warn client
         # For simplicity, we can return an error if model essential
         return jsonify({"message": "Analysis service starting up or unavailable"}), 503

    # Perform Aspect Analysis ONLY
    try:
        aspects = analysis.analyze_aspects(feedback_text)
        print(f"Aspect analysis complete. Found {len(aspects)} aspects.")
    except Exception as e:
        print(f"CRITICAL ERROR during aspect analysis execution: {e}")
        return jsonify({"message": "Internal error during analysis"}), 500

    # Derive overall sentiment from aspects
    overall_score, overall_label, derived_tone = derive_overall_sentiment(aspects)
    print(f"Derived Overall: Score={overall_score}, Label={overall_label}, Tone={derived_tone}")

    feedback_entry = {
        "text": feedback_text,
        "topic": topic_str,
        "timestamp": datetime.utcnow(),
        "aspects": aspects, # Store the detailed aspects
        # Store derived overall metrics
        "overall_sentiment_score": overall_score,
        "overall_sentiment_label": overall_label,
        "derived_tone": derived_tone,
        "processed_successfully": True # Mark as processed
        # 'emotions' field REMOVED
    }

    # Store in DB
    if db_status != "connected" or collection is None:
        return jsonify({"message": "Database service unavailable"}), 503
    try:
        result = collection.insert_one(feedback_entry)
        insert_id = str(result.inserted_id)
        print(f"Feedback stored with ID: {insert_id}")
        return jsonify({"status": "success", "id": insert_id}), 201
    except pymongo_errors.PyMongoError as e:
        print(f"ERROR storing feedback: {e}")
        return jsonify({"message": "Database error storing feedback"}), 500

@app.route('/get_results', methods=['GET'])
def get_results():
    """Retrieves feedback including derived overall sentiment/tone."""
    if db_status != "connected" or collection is None: return jsonify({"message": "DB unavailable"}), 503

    topic_filter_str = request.args.get('topic'); limit_str = request.args.get('limit', default='1000')
    query_filter = {}
    if topic_filter_str:
        if topic_filter_str not in VALID_TOPIC_VALUES: return jsonify({"message": "Invalid topic filter"}), 400
        query_filter["topic"] = topic_filter_str
    try: limit = max(1, min(5000, int(limit_str))) # Bounded limit
    except ValueError: limit = 1000

    try:
        # Fetch new derived fields, remove 'emotions'
        projection = {
            "text": 1, "topic": 1, "timestamp": 1, "aspects": 1,
            "overall_sentiment_score": 1, "overall_sentiment_label": 1, "derived_tone": 1,
             "_id": 1 # Keep _id for Pydantic mapping in Streamlit if needed
        }
        feedback_data = list(
            collection.find(query_filter, projection=projection).sort("timestamp", DESCENDING).limit(limit)
        )
        serialized_data = serialize_doc(feedback_data)
        return jsonify(serialized_data)
    except pymongo_errors.PyMongoError as e:
        print(f"ERROR retrieving feedback: {e}")
        return jsonify({"message": "DB error retrieving feedback"}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Provides operational status (simplified)."""
    global db_status, analysis_status
    current_db_status = db_status
    try: # Quick ping check
        if mongo_client: mongo_client.admin.command('ping'); current_db_status = "connected"
        else: current_db_status = "connection_failed"
    except Exception: current_db_status = "connection_error_runtime"

    current_analysis_status = "loaded" if aspect_model_available else analysis_status # Use specific flag

    if current_db_status == "connected" and current_analysis_status == "loaded": api_status = "ok"
    elif current_db_status.startswith("connection") or current_analysis_status == "loading_failed": api_status = "error"
    else: api_status = "degraded"

    return jsonify({
        "api_status": api_status,
        "database_status": current_db_status,
        "analysis_models_status": current_analysis_status # Reflects only aspect model now
    })

# --- Main Execution ---
if __name__ == '__main__':
    print("Starting Flask application setup...")
    print("Loading Aspect model before starting Flask server...")
    aspect_model_available = analysis.load_models() # Load model and set flag
    analysis_status = "loaded" if aspect_model_available else "loading_failed"
    if not aspect_model_available: print("WARN: Aspect model failed to load.")

    print("Starting Flask server...")
    # debug=True helpful for development, runs startup twice but auto-reloads
    app.run(host='0.0.0.0', port=5000, debug=False)

