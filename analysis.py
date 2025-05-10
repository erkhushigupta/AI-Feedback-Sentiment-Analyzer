# app/analysis.py - FINAL SIMPLIFIED VERSION (NO EMOTION MODEL)

import os
import time
import re # Import regular expressions for clause splitting

# --- PyABSA v1.x Imports ---
# Use the Checkpoint Manager, suitable for PyABSA v1.x
from pyabsa import ATEPCCheckpointManager

# --- VADER for rule-based sentiment ---
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    vader_analyzer = SentimentIntensityAnalyzer()
    print("VADER Sentiment Analyzer loaded successfully.")
    def get_vader_sentiment_label(text):
        if not text or not isinstance(text, str): return 'Neutral'
        vs = vader_analyzer.polarity_scores(text)
        if vs['compound'] >= 0.05: return 'Positive'
        elif vs['compound'] <= -0.05: return 'Negative'
        else: return 'Neutral'
except ImportError:
    print("WARN: vaderSentiment not installed (pip install vaderSentiment). Rule-based sentiment fallback to 'Neutral'.")
    vader_analyzer = None
    def get_vader_sentiment_label(text): return 'Neutral' # Fallback function

# --- Core Keywords for Rule-Based Aspect Detection ---
CORE_KEYWORDS = [
    "battery", "power", "charge", "screen", "display", "clarity", "brightness",
    "performance", "speed", "efficiency", "lag", "responsiveness", "design",
    "look", "feel", "build", "material", "quality", "software", "app",
    "interface", "update", "os", "bug", "support", "service",
    "customer service", "help", "warranty", "price", "cost", "value",
    "expensive", "cheap", "camera", "photo", "picture", "video", "sound",
    "audio", "speaker", "microphone", "connectivity", "wifi", "bluetooth", "network",
    # Add more Philips-specific terms if needed
]
CORE_KEYWORDS_LOWER = {kw.lower() for kw in CORE_KEYWORDS}

# --- Global Variables for Models ---
# ONLY Aspect Extractor needed
aspect_extractor = None # PyABSA model instance
models_loaded = False # Tracks if loading function has run
model_available = False # Tracks if the aspect model successfully loaded

# --- Model Loading Function ---
def load_models():
    """Loads ONLY the PyABSA Aspect Extractor model."""
    global aspect_extractor, models_loaded, model_available
    # Avoid reloading if already attempted
    if models_loaded:
        print("Model loading function already executed.")
        return model_available

    print("Initiating model loading sequence (Aspect Extractor only)...")
    start_time = time.time()
    loading_success = False # Assume failure initially

    # Load PyABSA v1.x Aspect Extractor Model
    try:
        print("Loading PyABSA v1.x Aspect Extractor model...")
        # Using multilingual checkpoint as potentially having broader coverage
        checkpoint_name = 'multilingual'
        print(f"Attempting checkpoint: '{checkpoint_name}' via CheckpointManager")
        # Initialize using the correct method for PyABSA v1.x
        aspect_extractor = ATEPCCheckpointManager.get_aspect_extractor(
            checkpoint=checkpoint_name,
            auto_device=True, # Automatically use GPU if available, else CPU
        )
        print("PyABSA Aspect Extractor loaded successfully.")
        loading_success = True # Mark as successful
    except ImportError as imp_e:
         print(f"ERROR: Failed to import required components for PyABSA: {imp_e}")
         print("Ensure PyABSA v1.x and its dependencies are correctly installed.")
         aspect_extractor = None # Ensure model is None if import fails
    except Exception as e:
        # Catch other potential errors during loading
        print(f"ERROR loading PyABSA model with checkpoint '{checkpoint_name}': {e}")
        print("Check PyABSA installation and checkpoint validity for v1.x.")
        aspect_extractor = None # Ensure model is None if loading fails

    # Finalize loading status
    end_time = time.time()
    print(f"Model loading sequence finished in {end_time - start_time:.2f} seconds.")
    models_loaded = True # Mark that the loading function has run
    model_available = loading_success # Set availability based on actual success
    print(f"Aspect Extractor model available: {model_available}")
    return model_available # Return the availability status

# --- Analysis Functions ---

# --- ENHANCED Aspect Analysis Function ---
def analyze_aspects(text: str) -> list:
    """
    Performs Aspect-Based Sentiment Analysis using PyABSA (if available)
    AND rule-based keyword spotting with clause-level sentiment.
    Combines results, removes duplicates, uses clause-level sentiment for rules.
    Returns list: [{'aspect': '...', 'sentiment': '...'}].
    """
    # Input validation
    if not text or not isinstance(text, str):
        print("WARN: analyze_aspects received invalid input.")
        return []

    # Use a dictionary to store results, keyed by lowercase aspect to handle duplicates
    final_aspects: dict = {}

    # --- Step 1: Run PyABSA Model (if available) ---
    if aspect_extractor is not None:
        try:
            # print(f"DEBUG: Running PyABSA on: '{text[:50]}...'") # Optional
            results = aspect_extractor.extract_aspect(
                inference_source=[text],
                pred_sentiment=True,      # Get sentiment from model
                save_result=False,
                print_result=False,
                ignore_error=True       # Try to prevent PyABSA internal errors stopping flow
            )
            # print(f"DEBUG: PyABSA result: {results}") # Optional

            # Process valid PyABSA results
            if results and isinstance(results, list) and len(results) > 0:
                data = results[0]
                aspects = data.get('aspect', [])
                sentiments = data.get('sentiment', [])
                if isinstance(aspects, list) and isinstance(sentiments, list) and len(aspects) == len(sentiments):
                     for asp, sent in zip(aspects, sentiments):
                        if isinstance(asp, str) and isinstance(sent, str):
                            aspect_term = asp.strip().lower()
                            # Store if aspect is valid and not already found (PyABSA takes priority)
                            if aspect_term and aspect_term not in final_aspects:
                                final_aspects[aspect_term] = {"aspect": asp.strip(), "sentiment": sent.strip()}
                        else:
                            print(f"WARN: PyABSA returned non-string: Aspect={type(asp)}, Sent={type(sent)}")
                # Log mismatch only if one list exists but lengths differ or types invalid
                elif (isinstance(aspects, list) and aspects) or (isinstance(sentiments, list) and sentiments):
                     print(f"WARN: PyABSA returned mismatched lists: A={aspects}, S={sentiments}")

        except Exception as e:
            print(f"ERROR during PyABSA aspect analysis execution: {e}")
            # Proceed to rule-based analysis even if PyABSA fails

    # --- Step 2: Rule-Based Keyword Spotting with Clause-Level Sentiment ---
    try:
        text_lower = text.lower()
        # Split text by contrastive conjunctions to isolate sentiment context
        clauses = re.split(r'\b(?:but|however|though|yet|although|while)\b', text, flags=re.IGNORECASE)
        # Remove empty strings resulting from split
        clauses = [clause.strip() for clause in clauses if clause.strip()]
        # If no conjunctions found, use the full text as the only clause
        if not clauses: clauses = [text]

        # Map keywords to the first clause they appear in for context
        aspect_to_clause = {}
        for clause in clauses:
            clause_lower = clause.lower()
            for keyword_lower in CORE_KEYWORDS_LOWER:
                if keyword_lower in clause_lower:
                    # Store the first clause encountered for this keyword
                    if keyword_lower not in aspect_to_clause:
                         aspect_to_clause[keyword_lower] = clause

        found_keywords_rule = set() # Keep track for debugging

        # Iterate through predefined core keywords
        for keyword_lower in CORE_KEYWORDS_LOWER:
            # Check if keyword is in text and NOT already found by the PyABSA model
            if keyword_lower not in final_aspects and keyword_lower in text_lower:
                # Use the specific clause associated with the keyword, or default to the first/only clause
                relevant_clause = aspect_to_clause.get(keyword_lower, clauses[0])

                # Get sentiment for the relevant clause using VADER
                rule_sentiment = get_vader_sentiment_label(relevant_clause) # Uses fallback if VADER failed to load

                # Attempt to find the original casing of the keyword in the text
                try:
                    start_index = text_lower.index(keyword_lower)
                    original_casing_keyword = text[start_index : start_index + len(keyword_lower)]
                except ValueError:
                    # Fallback if index finding fails (unlikely but safe)
                    original_casing_keyword = keyword_lower.capitalize()

                # Add the rule-based result to the final dictionary
                final_aspects[keyword_lower] = {"aspect": original_casing_keyword, "sentiment": rule_sentiment}
                found_keywords_rule.add(keyword_lower)

        # Optional Debugging log
        # if found_keywords_rule:
        #     print(f"DEBUG: Rule-based logic added aspects: {found_keywords_rule}")

    except Exception as e:
        # Catch errors specifically in the rule-based processing
        print(f"ERROR during rule-based aspect analysis: {e}")
        # Allow function to return aspects found before the error

    # --- Step 3: Convert consolidated dictionary values back to a list ---
    # This should now be correctly indented within the function
    try:
        result_list = list(final_aspects.values())
        # print(f"DEBUG: Returning final aspects list: {result_list}") # Optional
        return result_list
    except Exception as e:
        # Catch unexpected errors during the final conversion step
        print(f"CRITICAL ERROR: Failed to convert final_aspects dict to list: {e}")
        print(f"DEBUG: final_aspects type was: {type(final_aspects)}, value: {final_aspects}")
        return [] # Return empty list on failure


# --- Optional block for self-testing the module ---
# This code runs only if you execute `python app/analysis.py` directly
if __name__ == '__main__':
    print("--- Testing analysis.py module (Simplified: Aspects Only) ---")
    print("Checking VADER..."); print("VADER loaded." if vader_analyzer else "VADER not found.")
    print("\nAttempting to load Aspect model...")
    model_avail = load_models() # Load the model(s)

    # Proceed with tests if either the model or VADER (for rules) is available
    if model_avail or vader_analyzer:
        print("\n--- Running test analyses ---")
        sample_texts = [
            "The screen clarity is amazing, but the battery life is quite poor.",
            "Customer support was friendly and resolved my issue quickly.",
            "I find the mobile app interface very confusing to navigate.",
            "This product is okay, not great but not terrible either.",
            "battery is great but brightness is bad", # Contrastive test
            "its great but lacks efficiency", # Implicit negative rule test
            "the product works well but battery drains fast", # Implicit negative rule test
            "Good price, okay sound.", # Multiple aspects, one via rule
            "", # Empty String test
            None, # None input test
        ]

        for i, text in enumerate(sample_texts):
            print(f"\n--- Analyzing Sample {i+1} ---")
            print(f"Input Text: '{text}'")
            start_asp = time.time()
            # Call the enhanced analyze_aspects function
            aspects = analyze_aspects(text)
            end_asp = time.time()
            print(f"Result Aspects ({end_asp - start_asp:.3f}s): {aspects}")
    else:
        print("\n--- Aspect model failed to load AND VADER not found. Cannot run analysis tests. ---")
        print("Please check error messages above, PyABSA installation, and VADER installation.")
