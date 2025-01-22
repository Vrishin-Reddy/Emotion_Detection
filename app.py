from flask import Flask, render_template, request, jsonify
from flask_bootstrap import Bootstrap
from flask_caching import Cache
import os
import pandas as pd
import pickle

# Initialize Flask app
app = Flask(__name__)
Bootstrap(app)  # Add Bootstrap for enhanced UI styling

# Caching Configuration
cache = Cache(app, config={'CACHE_TYPE': 'SimpleCache', 'CACHE_DEFAULT_TIMEOUT': 300})

# Paths to the saved models and results
save_dir = "C:\\Users\\vrishin\\Documents\\AIT_526_MP\\model_saves"
results_file = os.path.join(save_dir, "results.csv")
vectorizer_file = os.path.join(save_dir, "tfidf_vect.pk")

# Load model results
@cache.cached(timeout=300, key_prefix='model_results')
def load_model_results():
    if os.path.exists(results_file):
        try:
            return pd.read_csv(results_file)
        except Exception as e:
            print(f"Error loading results file: {e}")
            return pd.DataFrame()
    return pd.DataFrame()

model_results = load_model_results()

# Load vectorizer
@cache.cached(timeout=300, key_prefix='vectorizer')
def load_vectorizer():
    if os.path.exists(vectorizer_file):
        try:
            with open(vectorizer_file, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Error loading vectorizer: {e}")
            return None
    return None

vectorizer = load_vectorizer()

# Load models dynamically
@cache.cached(timeout=300, key_prefix='models')
def load_models():
    loaded_models = {}
    for filename in os.listdir(save_dir):
        if filename.endswith("_text_emotion_model.sav"):
            try:
                model_name = filename.split("_text_emotion_model.sav")[0].replace("_", " ").title()
                with open(os.path.join(save_dir, filename), 'rb') as f:
                    loaded_models[model_name] = pickle.load(f)
            except Exception as e:
                print(f"Error loading model {filename}: {e}")
    return loaded_models

models = load_models()

@app.route('/')
def index():
    """Render the main HTML page."""
    best_model_info = None
    if not model_results.empty:
        best_model_row = model_results.loc[model_results['Composite Score'].idxmax()]
        best_model_info = {
            "model": best_model_row["Model"],
            "accuracy": best_model_row["Accuracy"],
            "composite_score": best_model_row["Composite Score"]
        }
    return render_template('index.html', best_model_info=best_model_info, model_list=list(models.keys()))

@app.route('/get_results', methods=['GET'])
@cache.cached(timeout=300)
def get_results():
    """API endpoint to fetch model results."""
    if model_results.empty:
        return jsonify({"error": "No results found"}), 404
    return jsonify(model_results.to_dict(orient='records'))

@app.route('/get_best_model', methods=['GET'])
@cache.cached(timeout=300)
def get_best_model():
    """API endpoint to fetch the best model."""
    if model_results.empty:
        return jsonify({"error": "No results found"}), 404
    best_model_row = model_results.loc[model_results['Composite Score'].idxmax()]
    return jsonify({
        "model": best_model_row["Model"],
        "accuracy": best_model_row["Accuracy"],
        "composite_score": best_model_row["Composite Score"]
    })

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint to predict emotion for a given sentence."""
    data = request.get_json()
    sentence = data.get('sentence', '').strip()

    if not sentence:
        return jsonify({"error": "No sentence provided"}), 400

    selected_model = request.args.get('model', 'Lasso Regression').replace("_", " ").title()

    if selected_model not in models:
        return jsonify({"error": f"Model '{selected_model}' not found"}), 404

    model = models[selected_model]
    if vectorizer is None:
        return jsonify({"error": "Vectorizer not found"}), 500

    # Preprocess the input
    cleaned_sentence = "".join([char.lower() for char in sentence if char.isalnum() or char.isspace()])
    try:
        vectorized_input = vectorizer.transform([cleaned_sentence])
    except Exception as e:
        return jsonify({"error": f"Vectorizer error: {str(e)}"}), 500

    # Make prediction
    try:
        prediction = model.predict(vectorized_input)
    except Exception as e:
        return jsonify({"error": f"Prediction error: {str(e)}"}), 500

    emotion_mapping = {
        0: 'Anger',
        1: 'Fear',
        2: 'Joy',
        3: 'Love',
        4: 'Sadness',
        5: 'Surprise'
    }
    emotion = emotion_mapping.get(prediction[0], "Unknown Emotion")

    return jsonify({"emotion": emotion})

@app.errorhandler(404)
def not_found_error(error):
    """Custom 404 error page."""
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    """Custom 500 error page."""
    return render_template('500.html'), 500

if __name__ == '__main__':
    app.run(debug=True)
