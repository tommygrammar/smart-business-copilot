import json
from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
from probabilistic_intent_language_model import ProbabilisticLanguageModel, defaultdict

def load_model(filename):
    with open(filename, "rb") as f:
        model = pickle.load(f)
    return model

model_filename = "probabilistic_intent_language_model.pkl"
try:
    model = load_model(model_filename)
except Exception as e:
    print("Error loading model:", e)
    exit(1)

app = Flask(__name__)
CORS(app)

# Monkey-patch the model's train method to handle dict responses and to never fail.
original_train = model.train
def patched_train(query, response):
    # Instead of stripping out graph data, pass the full response (or
    # handle both the narrative and graph data appropriately).
    try:
        return original_train(query, response)
    except Exception as e:
        #print("Error during training:", e)
        return None

    
    

# Always apply the monkey patch.
model.train = patched_train

@app.route('/query', methods=['POST'])
def query_endpoint():
    data = request.get_json()
    if not data or 'query' not in data:
        return jsonify({'error': 'No query provided'}), 400
    query_text = data['query']
    # Call process_query, which now uses the patched train method.
    response_obj = model.process_query(query_text)
    # Ensure that process_query returns a dict.
    if isinstance(response_obj, str):
        try:
            response_obj = json.loads(response_obj)
        except Exception as e:
            response_obj = {'response': response_obj}
    return jsonify(response_obj)

if __name__ == '__main__':
    app.run(debug=True, host = "0.0.0.0", port = "5000")
