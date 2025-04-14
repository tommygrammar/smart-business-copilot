from flask import Flask, request, jsonify
from flask_cors import CORS
from probabilistic_intent_language_model import model



app = Flask(__name__)
CORS(app)

# Monkey-patch the model's train method to handle dict responses and to never fail.
original_train = model.train
def patched_train(query, response):
    try:
        return original_train(query, response)
    except Exception as e:
        return None

# Always apply the monkey patch.
model.train = patched_train

@app.route('/query', methods=['POST'])
def query_endpoint():
    try:
        # Parse JSON body properly and defensively
        data = request.get_json(force=True)
    except Exception as e:
        return jsonify({'error': 'Invalid JSON format'}), 400

    if not data or 'query' not in data:
        return jsonify({'error': 'No query provided'}), 400

    query_text = data['query']
    print(query_text)

    # Directly pass to the model as you would normally
    response_obj = model.process_query(query_text)

    # Ensure consistent output formatting
    if isinstance(response_obj, dict):
        return jsonify(response_obj)
    else:
        # If model returned a string, wrap it in a dict
        return jsonify({'response': response_obj})

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port ="5000")
