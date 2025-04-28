from flask import Flask, request, jsonify
from flask_cors import CORS
from monkey_patch import model

app = Flask(__name__)
CORS(app)

@app.route('/query', methods=['POST'])
def query_endpoint():
    try:
        # Parse JSON body properly and force it
        data = request.get_json(force=True)
    except Exception as e:
        return jsonify({'error': 'Invalid JSON format'}), 400

    if not data or 'query' not in data:
        return jsonify({'error': 'No query provided'}), 400

    query_text = data['query']
    #print(query_text)

    # Pass the incoming query to the model
    response_obj = model.process_query(query_text)

    # Ensure consistent output formatting
    if isinstance(response_obj, dict):
        return jsonify(response_obj)
    else:
        # If model returned a string, wrap it in a dict
        return jsonify({'response': response_obj})

if __name__ == '__main__':
    app.run(debug=False, host="0.0.0.0", port = "5000")