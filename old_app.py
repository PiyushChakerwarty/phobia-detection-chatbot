from flask import Flask, request, jsonify
import joblib

from data_processing.detect_phobia import symptom_to_features, get_all_combined_symptoms

# Initialize the Flask app
app = Flask(__name__)

models_path = './models/'
# Load the pre-trained models (already trained using Grid Search)
knn_model = joblib.load(models_path + 'best_knn_model.pkl')
lr_model = joblib.load(models_path + 'best_lr_model.pkl')
dt_model = joblib.load(models_path + 'best_dt_model.pkl')

# Store conversation states (simple state management for conversation flow)
user_data = {}


# Intent classifier (basic keyword matching)
def detect_intent(user_input):
    user_input = user_input.lower()
    if 'hi' in user_input or 'hello' in user_input:
        return 'greet'
    elif 'no' in user_input or 'none' in user_input:
        return 'no_more_symptoms'
    elif 'goodbye' in user_input or 'bye' in user_input:
        return 'goodbye'
    else:
        return 'provide_symptoms'


# Integrating bioBERT for symptom extraction (simplified for this example)
def extract_symptoms_with_biobert(text):
    """
    bioBERT is used to extract symptoms from user input.
    In this example, assume bioBERT extracts 'SYMPTOM' entities.
    """
    # For now, assume bioBERT returns a list of symptoms from the text
    symptoms = get_all_combined_symptoms(text)  # Use your bioBERT model here
    return symptoms


# Conversational bot logic with bioBERT for symptom extraction
@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_id = data.get('user_id')
    user_input = data.get('message').lower()

    # Initialize the state if the user is new
    if user_id not in user_data:
        user_data[user_id] = {'state': 'greeting'}

    state = user_data[user_id]['state']
    intent = detect_intent(user_input)

    # Greeting state
    if intent == 'greet':
        user_data[user_id]['state'] = 'asking_symptoms'
        return jsonify(
            {"response": "Hello! I'm here to assist you. How are you feeling today? Could you describe your symptoms?"})

    # Providing symptoms state
    elif intent == 'provide_symptoms':
        symptoms = extract_symptoms_with_biobert(user_input)  # Use bioBERT to detect symptoms
        if symptoms:
            # Store symptoms and ask for more
            user_data[user_id]['symptoms'] = symptoms
            user_data[user_id]['state'] = 'more_symptoms'
            return jsonify({
                               "response": f"I detected the following symptoms: {', '.join(symptoms)}. Are there any more symptoms you're experiencing?"})
        else:
            return jsonify(
                {"response": "I couldn't recognize any symptoms. Could you describe them again in more detail?"})

    # Asking for more symptoms
    elif intent == 'no_more_symptoms':
        user_data[user_id]['state'] = 'predicting'
        return jsonify({"response": "Thank you! Let me analyze your symptoms and provide a prediction."})

    # Predict phobia based on symptoms
    elif state == 'predicting':
        symptoms = user_data[user_id].get('symptoms', [])
        symptom_list = symptoms  # Assuming bioBERT has already extracted symptoms

        # Convert symptoms to feature vector for models (you already have this logic)
        features = symptom_to_features(symptom_list, uploaded_symptom_weights_df)

        # Predict using all three models
        knn_prediction = knn_model.predict(features)[0]
        lr_prediction = lr_model.predict(features)[0]
        dt_prediction = dt_model.predict(features)[0]

        # Majority voting or consensus
        predictions = [knn_prediction, lr_prediction, dt_prediction]
        final_prediction = max(set(predictions), key=predictions.count)

        # Clear the state for next conversation
        user_data[user_id]['state'] = 'greeting'

        return jsonify({
            "response": f"Based on the symptoms, it seems you might be experiencing {final_prediction}. Feel free to ask more questions!"
        })

    # Goodbye intent
    elif intent == 'goodbye':
        user_data[user_id]['state'] = 'greeting'  # Reset state
        return jsonify({"response": "Goodbye! Stay safe and feel free to come back anytime if you need help."})

    # Default response for unrecognized input
    else:
        return jsonify({"response": "I'm not sure what you meant. Could you clarify or provide more details?"})


if __name__ == '__main__':
    app.run(debug=True)
