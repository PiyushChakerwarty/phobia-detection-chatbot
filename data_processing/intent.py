from fuzzywuzzy import process
import detect_phobia
# %%
# Load the CSV file into a DataFrame
df_intents = pd.read_csv("../data_collection/data/intent.csv")

# Convert the phrases and intents into lists for easy matching
phrases = df_intents["Word"].tolist()
intents = df_intents["Intent"].tolist()


def get_intent(user_input):
    # Find the closest matching phrase and its score
    match, score = process.extractOne(user_input, phrases)
    print(match, score)
    
    # Set a threshold for how close the match needs to be
    threshold = 90
    
    # If the match score is above the threshold, return the corresponding intent
    if score > threshold:
        intent_index = phrases.index(match)
        return intents[intent_index]
    else:
        # If no good match is found, return "symptoms" as default intent
        return "symptoms"


# %%
import random
# Personalized greeting responses
def personalized_greeting():
    responses = [
        "Hello! How can I assist you today?",
        "Hi there! What can I do for you?",
        "Hey! How are you feeling today?",
        "Good to see you! How can I help?"
    ]
    return random.choice(responses)

# Friendly goodbye responses
def friendly_goodbye():
    responses = [
        "Goodbye! Take care, and don't hesitate to reach out if you need help.",
        "Bye! Have a great day ahead!",
        "See you later! Stay safe.",
        "Goodbye! I'm here if you need anything."
    ]
    return random.choice(responses)

# No symptoms confirmation
def no_symptoms_response():
    
    responses = [
        "Glad to hear you're feeling well! Is there anything else I can help you with?",
        "That's great! Let me know if there's anything else you need assistance with.",
        "Awesome! If you ever need help, feel free to reach out.",
        "Wonderful to hear you're in good health. Is there anything else on your mind?"
    ]
    return random.choice(responses)

# Empathetic response for symptoms
def empathetic_symptom_response(symptoms):
    if symptoms:
        return f"It sounds like you're experiencing {', '.join(symptoms)}. I'm sorry to hear that. Can you tell me more about how long you've been feeling this way?"
    else:
        return "I understand you're not feeling well. Could you describe your symptoms in more detail?"

# Ask for clarification if the intent is unknown or unclear
def ask_for_clarification():
    responses = [
        "I'm not sure I understood that. Could you please rephrase?",
        "Can you clarify what you're asking about?",
        "I didn't catch that. Could you provide more details?",
        "Sorry, I didn’t quite get that. Can you try again?"
    ]
    return random.choice(responses)

# Main function to handle user input and return a response
def handle_input(user_input):
    intent = get_intent(user_input)
    print(intent)
    
    if intent == "greeting":
        return personalized_greeting()
    elif intent == "goodbye":
        return friendly_goodbye()
    elif intent == "no_symptoms":
        return no_symptoms_response()
    elif intent == "symptoms":
        symptoms = detect_phobia.get_all_combined_symptoms(user_input)
        return empathetic_symptom_response(symptoms)
    else:
        return ask_for_clarification()
    
user_input = "I am feeling some headaches and dizziness"
print(handle_input(user_input))