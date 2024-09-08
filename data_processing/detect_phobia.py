# %%
import pandas as pd
import os
from transformers import pipeline
import spacy
from spacy.matcher import PhraseMatcher
import nltk
from nltk import ngrams
import string

# %%
'''
0.1 to 0.3: Low association
0.4 to 0.6: Moderate association
0.7 to 1.0: Strong association
'''
# data_collection/data/nhs/symptom_weights.json
symptom_df = pd.read_csv('../data_collection/data/nhs/symptom_weights.csv')
symptom_df.head()

# %%
symptom_phrases = symptom_df['symptom'].tolist()  # Extract symptoms from DataFrame
symptom_phrases

# %%
user_input = "I feel shortness of breath and have a choking sensation and headaches."

# %% [markdown]
# ### Using Spacy PhraseMatcher to get list of symptoms

# %%
import spacy
spacy_model_base_path = spacy.util.get_package_path('en_core_web_sm')
subdir = 'en_core_web_sm-2.3.1'

# Join the paths
spacy_model_full_path = os.path.join(spacy_model_base_path, subdir)

# Print the result
print(spacy_model_full_path)


# %%
def extract_symptoms_phraser(user_input):
    # Load Spacy model
    nlp = spacy.load(spacy_model_full_path)

    # Initialize PhraseMatcher and add patterns for multi-word symptoms
    matcher = PhraseMatcher(nlp.vocab)

    patterns = [nlp(symptom) for symptom in symptom_phrases]
    matcher.add("SYMPTOM", patterns)

    # Process user input
    doc = nlp(user_input)

    # Find matching symptoms in the user input
    matches = matcher(doc)
    extracted_symptoms_phraser = [doc[start:end].text for match_id, start, end in matches]

    print("Phrase Matched Symptoms:", extracted_symptoms_phraser)
    return list(set(extracted_symptoms_phraser))

# %% [markdown]
# ### n-grams for Symptom Detection

# %%
def extract_symptoms_ngrams(user_input):
    # Preprocess the user input: convert to lowercase and remove punctuation
    user_input = user_input.lower()
    user_input = user_input.translate(str.maketrans('', '', string.punctuation))

    # Tokenize user input
    tokens = nltk.word_tokenize(user_input)

    # Generate bigrams and trigrams
    bigrams = list(ngrams(tokens, 2))
    trigrams = list(ngrams(tokens, 3))

    # Convert n-grams back to phrases
    bigrams_phrases = [' '.join(gram) for gram in bigrams]
    trigrams_phrases = [' '.join(gram) for gram in trigrams]

    # Combine bigrams and trigrams
    all_ngrams = bigrams_phrases + trigrams_phrases

    # Match n-grams with symptom list (case-insensitive)
    extracted_symptoms_ngrams = [phrase for phrase in all_ngrams if phrase in symptom_phrases]

    print("n-gram Matched Symptoms:", extracted_symptoms_ngrams)
    return list(set(extracted_symptoms_ngrams))


# %% [markdown]
# ## Dependency Parsing for Symptom Detection

# %%

# Function to extract symptoms based on dependencies
def extract_symptoms_doc(user_input):
    # Load Spacy model
    nlp = spacy.load(spacy_model_full_path)

    # Process user input
    doc = nlp(user_input)
    
    # Loop through tokens and print dependencies
    # for token in doc:
    #     print(f"Token: {token.text}, Dependency: {token.dep_}, Head: {token.head.text}, Children: {[child.text for child in token.children]}")
    
    
    symptoms = []
    for token in doc:
        # Check for "amod" + "dobj" (e.g., "choking sensation")
        if token.dep_ == "amod" and token.head.dep_ == "dobj":
            symptom = f"{token.text} {token.head.text}"
            symptoms.append(symptom)
        
        # Check for "attr" + "prep" + "pobj" (e.g., "shortness of breath")
        if token.dep_ == "attr" and len([child for child in token.children if child.dep_ == "prep"]) > 0:
            prep = [child for child in token.children if child.dep_ == "prep"][0]
            pobj = [child for child in prep.children if child.dep_ == "pobj"][0]
            symptom = f"{token.text} {prep.text} {pobj.text}"
            symptoms.append(symptom)
        
        # Check for "dobj" conjunction (e.g., "headaches and dizziness")
        if token.dep_ == "dobj" and len([child for child in token.children if child.dep_ == "conj"]) > 0:
            conj = [child for child in token.children if child.dep_ == "conj"][0]
            symptom = f"{token.text} and {conj.text}"
            symptoms.append(symptom)
        
    return symptoms

# Extract symptoms from the doc
# extracted_symptoms_doc = list(set(extract_symptoms_doc(user_input)))
# print("Extracted Symptoms from doc:", extracted_symptoms_doc)

# %% [markdown]
# ### Using BioBERT and Fuzzywuzzy for Symptom detection

# %%
from transformers import BertTokenizer, BertForTokenClassification, pipeline
from fuzzywuzzy import process

def extract_symptoms_biobert(user_input):
    # Load the pre-trained BioBERT model for NER
    model_name = "dmis-lab/biobert-base-cased-v1.1"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForTokenClassification.from_pretrained(model_name)

    # Use the pipeline for named entity recognition (NER)
    ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer)

    # Function to match extracted symptoms with the known symptom list using fuzzy matching
    def match_symptom(extracted_entity, symptom_phrases, threshold=90):
        match, score = process.extractOne(extracted_entity, symptom_phrases)
        if score >= threshold:
            return match
        return None

    # Perform Named Entity Recognition (NER) to extract potential symptoms
    entities = ner_pipeline(user_input)

    # Extract entities from the NER results
    extracted_entities = [entity['word'].lower() for entity in entities]

    # Match each extracted entity with the known symptom list
    extracted_symptoms_bert = [match_symptom(entity, symptom_phrases) for entity in extracted_entities if match_symptom(entity, symptom_phrases)]

    print("Matched Symptoms BERT and fuzzywuzzy:", set(extracted_symptoms_bert))
    return list(set(extracted_symptoms_bert))


# %% [markdown]
# ### Combined list of Symptoms

# %%
def get_all_combined_symptoms(user_input):
    combined_symptoms = extract_symptoms_phraser(user_input) + extract_symptoms_ngrams(user_input) + extract_symptoms_doc(user_input) + extract_symptoms_biobert(user_input)
    combined_symptoms = list(set(combined_symptoms))
    return combined_symptoms
get_all_combined_symptoms(user_input)

# %%
get_all_combined_symptoms("Having trouble breathing")

# %% [markdown]
# # Model Creation

# %%
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, LeaveOneOut, cross_val_predict
from sklearn.metrics import classification_report, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# %%
# Preprocess the data
X = symptom_df.drop(columns=['symptom'])  # Features
y = X.idxmax(axis=1)

# Initialize the Decision Tree model
decision_tree_model = DecisionTreeClassifier(random_state=42)

# Leave-One-Out Cross-Validation (LOOCV)
loo = LeaveOneOut()

# Fit and predict using LOOCV
y_pred_dt = cross_val_predict(decision_tree_model, X, y, cv=loo)

# Calculate metrics
dt_accuracy = accuracy_score(y, y_pred_dt)
dt_classification_report = classification_report(y, y_pred_dt)

# Results
print(f"Decision Tree Accuracy: {dt_accuracy}")
print(f"Decision Tree Classification Report: \n{dt_classification_report}")

# %%
# Initialize the KNN model
knn_model = KNeighborsClassifier(n_neighbors=3)

# Fit and predict using LOOCV
y_pred_knn = cross_val_predict(knn_model, X, y, cv=loo)

# Calculate metrics
knn_accuracy = accuracy_score(y, y_pred_knn)
knn_classification_report = classification_report(y, y_pred_knn)

# Results
print(f"KNN Accuracy: {knn_accuracy}")
print(f"KNN Classification Report: \n{knn_classification_report}")


# %%
# Initialize the Logistic Regression model
logistic_regression_model = LogisticRegression(max_iter=1000)

# Fit and predict using LOOCV
y_pred_lr = cross_val_predict(logistic_regression_model, X, y, cv=loo)

# Calculate metrics
lr_accuracy = accuracy_score(y, y_pred_lr)
lr_classification_report = classification_report(y, y_pred_lr)

# Results
print(f"Logistic Regression Accuracy: {lr_accuracy}")
print(f"Logistic Regression Classification Report: \n{lr_classification_report}")


# %%
# Initialize the Naive Bayes model
naive_bayes_model = MultinomialNB()

# Fit and predict using LOOCV
y_pred_nb = cross_val_predict(naive_bayes_model, X, y, cv=loo)

# Calculate metrics
nb_accuracy = accuracy_score(y, y_pred_nb)
nb_classification_report = classification_report(y, y_pred_nb)

# Results
print(f"Naive Bayes Accuracy: {nb_accuracy}")
print(f"Naive Bayes Classification Report: \n{nb_classification_report}")


# %%
print(f"Decision Tree Accuracy: {dt_accuracy}")
print(f"KNN Accuracy: {knn_accuracy}")
print(f"Logistic Regression Accuracy: {lr_accuracy}")
print(f"Naive Bayes Accuracy: {nb_accuracy}")


# %%
# Placeholder accuracy values for each model (replace these with actual accuracy scores)
model_names = ['Decision Tree', 'KNN', 'Logistic Regression', 'Naive Bayes']
accuracies = [dt_accuracy, knn_accuracy, lr_accuracy, nb_accuracy]

# Bar Plot for Accuracy Comparison
plt.figure(figsize=(10, 6))
sns.barplot(x=model_names, y=accuracies)
plt.title('Model Accuracy Comparison')
plt.ylabel('Accuracy')
plt.ylim(0, 1)  # Set y-axis limits to 0-1 for easier interpretation
plt.show()


# %%
# Function to plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=set(y), yticklabels=set(y))
    plt.title(f'Confusion Matrix for {model_name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

# Plot confusion matrix for each model
plot_confusion_matrix(y, y_pred_dt, 'Decision Tree')
plot_confusion_matrix(y, y_pred_knn, 'KNN')
plot_confusion_matrix(y, y_pred_lr, 'Logistic Regression')
plot_confusion_matrix(y, y_pred_nb, 'Naive Bayes')


# %% [markdown]
# ## Model Tuning

# %% [markdown]
# #### Grid Search for KNN

# %%
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

# Define hyperparameter grid for KNN
param_grid_knn = {
    'n_neighbors': [1, 3, 5, 7, 9]  # Different numbers of neighbors
}

# Initialize KNN model
knn_model = KNeighborsClassifier()

# Grid Search with Cross-Validation
grid_search_knn = GridSearchCV(knn_model, param_grid_knn, cv=5, scoring='accuracy')

# Fit the model to find the best hyperparameters
grid_search_knn.fit(X, y)

# Best parameters and the corresponding accuracy
best_knn_params = grid_search_knn.best_params_
best_knn_score = grid_search_knn.best_score_

print(f"Best KNN Params: {best_knn_params}")
print(f"Best KNN Accuracy: {best_knn_score}")


# %% [markdown]
# #### Grid Search for Logistic Regression

# %%
from sklearn.linear_model import LogisticRegression

# Define hyperparameter grid for Logistic Regression
param_grid_lr = {
    'C': [0.01, 0.1, 1, 10, 100]  # Regularization strength
}

# Initialize Logistic Regression model
logistic_regression_model = LogisticRegression(max_iter=1000)

# Grid Search with Cross-Validation
grid_search_lr = GridSearchCV(logistic_regression_model, param_grid_lr, cv=5, scoring='accuracy')

# Fit the model to find the best hyperparameters
grid_search_lr.fit(X, y)

# Best parameters and the corresponding accuracy
best_lr_params = grid_search_lr.best_params_
best_lr_score = grid_search_lr.best_score_

print(f"Best Logistic Regression Params: {best_lr_params}")
print(f"Best Logistic Regression Accuracy: {best_lr_score}")


# %% [markdown]
# #### Grid Search for Decision Tree

# %%
from sklearn.tree import DecisionTreeClassifier

# Define hyperparameter grid for Decision Tree
param_grid_dt = {
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

# Initialize Decision Tree model
decision_tree_model = DecisionTreeClassifier(random_state=42)

# Grid Search with Cross-Validation
grid_search_dt = GridSearchCV(decision_tree_model, param_grid_dt, cv=5, scoring='accuracy')

# Fit the model to find the best hyperparameters
grid_search_dt.fit(X, y)

# Best parameters and the corresponding accuracy
best_dt_params = grid_search_dt.best_params_
best_dt_score = grid_search_dt.best_score_

print(f"Best Decision Tree Params: {best_dt_params}")
print(f"Best Decision Tree Accuracy: {best_dt_score}")


# %% [markdown]
# ### Compare the Best Models

# %%
# Display results for all models
print(f"Best KNN Params: {best_knn_params}, Accuracy: {best_knn_score}")
print(f"Best Logistic Regression Params: {best_lr_params}, Accuracy: {best_lr_score}")
print(f"Best Decision Tree Params: {best_dt_params}, Accuracy: {best_dt_score}")


# %% [markdown]
# ## Predict Phobia

# %%
def symptom_to_features(symptom_list, dataset):
    """
    Convert a list of symptoms to the corresponding feature vector using the dataset.
    Input: List of symptoms, dataset (uploaded_symptom_weights_df)
    Output: A feature vector corresponding to the symptoms
    """
    feature_vector = [0] * len(dataset.columns[1:])  # Initialize a zero vector for all phobias
    for symptom in symptom_list:
        if symptom in dataset['symptom'].values:
            symptom_data = dataset[dataset['symptom'] == symptom].iloc[0, 1:].values
            feature_vector = [max(f, s) for f, s in zip(feature_vector, symptom_data)]  # Max of both vectors
    return feature_vector


# %%
def predict_phobia(symptom_list, models, dataset):
    """
    Predict phobia based on symptoms using the provided models (KNN, Logistic Regression, Decision Tree).
    Input: List of symptoms, trained models (dictionary), dataset
    Output: Predictions from each model
    """
    # Convert symptoms to feature vector
    features = [symptom_to_features(symptom_list, dataset)]
    
    # Dictionary to store results from each model
    predictions = {}
    
    # Predict using KNN
    predictions['KNN'] = models['KNN'].predict(features)
    
    # Predict using Logistic Regression
    predictions['Logistic Regression'] = models['Logistic Regression'].predict(features)
    
    # Predict using Decision Tree
    predictions['Decision Tree'] = models['Decision Tree'].predict(features)
    
    return predictions


# %%
# Assuming models are already trained and stored
models = {
    'KNN':grid_search_knn,  # The trained KNN model
    'Logistic Regression': grid_search_lr,  # The trained Logistic Regression model
    'Decision Tree': grid_search_dt  # The trained Decision Tree model
}

# Example symptoms provided by the user
user_symptoms = get_all_combined_symptoms(user_input)

# Predict phobia based on symptoms
predictions = predict_phobia(user_symptoms, models, symptom_df)

# Display the predictions
for model, prediction in predictions.items():
    print(f"{model} predicts: {prediction[0]}")


# %%
import joblib

# Save each of the trained models
joblib.dump(grid_search_knn, '../models/best_knn_model.pkl')
joblib.dump(grid_search_lr, '../models/best_lr_model.pkl')
joblib.dump(grid_search_dt, '../models/best_dt_model.pkl')


# %%
from collections import Counter

def majority_voting(predictions):
    """
    Use majority voting to select the final predicted phobia.
    Input: Predictions from different models
    Output: Final predicted phobia based on majority voting
    """
    votes = [prediction[0] for prediction in predictions.values()]  # Collect all model predictions
    vote_count = Counter(votes)  # Count the occurrences of each predicted phobia
    final_prediction = vote_count.most_common(1)[0][0]  # Select the most common prediction
    return final_prediction

# Example usage
final_prediction = majority_voting(predictions)
print(f"Final predicted phobia based on majority voting: {final_prediction}")


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
        symptoms = get_all_combined_symptoms(user_input)
        return empathetic_symptom_response(symptoms)
    else:
        return ask_for_clarification()

# Example interaction
user_input = "I'm feeling a bit dizzy and my heart is racing."
response = handle_input(user_input)
print(response)

# %%
# Example input
user_input = "I'm feeling a bit dizzy and I have a rapid heartbeat."
response = handle_input(user_input)
print(response)


# %%




