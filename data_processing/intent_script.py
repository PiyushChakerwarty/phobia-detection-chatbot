# %%
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import GridSearchCV
from fuzzywuzzy import process
import pandas as pd
import matplotlib.pyplot as plt
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset



# %% [markdown]
# ## Load Dataset

# %%
# Step 1: Load the dataset
intent_path = "../data_collection/data/intent.csv"
df_intents = pd.read_csv(intent_path)
df_intents.head()

# %% [markdown]
# ## Fuzzy Matching

# %%
# Convert the phrases and intents into lists for easy matching
phrases = df_intents["Word"].tolist()
intents = df_intents["Intent"].tolist()

def get_intent(user_input):
    # Find the closest matching phrase and its score
    match, score = process.extractOne(user_input, phrases)
    print(f"Matched phrase: {match}, Score: {score}")
    
    # Set a more lenient threshold
    threshold = 85
    
    # If the match score is above the threshold, return the corresponding intent
    if score >= threshold:
        intent_index = phrases.index(match)
        return intents[intent_index]
    else:
        # If no good match is found, return "unknown" and ask for clarification
        return "unknown"

# Asking for clarification if intent is "unknown"
def handle_input(user_input):
    intent = get_intent(user_input)
    return intent




# %%
user_input = "hello"
print(handle_input(user_input))  # Expected: A greeting intent response

user_input = "I don't have any issues"
print(handle_input(user_input))  # Expected: A no_symptoms intent response

user_input = "I'm fine."
print(handle_input(user_input))  # Expected: A no_symptoms intent response

user_input = "I'm experiencing headaches and dizziness."
print(handle_input(user_input))  # Expected: A symptoms intent response


# %% [markdown]
# ## Using Machine Learning Models

# %% [markdown]
# ### Split the dataset into phrases (X) and intents (y)

# %%

X = df_intents["Word"]
y = df_intents["Intent"]

# %% [markdown]
# ### Vectorize the text using TF-IDF

# %%
vectorizer = TfidfVectorizer()
X_vectors = vectorizer.fit_transform(X)

# %% [markdown]
# ### Split the data into training and testing sets

# %%

X_train, X_test, y_train, y_test = train_test_split(X_vectors, y, test_size=0.2, random_state=42)


# %% [markdown]
# ### Train and compare classifiers

# %% [markdown]
# #### Logistic Regression

# %%
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
y_pred_log_reg = log_reg.predict(X_test)
print("Logistic Regression")
print("Accuracy:", accuracy_score(y_test, y_pred_log_reg))
print("Classification Report:\n", classification_report(y_test, y_pred_log_reg))

# %% [markdown]
# #### Naive Bayes

# %%
nb = MultinomialNB()
nb.fit(X_train, y_train)
y_pred_nb = nb.predict(X_test)
print("\nNaive Bayes")
print("Accuracy:", accuracy_score(y_test, y_pred_nb))
print("Classification Report:\n", classification_report(y_test, y_pred_nb))

# %% [markdown]
# #### Support Vector Machine (SVM)

# %%
svm = SVC(kernel='linear')  # Using linear kernel for text classification
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)
print("\nSupport Vector Machine (SVM)")
print("Accuracy:", accuracy_score(y_test, y_pred_svm))
print("Classification Report:\n", classification_report(y_test, y_pred_svm))

# %% [markdown]
# #### Random Forest

# %%
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print("\nRandom Forest")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Classification Report:\n", classification_report(y_test, y_pred_rf))

# %% [markdown]
# ### Calculate Results

# %%
# Calculate metrics for each model
models = ["Logistic Regression", "SVM", "Random Forest"]
y_preds = [y_pred_log_reg, y_pred_svm, y_pred_rf]

metrics_data = {
    'Model': [],
    'Accuracy': [],
    'Precision': [],
    'Recall': [],
    'F1 Score': []
}

for i, y_pred in enumerate(y_preds):
    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
    
    metrics_data['Model'].append(models[i])
    metrics_data['Accuracy'].append(acc)
    metrics_data['Precision'].append(prec)
    metrics_data['Recall'].append(rec)
    metrics_data['F1 Score'].append(f1)




# %%
# Convert to DataFrame
metrics_df = pd.DataFrame(metrics_data)
metrics_df

# %%
# Plot the results
plt.figure(figsize=(12, 8))

# Accuracy plot
plt.subplot(2, 2, 1)
plt.bar(metrics_df['Model'], metrics_df['Accuracy'], color='blue')
plt.title('Accuracy Comparison')
plt.ylabel('Accuracy')

# Precision plot
plt.subplot(2, 2, 2)
plt.bar(metrics_df['Model'], metrics_df['Precision'], color='orange')
plt.title('Precision Comparison')
plt.ylabel('Precision')

# Recall plot
plt.subplot(2, 2, 3)
plt.bar(metrics_df['Model'], metrics_df['Recall'], color='green')
plt.title('Recall Comparison')
plt.ylabel('Recall')

# F1 Score plot
plt.subplot(2, 2, 4)
plt.bar(metrics_df['Model'], metrics_df['F1 Score'], color='red')
plt.title('F1 Score Comparison')
plt.ylabel('F1 Score')

plt.tight_layout()
plt.show()


# %% [markdown]
# ## Hypertuning

# %%

# 1. Logistic Regression
log_reg_param_grid = {
    'C': [0.1, 1, 10],
    'penalty': ['l2'],  # L1 is not supported by LogisticRegression with liblinear
    'solver': ['lbfgs', 'liblinear']
}

# 2. Naive Bayes (MultinomialNB doesn't have many hyperparameters, so we skip it for simplicity)

# 3. Support Vector Machine (SVM)
svm_param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],  # Linear and Radial Basis Function kernels
    'gamma': ['scale', 'auto']  # Relevant for non-linear kernels
}

# 4. Random Forest
rf_param_grid = {
    'n_estimators': [50, 100, 200],  # Number of trees
    'max_depth': [None, 10, 20, 30],  # Tree depth
    'min_samples_split': [2, 5, 10],  # Minimum samples required to split a node
    'min_samples_leaf': [1, 2, 4]  # Minimum samples required in a leaf node
}


# %% [markdown]
# ### Set up GridSearchCV for each model

# %%
# Logistic Regression
grid_search_log_reg = GridSearchCV(log_reg, log_reg_param_grid, cv=5, scoring='accuracy', verbose=1, n_jobs=-1)
grid_search_log_reg.fit(X_train, y_train)

# Support Vector Machine
grid_search_svm = GridSearchCV(svm, svm_param_grid, cv=5, scoring='accuracy', verbose=1, n_jobs=-1)
grid_search_svm.fit(X_train, y_train)

# Random Forest
grid_search_rf = GridSearchCV(rf, rf_param_grid, cv=5, scoring='accuracy', verbose=1, n_jobs=-1)
grid_search_rf.fit(X_train, y_train)

# %% [markdown]
# #### Get the best models and parameters after tuning

# %%
best_log_reg = grid_search_log_reg.best_estimator_
best_svm = grid_search_svm.best_estimator_
best_rf = grid_search_rf.best_estimator_

print("Best parameters for Logistic Regression:", grid_search_log_reg.best_params_)
print("Best parameters for SVM:", grid_search_svm.best_params_)
print("Best parameters for Random Forest:", grid_search_rf.best_params_)

# %% [markdown]
# #### Test the best models on the test set

# %%
# Predictions with the best models after tuning
y_pred_log_reg_tuned = best_log_reg.predict(X_test)
y_pred_svm_tuned = best_svm.predict(X_test)
y_pred_rf_tuned = best_rf.predict(X_test)


# %% [markdown]
# #### Calculate metrics

# %%
# Calculate metrics for the tuned models
y_preds_tuned = [y_pred_log_reg_tuned, y_pred_svm_tuned, y_pred_rf_tuned]
metrics_data_tuned = {
    'Model': [],
    'Accuracy': [],
    'Precision': [],
    'Recall': [],
    'F1 Score': []
}

for i, y_pred in enumerate(y_preds_tuned):
    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
    
    metrics_data_tuned['Model'].append(models[i])
    metrics_data_tuned['Accuracy'].append(acc)
    metrics_data_tuned['Precision'].append(prec)
    metrics_data_tuned['Recall'].append(rec)
    metrics_data_tuned['F1 Score'].append(f1)



# %%
# Convert to DataFrame
metrics_df_tuned = pd.DataFrame(metrics_data_tuned)
metrics_df_tuned

# %% [markdown]
# #### Plot Results

# %%
# Create a two-bar comparison for before and after tuning

# Plot the results comparing before and after tuning
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Accuracy plot
axes[0, 0].bar(metrics_df['Model'], metrics_df['Accuracy'], color='blue', alpha=0.6, label="Before Tuning", width=0.4, align='center')
axes[0, 0].bar(metrics_df_tuned['Model'], metrics_df_tuned['Accuracy'], color='blue', alpha=1.0, label="After Tuning", width=0.4, align='edge')
axes[0, 0].set_title('Accuracy Comparison')
axes[0, 0].set_ylabel('Accuracy')
axes[0, 0].legend()

# Precision plot
axes[0, 1].bar(metrics_df['Model'], metrics_df['Precision'], color='orange', alpha=0.6, label="Before Tuning", width=0.4, align='center')
axes[0, 1].bar(metrics_df_tuned['Model'], metrics_df_tuned['Precision'], color='orange', alpha=1.0, label="After Tuning", width=0.4, align='edge')
axes[0, 1].set_title('Precision Comparison')
axes[0, 1].set_ylabel('Precision')
axes[0, 1].legend()

# Recall plot
axes[1, 0].bar(metrics_df['Model'], metrics_df['Recall'], color='green', alpha=0.6, label="Before Tuning", width=0.4, align='center')
axes[1, 0].bar(metrics_df_tuned['Model'], metrics_df_tuned['Recall'], color='green', alpha=1.0, label="After Tuning", width=0.4, align='edge')
axes[1, 0].set_title('Recall Comparison')
axes[1, 0].set_ylabel('Recall')
axes[1, 0].legend()

# F1 Score plot
axes[1, 1].bar(metrics_df['Model'], metrics_df['F1 Score'], color='red', alpha=0.6, label="Before Tuning", width=0.4, align='center')
axes[1, 1].bar(metrics_df_tuned['Model'], metrics_df_tuned['F1 Score'], color='red', alpha=1.0, label="After Tuning", width=0.4, align='edge')
axes[1, 1].set_title('F1 Score Comparison')
axes[1, 1].set_ylabel('F1 Score')
axes[1, 1].legend()

plt.tight_layout()
plt.show()



# %% [markdown]
# ## Expoting models

# %%
import joblib
import os

# path for intent models
INTENT_MODEL_PATH = "../models/intent"

# Save the Logistic Regression model
joblib.dump(best_log_reg, os.path.join(INTENT_MODEL_PATH, 'logistic_regression_model.pkl'))

# Save the TF-IDF vectorizer
joblib.dump(vectorizer, os.path.join(INTENT_MODEL_PATH,'tfidf_vectorizer.pkl'))

print("Model and vectorizer exported successfully!")


# %%
import joblib

# Load the Logistic Regression model
loaded_model = joblib.load(os.path.join(INTENT_MODEL_PATH, 'logistic_regression_model.pkl'))

# Load the TF-IDF vectorizer
loaded_vectorizer = joblib.load(os.path.join(INTENT_MODEL_PATH, 'tfidf_vectorizer.pkl'))

print("Model and vectorizer loaded successfully!")

# Example user input
new_input = ["I am feeling dizzy and have a headache"]

# Transform the input using the loaded TF-IDF vectorizer
new_input_vectorized = loaded_vectorizer.transform(new_input)

# Predict using the loaded model
predicted_intent = loaded_model.predict(new_input_vectorized)

print(f"Predicted Intent: {predicted_intent[0]}")



