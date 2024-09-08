import os
import string

import nltk
import pandas as pd
import spacy
from fuzzywuzzy import process
from nltk import ngrams
from spacy.matcher import PhraseMatcher
from transformers import pipeline
from transformers import BertTokenizer, BertForTokenClassification

symptom_df = pd.read_csv('./data_collection/data/nhs/symptom_weights.csv')
symptom_phrases = symptom_df['symptom'].tolist()  # Extract symptoms from DataFrame

spacy_model_base_path = spacy.util.get_package_path('en_core_web_sm')
subdir = 'en_core_web_sm-2.3.1'

# Join the paths
spacy_model_full_path = os.path.join(spacy_model_base_path, subdir)


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


def get_all_combined_symptoms(user_input):
    combined_symptoms = extract_symptoms_phraser(user_input) + extract_symptoms_ngrams(user_input) + extract_symptoms_doc(user_input) + extract_symptoms_biobert(user_input)
    combined_symptoms = list(set(combined_symptoms))
    return combined_symptoms