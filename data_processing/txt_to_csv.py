import re
import pandas as pd
import os

input_file = os.path.join('data_collection', 'data', 'nhs', 'symptoms.txt')

# Step 1: Load the text file
with open(input_file, 'r') as file:
    text = file.read()

# Step 2: Parse the text to extract symptoms and their associated phobias
# This regular expression identifies sections of the text that list symptoms
# and assumes the symptoms and associated phobias are described close to each other.

# Regular expressions to identify symptoms and related text
symptom_pattern = r'(\w[\w\s-]+):\s*(\w[\w\s-]*)'

# Extract symptoms and associated text
matches = re.findall(symptom_pattern, text)

# Step 3: Organize the extracted data into a structured format
data = []
for match in matches:
    symptom = match[0].strip()
    associated_phobia = match[1].strip()
    data.append({'Symptom': symptom, 'Associated Phobia': associated_phobia})

# Convert to DataFrame
df = pd.DataFrame(data)

otput_file = os.path.join('data_collection', 'data', 'nhs', 'symptoms.csv')
# Step 4: Save the structured data to a CSV file
df.to_csv(otput_file, index=False)

print("Structured dataset created and saved as 'phobia_symptoms.csv'")
