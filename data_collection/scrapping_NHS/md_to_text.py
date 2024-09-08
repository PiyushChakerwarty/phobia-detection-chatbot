import re

# Function to clean and normalize text
def clean_text(text):
    # Remove markdown syntax
    text = re.sub(r'\*\*|\#|\-|\\|\_', '', text)
    # Convert to lowercase
    text = text.lower()
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Clean the content of each file
cleaned_contents = {}
for filename, lines in file_contents.items():
    cleaned_lines = [clean_text(line) for line in lines]
    cleaned_contents[filename] = cleaned_lines

# Display the cleaned first few lines of each file
cleaned_contents
