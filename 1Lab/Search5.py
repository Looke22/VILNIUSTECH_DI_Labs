import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import string
import streamlit as st

import os
os.chdir("D:/VGTU/3 Kursas/2/DI/1Lab/")

# PARAMETERS
dataset = "nyt_data.csv"
rows_to_read = 10000

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load stop words
stop_words = set(stopwords.words('english'))

# Additional stopwords
additional_stopwords = {"new", "year", "say", "photo", "york", "year", "says", "years", "time", "times",
                        "article", "said", "one", "two", "three", "last", "photos"}
stop_words = stop_words.union(additional_stopwords)

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Function for text preprocessing
def preprocess_text(text):
    # Tokenize text
    tokens = nltk.word_tokenize(text)
    
    # Remove punctuation and digits
    tokens = [token for token in tokens if token not in string.punctuation and not token.isdigit()]
    
    # Lowercase tokens
    tokens = [token.lower() for token in tokens]
    
    # Remove stop words
    tokens = [token for token in tokens if token not in stop_words]
    
    # Lemmatization
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    # Join tokens back into text
    preprocessed_text = ' '.join(tokens)
    
    return preprocessed_text

# Read the .csv file into a dataframe
df = pd.read_csv(dataset, nrows=rows_to_read)

# Select the desired column(s)
df = df.iloc[:, [0, 1]]  # Assuming the first column is 'abstract' and the second column is 'web_url'

# Replace NaN values with empty strings in the 'abstract' column
df['abstract'] = df['abstract'].fillna('') 

# Preprocess text
df['preprocessed_text'] = df['abstract'].apply(preprocess_text)

# Create a TfidfVectorizer object
vectorizer = TfidfVectorizer()

# Fit and transform the preprocessed text into a sparse matrix
matrix = vectorizer.fit_transform(df['preprocessed_text'])

# Get the feature names (words) from the vectorizer
words = vectorizer.get_feature_names_out()

# Get the word weights from the matrix
weights = matrix.sum(axis=0).A1

# Zip the words and weights into a list of tuples
freq = list(zip(words, weights))

# Sort the list by weight in descending order
sorted_words = sorted(freq, key=lambda x: x[1], reverse=True)

# Create categories based on the most common words
categories = {}
for word, weight in sorted_words[:5]:
    categories[word] = []

# Find the rows that contain the most common words
for word in categories:
    # Get the indices of the rows that contain the word
    indices = df[df['preprocessed_text'].str.contains(word)].index
    
    # Get the abstracts and article links of the rows that contain the word
    abstracts_links = df.loc[indices, ['abstract', 'web_url']]
    
    # Add the first 5 abstracts and links to the corresponding category
    for abstract, link in abstracts_links.values[:5]:
        categories[word].append((abstract, link))

# Streamlit App
st.title("Search and Explore Articles")

# Search bar
search_query = st.text_input("Search for keywords:")

if search_query:
    # Filter the dataframe based on search query
    filtered_df = df[df['preprocessed_text'].str.contains(search_query)]
    # Limit the number of search results to 10 articles
    filtered_df = filtered_df.head(10)
    # Display search results
    if not filtered_df.empty:
        st.header("Search Results:")
        for _, row in filtered_df.iterrows():
            # Check if any word in the abstract is longer than 20 letters
            if any(len(word) > 20 for word in row['abstract'].split()):
                continue  # Skip this abstract
            st.write(row['abstract'])
            st.write(f"Read more: [{row['web_url']}]({row['web_url']})")
    else:
        st.write("No matching articles found.")
else:
# Display top 5 categories and their abstracts
    st.header("Hot topics")
    for category, abstracts_links in categories.items():
        st.subheader(category)
        for abstract, link in abstracts_links:
             # Check if any word in the abstract is longer than 20 letters
            if any(len(word) > 20 for word in abstract.split()):
                continue  # Skip this abstract
            # Limit the length of the abstract to 300 characters
            truncated_abstract = abstract[:300] + '...' if len(abstract) > 300 else abstract
            st.write(truncated_abstract)
            st.write(f"Read more: [{link}]({link})")
        st.write('---')
