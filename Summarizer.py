# app.py

import streamlit as st
import nltk
import numpy as np
import networkx as nx
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import pipeline


# Download NLTK data (only need to run once)
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Set up the Streamlit app
st.title('📝 Text Summarization Tool')
st.write('This application summarizes long pieces of text using Extractive or Abstractive methods.')

# Sidebar options
st.sidebar.title('Summarization Options')
summarization_type = st.sidebar.selectbox('Select Summarization Type', ('Extractive', 'Abstractive'))

if summarization_type == 'Extractive':
    num_sentences = st.sidebar.slider('Number of sentences in summary', min_value=1, max_value=10, value=3)
else:
    max_length = st.sidebar.slider('Maximum summary length', min_value=50, max_value=500, value=130)
    min_length = st.sidebar.slider('Minimum summary length', min_value=10, max_value=100, value=30)

# Text input
text_input = st.text_area('Enter the text you want to summarize', height=300)

# Preprocess text function
def preprocess_text(text):
    # Sentence Tokenization
    sentences = sent_tokenize(text)
    # Word Tokenization and Cleaning
    processed_sentences = []
    for sentence in sentences:
        words = word_tokenize(sentence)
        words = [word.lower() for word in words if word.isalpha()]
        # Remove stopwords
        words = [word for word in words if word not in stopwords.words('english')]
        processed_sentences.append(' '.join(words))
    return sentences, processed_sentences

# Extractive summarization function
from networkx.convert_matrix import from_scipy_sparse_array

def extractive_summarization(text, num_sentences=3):
    sentences, processed_sentences = preprocess_text(text)

    # Check if the number of sentences is less than desired summary length
    if len(sentences) <= num_sentences:
        return text

    # Create TF-IDF matrix
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(processed_sentences)

    # Compute similarity matrix
    similarity_matrix = tfidf_matrix * tfidf_matrix.T

    # Build graph and rank sentences
    nx_graph = from_scipy_sparse_array(similarity_matrix)
    scores = nx.pagerank(nx_graph)

    # Rank sentences
    ranked_sentences = sorted(
        ((scores[i], s) for i, s in enumerate(sentences)), reverse=True
    )

    # Select top N sentences
    summary_sentences = [sentence for _, sentence in ranked_sentences[:num_sentences]]
    summary = ' '.join(summary_sentences)
    return summary


# Abstractive summarization function
@st.cache_resource  # Cache the model for faster loading
def load_summarizer():
    return pipeline('summarization', model='facebook/bart-large-cnn')

def abstractive_summarization(text, max_length=130, min_length=30):
    summarizer = load_summarizer()
    # Ensure max_length is greater than min_length
    if max_length <= min_length:
        st.error('Maximum length must be greater than minimum length.')
        return ''
    # Summarize text
    summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
    return summary[0]['summary_text']

# Summarize button
if st.button('🔍 Summarize'):
    if text_input.strip() == '':
        st.error('Please enter text to summarize.')
    else:
        with st.spinner('Summarizing...'):
            if summarization_type == 'Extractive':
                summary = extractive_summarization(text_input, num_sentences=num_sentences)
            else:
                summary = abstractive_summarization(text_input, max_length=max_length, min_length=min_length)
        st.subheader('Summary')
        st.write(summary)