#!/usr/bin/env python
# coding: utf-8

# In[4]:

import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import PyPDF2
from docx import Document
import re
from nltk.tokenize import RegexpTokenizer
import nltk  
from nltk.corpus import stopwords  
import pandas as pd

# Load the TF-IDF vectorizer and the classifier model
tfidf_vector = TfidfVectorizer(sublinear_tf=True, stop_words='english')
clf = pickle.load(open('model_knn.pickle', 'rb'))
tfidf = pickle.load(open('tfidf_vector.pickle', 'rb'))

import spacy

nlp = spacy.load("en_core_web_sm")
def clean_resume(sentence):
    sentence = str(sentence)
    sentence = sentence.lower()
    sentence = sentence.replace('{html}',"") 
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', sentence)
    rem_url = re.sub(r'http\S+', '',cleantext)
    rem_num = re.sub('[0-9]+', '', rem_url)
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(rem_num)  
    # Filter out stopwords
    filtered_words = [w for w in tokens if len(w) > 2 if not w in stopwords.words('english')]
    return " ".join(filtered_words)

# Define the main function
def main():
    st.title("Resume screening app")
    st.write("Upload your resume to see the predicted category.")

    # Upload the resume file
    uploaded_file = st.file_uploader("Upload Resume", type=["doc", "pdf"])
    
    if uploaded_file is not None:
        try:
            # Read the resume file and decode it
            resume_bytes = uploaded_file.read()
            resume_text = resume_bytes.decode('utf-8')
        except UnicodeDecodeError:
            # If decoding fails, try decoding using Latin-1 encoding
            resume_text = resume_bytes.decode('latin-1')

        # Clean the resume text (you need to define clean_resume() function)
        cleaned_resume = clean_resume(resume_text)

        # Transform the cleaned resume text using tfidf vectorizer
        input_features = tfidf.transform([cleaned_resume])

        # Make prediction using the classifier
        prediction_id = clf.predict(input_features)[0]
        category_name = category_mapping.get(prediction_id,"Unknown")
        st.write("Predicted category:", prediction_id)
        st.write("Predicted category name:", category_name)
        

category_mapping = {
    0: "PeopleSoft",
    1: "SQLdeveloper",
    2: "jsdeveloper",
    3: "workday"
}

# Call the main function
if __name__ == "__main__":
    main()

# In[ ]:




