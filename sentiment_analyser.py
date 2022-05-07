
#streamlit run "C:\Users\yeosi\Documents\Python MAGES\07 DS106\Sentiment Analysis\sentiment_analyser.py"
# Run the relevant libraries
import streamlit as st
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch 
import re
import pandas as pd 
import numpy as np 


# Create user interface on Streamlit
st.title('Sentiment Analyser App')
st.subheader('Welcome to my sentiment analysis app!')
st.subheader('This app will analyse the sentiment of the text you enter.')
st.write('This app uses the Hugging Face Transformers [sentiment analyser](https://huggingface.co/course/chapter1/3?fw=tf) library to classify the sentiment of your input as postive or negative. The web app is built using [Streamlit](https://docs.streamlit.io/en/stable/getting_started.html).')
st.write('To see my source code, have a look at my [GitHub repo](https://github.com/SiewPingYeo/Sentiment-Analyzer-Streamlit-and-HuggingFace-).')

# Creating the form where user inputs the text
form = st.form(key='sentiment-form')
user_input = form.text_area('Enter your text')
submit = form.form_submit_button('Submit')


# Use the pre-trained model BERT-Base-Uncased to classify the sentiment of the text entered by the user
tokenizer = AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
model = AutoModelForSequenceClassification.from_pretrained ('nlptown/bert-base-multilingual-uncased-sentiment')


# Define a function to classify the sentiment of the text entered by the user
def sentiment_score(review):
        tokens = tokenizer.encode (review, return_tensors = 'pt')
        result = model(tokens)
        result = int(torch.argmax(result.logits))+1
    
        if result == 1 or result == 2:
            return 'Negative'
        elif result == 3:
            return 'Neutral'
        else:
            return 'Positive'

# Run the function on the text entered by the user
if submit:
   
    variable = sentiment_score(user_input)
    st.write(f'The sentiment of your text is {variable}')





