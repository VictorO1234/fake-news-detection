import streamlit as st
import joblib
import os
import re
import nltk
from nltk.corpus import stopwords
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
from html import escape
import numpy as np

# ----------------------
# Setup
# ----------------------
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load dataset
df = pd.read_csv('data/fake_news.csv')

# Load model and vectorizer
current_dir = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(current_dir, 'models', 'fake_news_model.pkl'))
vectorizer = joblib.load(os.path.join(current_dir, 'models', 'vectorizer.pkl'))

# ----------------------
# Functions
# ----------------------
def clean_text(text):
    text = re.sub(r'http\S+', '', text)  # remove URLs
    text = re.sub(r'\W', ' ', text)      # remove punctuation
    text = text.lower()
    words = [w for w in text.split() if w not in stop_words]
    return ' '.join(words)

def get_top_words(model, vectorizer, text, n=10):
    vect_text = vectorizer.transform([text])
    feature_names = vectorizer.get_feature_names_out()
    coef = model.coef_[0]
    contributions = vect_text.toarray()[0] * coef
    
    top_pos_idx = contributions.argsort()[-n:][::-1]
    top_neg_idx = contributions.argsort()[:n]
    
    top_pos_words = [(feature_names[i], contributions[i]) for i in top_pos_idx]
    top_neg_words = [(feature_names[i], contributions[i]) for i in top_neg_idx]
    
    return top_pos_words, top_neg_words, contributions

def generate_wordcloud(text_series, title):
    text = " ".join(text_series)
    wc = WordCloud(width=400, height=200, background_color='white', stopwords=stop_words).generate(text)
    fig, ax = plt.subplots()
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(title)
    st.sidebar.pyplot(fig)
    plt.close(fig)

def highlight_text_intensity(text, top_pos, top_neg):
    """
    Highlights words in text with intensity proportional to contribution.
    Red = FAKE, Green = REAL.
    """
    pos_dict = dict(top_pos)
    neg_dict = dict(top_neg)
    
    words = text.split()
    highlighted_words = []
    
    # Normalize contribution for intensity (0.3 to 1)
    def get_color_intensity(score, max_score):
        return 0.3 + 0.7 * min(abs(score)/max_score, 1)
    
    max_pos = max([abs(val) for _, val in top_pos]) if top_pos else 1
    max_neg = max([abs(val) for _, val in top_neg]) if top_neg else 1
    
    for word in words:
        clean_word = re.sub(r'\W', '', word.lower())
        if clean_word in pos_dict:
            intensity = get_color_intensity(pos_dict[clean_word], max_pos)
            highlighted_words.append(
                f"<span style='color:rgba(255,0,0,{intensity});font-weight:bold'>{escape(word)}</span>"
            )
        elif clean_word in neg_dict:
            intensity = get_color_intensity(neg_dict[clean_word], max_neg)
            highlighted_words.append(
                f"<span style='color:rgba(0,128,0,{intensity});font-weight:bold'>{escape(word)}</span>"
            )
        else:
            highlighted_words.append(escape(word))
    
    return " ".join(highlighted_words)

# ----------------------
# Streamlit Dashboard
# ----------------------
st.title("Fake News Detection Dashboard")

# Sidebar: Dataset Statistics
st.sidebar.header("Dataset Statistics")
st.sidebar.write(f"Total articles: {len(df)}")
st.sidebar.write(f"FAKE articles: {len(df[df['label']=='FAKE'])}")
st.sidebar.write(f"REAL articles: {len(df[df['label']=='REAL'])}")

# Class Distribution Plot
st.sidebar.subheader("Class Distribution")
fig, ax = plt.subplots()
sns.countplot(x='label', data=df, palette='Set2', ax=ax)
ax.set_title("Fake vs Real Articles")
st.sidebar.pyplot(fig)
plt.close(fig)

# Word Clouds
st.sidebar.subheader("Word Clouds")
generate_wordcloud(df[df['label']=='FAKE']['text'], "FAKE News Word Cloud")
generate_wordcloud(df[df['label']=='REAL']['text'], "REAL News Word Cloud")

# User Input
st.subheader("Enter news text to predict:")
user_input = st.text_area("News Text", height=200)

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text to predict.")
    else:
        cleaned = clean_text(user_input)
        vect_input = vectorizer.transform([cleaned])
        
        # Predict label
        prediction = model.predict(vect_input)[0]
        
        # Confidence score
        probability = model.predict_proba(vect_input).max()
        confidence = round(probability * 100, 2)
        
        st.success(f"Prediction: **{prediction}**")
        st.info(f"Confidence: **{confidence}%**")
        
        # Top contributing words
        top_pos, top_neg, contributions = get_top_words(model, vectorizer, cleaned, n=10)
        
        st.subheader("Top Contributing Words")
        st.write("**Words pushing prediction toward FAKE:**")
        st.write(", ".join([f"{w} ({round(score,4)})" for w, score in top_pos]))
        st.write("**Words pushing prediction toward REAL:**")
        st.write(", ".join([f"{w} ({round(score,4)})" for w, score in top_neg]))
        
        # Highlight words in text with intensity
        st.subheader("Input Text Highlighted by Contribution")
        highlighted_html = highlight_text_intensity(user_input, top_pos, top_neg)
        st.markdown(highlighted_html, unsafe_allow_html=True)
        
        # Show bar chart of total FAKE vs REAL contributions
        st.subheader("Overall Contribution Scores")
        total_fake = sum([score for _, score in top_pos])
        total_real = -sum([score for _, score in top_neg])  # make positive for chart
        fig2, ax2 = plt.subplots()
        ax2.bar(['FAKE', 'REAL'], [total_fake, total_real], color=['red', 'green'])
        ax2.set_ylabel("Sum of Contribution Scores")
        ax2.set_title("Total Contribution Toward Prediction")
        st.pyplot(fig2)
        plt.close(fig2)
