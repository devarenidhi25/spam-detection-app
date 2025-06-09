import streamlit as st
import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))


# Preprocessing function
def preprocess(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text


# Page config
st.set_page_config(
    page_title="SpamShield - Spam Detector",
    page_icon="📧",
    layout="wide"
)


st.title("📧 Spam Detector App")
st.markdown("Upload a CSV file with SMS data and test spam detection.")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, encoding='latin-1')
    df = df[['v1', 'v2']]
    df.columns = ['label', 'message']

    st.subheader("📋 Raw Data")
    st.dataframe(df.head())

    # Clean data
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    df['message'] = df['message'].apply(preprocess)

    # Vectorize and train
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(df['message'])
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = MultinomialNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    st.success(f"✅ Model trained with accuracy: {acc:.2f}")

    st.subheader("🔍 Try it out")
    user_input = st.text_area("Enter a message to check if it's spam")
    if st.button("Predict"):
        processed = preprocess(user_input)
        vect = vectorizer.transform([processed])
        prediction = model.predict(vect)[0]
        st.write(f"📨 The message is likely: **{prediction.upper()}**")
else:
    st.info("Please upload a valid CSV file to proceed.")
