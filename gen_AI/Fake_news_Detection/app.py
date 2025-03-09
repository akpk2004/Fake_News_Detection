#File for the execution of the project
import streamlit as st
import joblib

vectorizer = joblib.load("C:/Users/anubh/Desktop/python/gen_AI/Fake_news_Detection/vectorizer.jb")
model = joblib.load("C:/Users/anubh/Desktop/python/gen_AI/Fake_news_Detection/lr_model.jb")

st.title("Fake News Detector")
st.write("Enter a news article below to check whether it is fake or real.")

news_input = st.text_area("News Article:","")

if st.button("Check News"):
    if news_input.strip():
        transform_input = vectorizer.transform([news_input])
        prediction = model.predict(transform_input)

        if prediction[0]==1:
            st.success("The News is Real!")
        else:
            st.error("The news is Fake!")
    else:
        st.warning("Please enter any text to analyze.")