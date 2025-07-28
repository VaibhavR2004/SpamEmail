import streamlit as st
import pickle
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model and vectorizer
model = load_model('tfidf_ann_model.h5')

with open('tfidf_vectorizer.pkl', 'rb') as file:
    tfidf_vectorizer = pickle.load(file)

# Session state to control flow
if 'show_result' not in st.session_state:
    st.session_state.show_result = False

st.set_page_config(page_title="Spam Email Classifier", page_icon="📧")
st.title("📧 Spam Email Classifier")
st.write("Enter an email message below to classify it as **Spam** or **Ham (Not Spam)**.")

# If prediction not yet done
if not st.session_state.show_result:
    # Text input
    email_input = st.text_area("✉️ Enter email content here:", key="email_input")

    # Predict
    if st.button("🔍 Predict"):
        if not email_input.strip():
            st.warning("Please enter some text.")
        else:
            # Transform and predict
            transformed = tfidf_vectorizer.transform([email_input]).toarray()
            prediction = model.predict(transformed)[0][0]
            label = 1 if prediction >= 0.5 else 0
            result = "🚫 Spam" if label == 1 else "✅ Ham (Not Spam)"
            st.session_state.prediction_result = result
            st.session_state.show_result = True
            st.experimental_rerun()

# Show prediction and reset button
else:
    st.subheader(f"**Prediction:** {st.session_state.prediction_result}")
    if st.session_state.prediction_result.startswith("🚫"):
        st.error("This message is Spam.")
    else:
        st.success("This message is Not Spam.")

    # Reset button
    if st.button("🔁 Check Another Email"):
        st.session_state.show_result = False
        st.session_state.email_input = ""  # reset input
        st.experimental_rerun()
