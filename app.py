import streamlit as st
import pickle

OPTIMAL_THRESHOLD = 0.5301

model_path = 'Logistic_regression_final.pkl'
tfidf_vectorizer_path = 'best_tfidf_vectorizer.pkl'

with open(model_path, 'rb') as model_file:
    model = pickle.load(model_file)

with open(tfidf_vectorizer_path, 'rb') as tfidf_file:
    tfidf_vectorizer = pickle.load(tfidf_file)

st.title("🎮 Steam Reviews Sentiment Analysis")
st.markdown("### Predict Positive or Negative Sentiment")

# Add space before the input area
st.markdown("<br>", unsafe_allow_html=True)

user_input = st.text_area(
    "Enter your review:",
    height=150,
    max_chars=2000,
    placeholder="Type your review here..."
)

# Add space after the input
st.markdown("<br>", unsafe_allow_html=True)

if st.button("Predict Sentiment"):
    if user_input:
        user_input = user_input.strip()
        user_input_vectorized = tfidf_vectorizer.transform([user_input])

        # Get probabilities for both classes
        prob_negative = model.predict_proba(user_input_vectorized)[0][0]
        prob_positive = model.predict_proba(user_input_vectorized)[0][1]

        # Round to percentage
        positive_conf = round(prob_positive * 100, 2)
        negative_conf = round(prob_negative * 100, 2)

        if prob_positive >= OPTIMAL_THRESHOLD:
            st.success(f"Prediction: **Positive Sentiment** 😊")
        else:
            st.error(f"Prediction: **Negative Sentiment** ☹️")

        st.markdown(f"**Confidence Scores:**")
        st.markdown(f"- Positive Sentiment: `{positive_conf}%`")
        st.markdown(f"- Negative Sentiment: `{negative_conf}%`")

    else:
        st.warning("Please enter a review before predicting.")
