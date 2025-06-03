import streamlit as st
import requests

st.set_page_config(page_title="Fake News Firewall 2.0", layout="wide")
st.markdown('<style>{}</style>'.format(open('frontend/static/style.css').read()), unsafe_allow_html=True)

st.title("Fake News Firewall 2.0")

st.header("Classify News Article")
text = st.text_area("Enter a news article", height=200)
if st.button("Classify"):
    if text:
        try:
            response = requests.post("http://localhost:8080/classify", json={"text": text})
            if response.status_code == 200:
                result = response.json()
                st.success(f"*Prediction*: {result['label'].capitalize()}")
                st.write(f"*Confidence*: {result['confidence']:.2f}%")
                st.write("*Explanation*:")
                explanation = result.get('explanation', [])
                if isinstance(explanation, list):
                    for item in explanation:
                        st.write(f"- {item['word']}: Impact {item['impact']:.4f}")
                else:
                    st.warning("No explanation available")
            else:
                st.error(f"Error classifying article: {response.status_code}")
        except Exception as e:
            st.error(f"Request failed: {str(e)}")

st.header("Provide Feedback")
feedback_label = st.selectbox("Correct label", ["credible", "misleading", "fake"])
if st.button("Submit Feedback"):
    if text and feedback_label:
        try:
            response = requests.post("http://localhost:8080/feedback", json={"text": text, "label": feedback_label})
            if response.status_code == 200:
                st.success("Feedback submitted!")
            else:
                st.error(f"Error submitting feedback: {response.status_code}")
        except Exception as e:
            st.error(f"Feedback request failed: {str(e)}")