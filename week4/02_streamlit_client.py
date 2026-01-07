import streamlit as st
import requests

st.set_page_config(page_title="GenAI Agent Client")
st.title("Email / Task Automation Agent")

user_text = st.text_area("Enter text:", height=200)

if st.button("Process"):
    if not user_text.strip():
        st.warning("Please enter some text.")
    else:
        response = requests.post(
            "http://127.0.0.1:8000/process",
            json={"text":user_text}
        )

        if response.status_code == 200:
            data = response.json()

            st.subheader("Type")
            st.write(data["type"])

            st.subheader("Intent")
            st.write(data["intent"])

            st.subheader("Output")
            st.write(data["output"])

        else:
            st.error("API request failed.")