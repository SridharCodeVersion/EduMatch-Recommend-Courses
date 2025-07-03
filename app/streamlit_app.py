import streamlit as st
from main import recommend

st.title("🎓 EduMatch: Course Recommender")

resume = st.text_area("Paste your resume or skills:")

if st.button("Recommend Courses"):
    if resume.strip() == "":
        st.warning("Please enter some resume content.")
    else:
        results = recommend(resume)
        st.subheader("📚 Recommended Courses:")
        for _, row in results.iterrows():
            st.markdown(f"**{row['title']}**  \n{row['description']}\n---")
