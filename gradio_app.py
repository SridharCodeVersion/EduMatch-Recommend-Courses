import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import gradio as gr

# Load and preprocess dataset
df = pd.read_csv("edx_courses.csv")
df['Full_Description'] = df['About'] + " " + df['Course Description']

# TF-IDF vectorizer
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['Full_Description'])

# Recommender function
def recommend_courses(input_text, top_n):
    if input_text.strip() == "":
        return "❌ Please enter some skills/domains to search for courses."

    query_vec = tfidf.transform([input_text])
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    indices = similarities.argsort()[-top_n:][::-1]

    output = f"## 🔍 Recommendations for: `{input_text}`\n\n"
    for i in indices:
        course = df.iloc[i]
        output += f"""
### 🎓 [{course['Name']}]({course['Link']})
**🏫 University:** {course['University']}
**📈 Difficulty:** {course['Difficulty Level']}

📘 **Summary:**
{course['Full_Description'][:500]}...

---
"""
    return output

# Gradio interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("## 📘 **EduMatch: AI Course Recommender**")
    gr.Markdown("Fill your resume gaps by finding the best **edX courses** based on your input skills and domains.")

    with gr.Row():
        input_text = gr.Textbox(placeholder="E.g. python, AI, data science", label="🧠 Enter your skill/domain gaps")
        top_n = gr.Slider(3, 10, value=5, step=1, label="📌 No. of courses")

    with gr.Row():
        submit_btn = gr.Button("🚀 Recommend Courses")

    output_md = gr.Markdown()

    submit_btn.click(fn=recommend_courses, inputs=[input_text, top_n], outputs=output_md)

demo.launch(share=True)
