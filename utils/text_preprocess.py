import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import gradio as gr

df = pd.read_csv("edx_courses.csv")
df['Full_Description'] = df['About'] + " " + df['Course Description']

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['Full_Description'])


def recommend_courses(input_text, top_n):
    query_vec = tfidf.transform([input_text])
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    indices = similarities.argsort()[-top_n:][::-1]
    results = []
    for i in indices:
        row = df.iloc[i]
        result = f"🎓 **{row['Name']}** ({row['University']})\\n"
        result += f"📈 Level: {row['Difficulty Level']}\\n"
        result += f"🔗 [Course Link]({row['Link']})\\n\\n"
        result += f"{row['Full_Description'][:500]}...\\n"
        results.append(result)
    return "\\n---\\n".join(results)

