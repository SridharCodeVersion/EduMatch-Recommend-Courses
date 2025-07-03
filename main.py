import pandas as pd
from utils.text_preprocess import preprocess
from sentence_transformers import SentenceTransformer
import pickle
from sklearn.metrics.pairwise import cosine_similarity

def build_embeddings():
    df = pd.read_csv('data/edx_courses.csv')
    df['text'] = (df['title'] + ' ' + df['description']).fillna('').apply(preprocess)

    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(df['text'].tolist())

    with open('models/model.pkl', 'wb') as f:
        pickle.dump((df[['title', 'description']], embeddings), f)

def recommend(resume_text, top_n=5):
    with open('models/model.pkl', 'rb') as f:
        df, embeddings = pickle.load(f)

    model = SentenceTransformer('all-MiniLM-L6-v2')
    vec = model.encode([preprocess(resume_text)])

    scores = cosine_similarity(embeddings, vec).flatten()
    top_idxs = scores.argsort()[-top_n:][::-1]

    return df.iloc[top_idxs]
