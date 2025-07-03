# EduMatch – Recommend Courses Based on Resume

Paste your resume, and get top courses to fill your skill gaps!

## 🚀 How to Run

```bash
pip install -r requirements.txt
python -m nltk.downloader punkt stopwords
python -m spacy download en_core_web_sm

python main.py          # First build model
streamlit run app/streamlit_app.py  # Run app
```
