import re
import fitz
import easyocr
import docx
import csv
import os
from PIL import Image
from flask import Flask, request, render_template
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
reader = easyocr.Reader(['en'], gpu=False)

LOG_FILE = "score_log.csv"

def clean_text(text):
    return re.sub(r'\W+', ' ', text.lower())

def extract_text_from_pdf(file_storage):
    try:
        doc = fitz.open(stream=file_storage.read(), filetype="pdf")
        text = ""
        for page in doc:
            page_text = page.get_text()
            if not page_text.strip():
                pix = page.get_pixmap()
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                ocr_result = reader.readtext(img, detail=0)
                page_text = " ".join(ocr_result)
            text += page_text + "\n"
        return text
    except Exception as e:
        print(f"PDF extraction failed: {e}")
        return ""

def extract_text_from_docx(file_storage):
    try:
        doc = docx.Document(file_storage)
        return "\n".join([para.text for para in doc.paragraphs])
    except Exception as e:
        print(f"DOCX extraction failed: {e}")
        return ""

def calculate_score(resume, jd):
    resume_clean = clean_text(resume)
    jd_clean = clean_text(jd)
    vectorizer = CountVectorizer().fit_transform([resume_clean, jd_clean])
    vectors = vectorizer.toarray()
    score = cosine_similarity([vectors[0]], [vectors[1]])[0][0]
    return round(score * 100, 2)

def keyword_match(resume, jd):
    resume_words = set(clean_text(resume).split())
    jd_words = set(clean_text(jd).split())
    matched = resume_words & jd_words
    missing = jd_words - resume_words
    return matched, missing

def log_score(resume, jd, score):
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Resume Snippet", "JD Snippet", "Score"])
    with open(LOG_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([resume[:100], jd[:100], score])

@app.route('/', methods=['GET', 'POST'])
def index():
    score = None
    matched_keywords = []
    missing_keywords = []
    if request.method == 'POST':
        resume_text = request.form.get('resume', '')
        jd_text = request.form.get('jd', '')

        resume_file = request.files.get('resume_file')
        jd_file = request.files.get('jd_file')

        if resume_file:
            if resume_file.filename.endswith('.pdf'):
                resume_text = extract_text_from_pdf(resume_file)
            elif resume_file.filename.endswith('.docx'):
                resume_text = extract_text_from_docx(resume_file)

        if jd_file:
            if jd_file.filename.endswith('.pdf'):
                jd_text = extract_text_from_pdf(jd_file)
            elif jd_file.filename.endswith('.docx'):
                jd_text = extract_text_from_docx(jd_file)

        score = calculate_score(resume_text, jd_text)
        matched_keywords, missing_keywords = keyword_match(resume_text, jd_text)
        log_score(resume_text, jd_text, score)

    return render_template('index.html', score=score,
                           matched=matched_keywords,
                           missing=missing_keywords)

if __name__ == '__main__':
    app.run(debug=True)