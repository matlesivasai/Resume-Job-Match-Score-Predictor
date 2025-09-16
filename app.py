"""
Resume Job Match Score Predictor
Flask API for calculating job fit scores using NLP and ML
"""

import os
import json
import re
from io import BytesIO
from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired, FileAllowed
from wtforms import TextAreaField, SubmitField
from wtforms.validators import DataRequired
from werkzeug.utils import secure_filename
import numpy as np
import pandas as pd

# NLP and ML imports
import spacy
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
import fitz  # PyMuPDF
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-change-in-production'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create upload directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variables for models
nlp = None
bert_tokenizer = None
bert_model = None

def initialize_models():
    """Initialize NLP models on first request"""
    global nlp, bert_tokenizer, bert_model

    if nlp is None:
        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("spaCy model not found. Please install it with: python -m spacy download en_core_web_sm")
            nlp = None

    if bert_tokenizer is None or bert_model is None:
        try:
            bert_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
            bert_model = AutoModel.from_pretrained('bert-base-uncased')
        except Exception as e:
            print(f"Error loading BERT model: {e}")
            bert_tokenizer = None
            bert_model = None

class UploadForm(FlaskForm):
    """Form for file upload and job description input"""
    resume_file = FileField('Resume (PDF)', validators=[
        FileRequired(),
        FileAllowed(['pdf'], 'Only PDF files are allowed!')
    ])
    job_description = TextAreaField('Job Description', validators=[
        DataRequired()
    ], render_kw={"placeholder": "Enter the job description here..."})
    submit = SubmitField('Calculate Match Score')

def extract_text_from_pdf(file_content):
    """Extract text from PDF using both PyPDF2 and PyMuPDF as fallback"""
    text = ""

    try:
        # Method 1: PyPDF2
        pdf_reader = PyPDF2.PdfReader(BytesIO(file_content))
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"

        if not text.strip():
            # Method 2: PyMuPDF as fallback
            pdf_document = fitz.open(stream=file_content, filetype="pdf")
            for page_num in range(pdf_document.page_count):
                page = pdf_document[page_num]
                text += page.get_text() + "\n"
            pdf_document.close()

    except Exception as e:
        print(f"Error extracting PDF text: {e}")
        return ""

    return text

def preprocess_text(text):
    """Clean and preprocess text for NLP analysis"""
    if not text:
        return ""

    # Remove extra whitespace and newlines
    text = re.sub(r'\s+', ' ', text)

    # Remove special characters but keep alphanumeric and spaces
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)

    # Convert to lowercase
    text = text.lower()

    # Remove extra spaces
    text = ' '.join(text.split())

    return text

def extract_entities_with_spacy(text):
    """Extract named entities using spaCy NER"""
    if not nlp or not text:
        return {}

    doc = nlp(text)

    entities = {
        'PERSON': [],
        'ORG': [],
        'SKILL': [],
        'EDUCATION': [],
        'EXPERIENCE': []
    }

    for ent in doc.ents:
        if ent.label_ == 'PERSON':
            entities['PERSON'].append(ent.text)
        elif ent.label_ == 'ORG':
            entities['ORG'].append(ent.text)

    # Extract skills using keyword matching
    skill_keywords = [
        'python', 'java', 'javascript', 'sql', 'html', 'css', 'react', 'angular',
        'node.js', 'django', 'flask', 'aws', 'azure', 'docker', 'kubernetes',
        'machine learning', 'data analysis', 'project management', 'agile'
    ]

    text_lower = text.lower()
    for skill in skill_keywords:
        if skill in text_lower:
            entities['SKILL'].append(skill)

    return entities

def get_bert_embeddings(text):
    """Generate BERT embeddings for text"""
    if not bert_tokenizer or not bert_model or not text:
        return None

    try:
        # Tokenize and encode
        inputs = bert_tokenizer(text, return_tensors="pt", 
                              max_length=512, truncation=True, padding=True)

        # Get embeddings
        with torch.no_grad():
            outputs = bert_model(**inputs)
            # Use CLS token embedding as sentence representation
            embeddings = outputs.last_hidden_state[:, 0, :].numpy()

        return embeddings[0]  # Return as 1D array

    except Exception as e:
        print(f"Error generating BERT embeddings: {e}")
        return None

def calculate_tfidf_similarity(resume_text, job_description):
    """Calculate TF-IDF based cosine similarity"""
    try:
        # Create TF-IDF vectorizer
        vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )

        # Fit and transform both texts
        tfidf_matrix = vectorizer.fit_transform([resume_text, job_description])

        # Calculate cosine similarity
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])

        return similarity[0][0]

    except Exception as e:
        print(f"Error calculating TF-IDF similarity: {e}")
        return 0.0

def calculate_bert_similarity(resume_text, job_description):
    """Calculate BERT-based cosine similarity"""
    try:
        resume_embedding = get_bert_embeddings(resume_text)
        job_embedding = get_bert_embeddings(job_description)

        if resume_embedding is None or job_embedding is None:
            return 0.0

        # Calculate cosine similarity
        similarity = cosine_similarity([resume_embedding], [job_embedding])
        return similarity[0][0]

    except Exception as e:
        print(f"Error calculating BERT similarity: {e}")
        return 0.0

def calculate_job_fit_score(resume_text, job_description):
    """Calculate comprehensive job fit score"""

    # Preprocess texts
    resume_clean = preprocess_text(resume_text)
    job_clean = preprocess_text(job_description)

    # Extract entities
    resume_entities = extract_entities_with_spacy(resume_text)
    job_entities = extract_entities_with_spacy(job_description)

    # Calculate different similarity scores
    tfidf_score = calculate_tfidf_similarity(resume_clean, job_clean)
    bert_score = calculate_bert_similarity(resume_clean, job_clean)

    # Calculate skill match score
    resume_skills = set([skill.lower() for skill in resume_entities['SKILL']])
    job_skills = set([skill.lower() for skill in job_entities['SKILL']])

    if job_skills:
        skill_match = len(resume_skills.intersection(job_skills)) / len(job_skills)
    else:
        skill_match = 0.0

    # Weighted final score
    final_score = (
        0.4 * bert_score +
        0.3 * tfidf_score +
        0.3 * skill_match
    )

    # Convert to percentage
    final_score_pct = min(100, max(0, final_score * 100))

    return {
        'overall_score': round(final_score_pct, 2),
        'bert_similarity': round(bert_score * 100, 2),
        'tfidf_similarity': round(tfidf_score * 100, 2),
        'skill_match': round(skill_match * 100, 2),
        'resume_entities': resume_entities,
        'job_entities': job_entities,
        'matched_skills': list(resume_skills.intersection(job_skills)),
        'missing_skills': list(job_skills - resume_skills)
    }

@app.route('/', methods=['GET', 'POST'])
def index():
    """Main route for file upload and processing"""
    form = UploadForm()

    if form.validate_on_submit():
        try:
            # Initialize models if not done already
            initialize_models()

            # Get uploaded file
            resume_file = form.resume_file.data
            job_description = form.job_description.data

            # Extract text from PDF
            file_content = resume_file.read()
            resume_text = extract_text_from_pdf(file_content)

            if not resume_text:
                flash('Could not extract text from the PDF file. Please try another file.', 'error')
                return redirect(url_for('index'))

            # Calculate job fit score
            results = calculate_job_fit_score(resume_text, job_description)

            return render_template('results.html', results=results, 
                                 resume_text=resume_text[:500] + "..." if len(resume_text) > 500 else resume_text)

        except Exception as e:
            flash(f'An error occurred while processing your request: {str(e)}', 'error')
            return redirect(url_for('index'))

    return render_template('index.html', form=form)

@app.route('/api/match', methods=['POST'])
def api_match():
    """API endpoint for job matching"""
    try:
        initialize_models()

        # Check if file is present
        if 'resume' not in request.files:
            return jsonify({'error': 'No resume file provided'}), 400

        resume_file = request.files['resume']
        job_description = request.form.get('job_description', '')

        if not job_description:
            return jsonify({'error': 'No job description provided'}), 400

        # Extract text from PDF
        file_content = resume_file.read()
        resume_text = extract_text_from_pdf(file_content)

        if not resume_text:
            return jsonify({'error': 'Could not extract text from PDF'}), 400

        # Calculate job fit score
        results = calculate_job_fit_score(resume_text, job_description)

        return jsonify({
            'success': True,
            'data': results
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
