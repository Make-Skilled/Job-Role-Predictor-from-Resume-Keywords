from flask import Flask, render_template, request, redirect, url_for, flash, session
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import joblib
import os
from utils import preprocess_text, extract_skills, get_skill_categories
from collections import defaultdict
import json
from datetime import datetime
import PyPDF2
import docx
import io

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Required for flash messages
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Custom template filter for datetime formatting
@app.template_filter('datetime')
def format_datetime(value):
    try:
        dt = datetime.fromisoformat(value)
        return dt.strftime('%B %d, %Y %H:%M')
    except:
        return value

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

# Load the trained model
model = joblib.load('model/job_role_model.pkl')
vectorizer = joblib.load('model/vectorizer.pkl')

# Initialize prediction history storage
PREDICTION_HISTORY_FILE = 'data/prediction_history.json'

def extract_text_from_file(file, file_extension):
    try:
        # Reset file pointer to beginning
        file.seek(0)
        
        if file_extension == 'pdf':
            # Handle PDF files
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text.strip()
            
        elif file_extension in ['doc', 'docx']:
            # Handle Word documents
            doc = docx.Document(file)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text.strip()
            
        else:  # txt files
            # Handle text files
            text = file.read().decode('utf-8')
            return text.strip()
            
    except Exception as e:
        raise Exception(f"Error reading file: {str(e)}")

def load_prediction_history():
    if os.path.exists(PREDICTION_HISTORY_FILE):
        with open(PREDICTION_HISTORY_FILE, 'r') as f:
            return json.load(f)
    return {'predictions': [], 'popular_roles': defaultdict(int)}

def save_prediction_history(history):
    os.makedirs(os.path.dirname(PREDICTION_HISTORY_FILE), exist_ok=True)
    with open(PREDICTION_HISTORY_FILE, 'w') as f:
        json.dump(history, f)

def get_recommendations(predicted_role, skills, skill_categories):
    # Add role-specific recommendations based on predicted role and skills
    recommendations = []
    
    # Add skill-based recommendations
    if skill_categories.get('programming'):
        recommendations.append(f"Consider highlighting your programming skills in {', '.join(skill_categories['programming'][:3])}")
    if skill_categories.get('web'):
        recommendations.append(f"Emphasize your web development experience with {', '.join(skill_categories['web'][:3])}")
    if skill_categories.get('database'):
        recommendations.append(f"Showcase your database expertise in {', '.join(skill_categories['database'][:3])}")
    
    # Add role-specific recommendations
    role_recommendations = {
        'Software Developer': [
            "Focus on your software development projects and contributions",
            "Highlight your experience with version control and collaboration tools",
            "Emphasize your problem-solving abilities and code quality"
        ],
        'Data Scientist': [
            "Showcase your data analysis and visualization projects",
            "Highlight your experience with statistical analysis and machine learning",
            "Emphasize your ability to derive insights from data"
        ],
        'Full Stack Developer': [
            "Demonstrate your end-to-end development capabilities",
            "Highlight your experience with both frontend and backend technologies",
            "Showcase your ability to build complete web applications"
        ],
        'DevOps Engineer': [
            "Emphasize your experience with CI/CD pipelines",
            "Highlight your infrastructure automation skills",
            "Showcase your knowledge of cloud platforms and containerization"
        ],
        'Frontend Developer': [
            "Showcase your UI/UX design skills",
            "Highlight your experience with modern frontend frameworks",
            "Emphasize your attention to detail and user experience"
        ],
        'Backend Developer': [
            "Focus on your API design and implementation experience",
            "Highlight your database optimization skills",
            "Emphasize your understanding of system architecture"
        ],
        'Machine Learning Engineer': [
            "Showcase your ML model development experience",
            "Highlight your experience with deep learning frameworks",
            "Emphasize your ability to deploy ML models in production"
        ],
        'Cloud Architect': [
            "Demonstrate your cloud infrastructure design experience",
            "Highlight your knowledge of cloud security best practices",
            "Showcase your experience with multi-cloud environments"
        ],
        'Mobile Developer': [
            "Showcase your mobile app development experience",
            "Highlight your knowledge of mobile UI/UX principles",
            "Emphasize your experience with mobile app performance optimization"
        ],
        'Database Administrator': [
            "Focus on your database optimization experience",
            "Highlight your knowledge of database security",
            "Emphasize your experience with database scaling"
        ],
        'Security Engineer': [
            "Showcase your security assessment experience",
            "Highlight your knowledge of security protocols",
            "Emphasize your experience with security tools and practices"
        ],
        'Product Manager': [
            "Focus on your product strategy experience",
            "Highlight your user research and market analysis skills",
            "Emphasize your experience with agile methodologies"
        ],
        'UI/UX Designer': [
            "Showcase your design portfolio",
            "Highlight your user research experience",
            "Emphasize your knowledge of design systems"
        ],
        'QA Engineer': [
            "Focus on your testing methodology experience",
            "Highlight your automated testing skills",
            "Emphasize your attention to detail"
        ],
        'Systems Administrator': [
            "Showcase your system administration experience",
            "Highlight your knowledge of operating systems",
            "Emphasize your troubleshooting skills"
        ],
        'Business Analyst': [
            "Focus on your requirements gathering experience",
            "Highlight your data analysis skills",
            "Emphasize your communication abilities"
        ],
        'Technical Writer': [
            "Showcase your documentation experience",
            "Highlight your technical communication skills",
            "Emphasize your attention to detail"
        ],
        'Blockchain Developer': [
            "Focus on your smart contract development experience",
            "Highlight your knowledge of blockchain platforms",
            "Emphasize your understanding of cryptography"
        ],
        'Game Developer': [
            "Showcase your game development experience",
            "Highlight your knowledge of game engines",
            "Emphasize your understanding of game design principles"
        ],
        'Embedded Systems Engineer': [
            "Focus on your embedded systems experience",
            "Highlight your knowledge of microcontrollers",
            "Emphasize your understanding of real-time systems"
        ],
        'Data Engineer': [
            "Showcase your ETL pipeline experience",
            "Highlight your big data processing skills",
            "Emphasize your data warehousing knowledge"
        ],
        'AI Researcher': [
            "Focus on your research experience",
            "Highlight your publications and contributions",
            "Emphasize your understanding of AI algorithms"
        ],
        'Network Engineer': [
            "Showcase your network infrastructure experience",
            "Highlight your knowledge of network protocols",
            "Emphasize your troubleshooting skills"
        ],
        'SRE': [
            "Focus on your system reliability experience",
            "Highlight your monitoring and alerting skills",
            "Emphasize your incident response capabilities"
        ]
    }
    
    # Add role-specific recommendations
    if predicted_role in role_recommendations:
        recommendations.extend(role_recommendations[predicted_role])
    
    return recommendations[:5]  # Return top 5 recommendations

@app.route('/')
def home():
    return render_template('landing.html')

@app.route('/predictions')
def predictions():
    history = load_prediction_history()
    recent_predictions = history['predictions'][-5:]  # Get last 5 predictions
    popular_roles = [{'name': role, 'count': count} 
                    for role, count in sorted(history['popular_roles'].items(), 
                                           key=lambda x: x[1], 
                                           reverse=True)[:5]]
    
    # Get current prediction from session
    current_prediction = session.get('current_prediction', {})
    
    # Ensure all required attributes are present
    if current_prediction:
        if 'skill_categories' not in current_prediction:
            current_prediction['skill_categories'] = {}
        if 'extracted_skills' not in current_prediction:
            current_prediction['extracted_skills'] = []
        if 'recommendations' not in current_prediction:
            current_prediction['recommendations'] = []
    
    return render_template('predictions.html',
                         prediction=current_prediction,
                         recent_predictions=recent_predictions,
                         popular_roles=popular_roles)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'resume' not in request.files:
            flash('No resume file uploaded', 'error')
            return redirect(url_for('home'))
        
        resume_file = request.files['resume']
        if resume_file.filename == '':
            flash('No file selected', 'error')
            return redirect(url_for('home'))
        
        # Check file extension
        allowed_extensions = {'pdf', 'txt', 'doc', 'docx'}
        file_extension = resume_file.filename.rsplit('.', 1)[1].lower() if '.' in resume_file.filename else ''
        
        if file_extension not in allowed_extensions:
            flash('Invalid file format. Please upload PDF, TXT, DOC, or DOCX files.', 'error')
            return redirect(url_for('home'))
        
        # Extract text from the file
        try:
            resume_text = extract_text_from_file(resume_file, file_extension)
        except Exception as e:
            flash(f'Error reading file: {str(e)}', 'error')
            return redirect(url_for('home'))
        
        if not resume_text:
            flash('The uploaded file appears to be empty or could not be read properly.', 'error')
            return redirect(url_for('home'))
        
        # Process the text
        processed_text = preprocess_text(resume_text)
        skills = extract_skills(processed_text)
        skill_categories = get_skill_categories(skills)
        
        if not skills:
            flash('No skills could be extracted from the resume. Please ensure the file contains readable text.', 'error')
            return redirect(url_for('home'))
        
        # Transform the text using the vectorizer
        text_vector = vectorizer.transform([processed_text])
        
        # Make prediction
        prediction = model.predict(text_vector)[0]
        probability = model.predict_proba(text_vector).max()
        
        # Get recommendations
        recommendations = get_recommendations(prediction, skills, skill_categories)
        
        # Prepare prediction results
        prediction_results = {
            'predicted_role': prediction,
            'confidence': float(probability),
            'extracted_skills': skills,
            'skill_categories': skill_categories,
            'recommendations': recommendations,
            'timestamp': datetime.now().isoformat()
        }
        
        # Update prediction history
        history = load_prediction_history()
        history['predictions'].append({
            'role': prediction,
            'confidence': float(probability),
            'timestamp': datetime.now().isoformat()
        })
        history['popular_roles'][prediction] += 1
        save_prediction_history(history)
        
        # Store prediction in session for the predictions page
        session['current_prediction'] = prediction_results
        
        flash('Resume analyzed successfully!', 'success')
        return redirect(url_for('predictions'))
        
    except Exception as e:
        flash(f'Error processing resume: {str(e)}', 'error')
        return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(debug=True) 