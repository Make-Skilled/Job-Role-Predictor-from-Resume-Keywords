# Job Role Predictor from Resume Keywords

This project uses Natural Language Processing (NLP) to analyze resume content and suggest suitable job roles based on the skills and experience mentioned in the resume.

## Features

- Resume text analysis using NLTK
- Skill extraction from resume content
- Job role prediction using machine learning
- Web interface for easy interaction
- Support for PDF and TXT resume formats

## Technologies Used

- Python 3.8+
- Flask (Web Framework)
- NLTK (Natural Language Processing)
- Scikit-learn (Machine Learning)
- Bootstrap 5 (Frontend)
- jQuery (Frontend Interactivity)

## Setup Instructions

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Job-Role-Predictor-from-Resume-Keywords.git
cd Job-Role-Predictor-from-Resume-Keywords
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

4. Download required NLTK data:
```python
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('averaged_perceptron_tagger')"
```

5. Train the model:
```bash
python train_model.py
```

6. Run the application:
```bash
python app.py
```

7. Open your browser and navigate to `http://localhost:5000`

## Project Structure

```
Job-Role-Predictor-from-Resume-Keywords/
├── app.py                 # Main Flask application
├── train_model.py         # Model training script
├── utils.py              # Utility functions
├── requirements.txt      # Project dependencies
├── static/              # Static files
│   ├── style.css        # CSS styles
│   └── script.js        # JavaScript code
├── templates/           # HTML templates
│   └── index.html       # Main page template
└── model/              # Trained model files
    ├── job_role_model.pkl
    └── vectorizer.pkl
```

## Usage

1. Open the web interface
2. Upload a resume (PDF or TXT format)
3. Click "Predict Job Role"
4. View the predicted job role, confidence score, and extracted skills

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.