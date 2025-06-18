import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import defaultdict
import re

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Define comprehensive skill dictionaries
SKILLS = {
    'programming_languages': {
        'python': ['python', 'py', 'pandas', 'numpy', 'scipy'],
        'java': ['java', 'spring', 'hibernate', 'j2ee'],
        'javascript': ['javascript', 'js', 'node', 'react', 'angular', 'vue'],
        'c++': ['c++', 'cpp', 'stl'],
        'c#': ['c#', 'csharp', '.net'],
        'ruby': ['ruby', 'rails'],
        'php': ['php', 'laravel'],
        'swift': ['swift', 'ios'],
        'kotlin': ['kotlin', 'android'],
        'go': ['go', 'golang'],
        'rust': ['rust'],
        'scala': ['scala', 'spark'],
        'r': ['r', 'rstudio'],
        'matlab': ['matlab'],
        'typescript': ['typescript', 'ts']
    },
    'web_technologies': {
        'html': ['html', 'html5'],
        'css': ['css', 'css3', 'sass', 'less'],
        'react': ['react', 'reactjs', 'redux'],
        'angular': ['angular', 'angularjs'],
        'vue': ['vue', 'vuejs'],
        'django': ['django'],
        'flask': ['flask'],
        'express': ['express', 'expressjs'],
        'spring': ['spring', 'springboot'],
        'laravel': ['laravel'],
        'rails': ['rails', 'ruby on rails'],
        'asp.net': ['asp.net', 'aspnet'],
        'graphql': ['graphql'],
        'rest': ['rest', 'restful', 'api'],
        'websocket': ['websocket', 'socket.io']
    },
    'databases': {
        'mysql': ['mysql'],
        'postgresql': ['postgresql', 'postgres'],
        'mongodb': ['mongodb', 'mongo'],
        'oracle': ['oracle'],
        'sql server': ['sql server', 'mssql'],
        'redis': ['redis'],
        'cassandra': ['cassandra'],
        'elasticsearch': ['elasticsearch', 'elastic'],
        'dynamodb': ['dynamodb'],
        'neo4j': ['neo4j'],
        'sqlite': ['sqlite']
    },
    'cloud_devops': {
        'aws': ['aws', 'amazon web services', 'ec2', 's3', 'lambda'],
        'azure': ['azure', 'microsoft azure'],
        'gcp': ['gcp', 'google cloud', 'google cloud platform'],
        'docker': ['docker', 'container'],
        'kubernetes': ['kubernetes', 'k8s'],
        'terraform': ['terraform'],
        'ansible': ['ansible'],
        'jenkins': ['jenkins'],
        'git': ['git', 'github', 'gitlab'],
        'ci/cd': ['ci/cd', 'continuous integration', 'continuous deployment'],
        'linux': ['linux', 'unix'],
        'nginx': ['nginx'],
        'apache': ['apache']
    },
    'data_science_ml': {
        'machine learning': ['machine learning', 'ml', 'tensorflow', 'pytorch', 'scikit-learn'],
        'deep learning': ['deep learning', 'neural networks', 'cnn', 'rnn'],
        'data analysis': ['data analysis', 'data analytics', 'pandas', 'numpy'],
        'data visualization': ['data visualization', 'matplotlib', 'seaborn', 'plotly'],
        'nlp': ['nlp', 'natural language processing', 'text mining'],
        'computer vision': ['computer vision', 'opencv', 'image processing'],
        'big data': ['big data', 'hadoop', 'spark', 'hive'],
        'statistics': ['statistics', 'statistical analysis', 'r'],
        'reinforcement learning': ['reinforcement learning', 'rl'],
        'time series': ['time series', 'forecasting']
    },
    'mobile_dev': {
        'ios': ['ios', 'swift', 'objective-c'],
        'android': ['android', 'kotlin', 'java'],
        'react native': ['react native'],
        'flutter': ['flutter', 'dart'],
        'xamarin': ['xamarin'],
        'mobile ui': ['mobile ui', 'mobile design']
    },
    'other_tech': {
        'blockchain': ['blockchain', 'ethereum', 'solidity', 'web3'],
        'game dev': ['unity', 'unreal', 'game development'],
        'embedded': ['embedded', 'arduino', 'raspberry pi', 'iot'],
        'security': ['security', 'cybersecurity', 'penetration testing'],
        'networking': ['networking', 'cisco', 'juniper', 'routing'],
        'testing': ['testing', 'qa', 'selenium', 'junit']
    }
}

# Define role-specific skill requirements
ROLE_REQUIREMENTS = {
    'Software Developer': {
        'required': ['programming_languages', 'web_technologies'],
        'preferred': ['databases', 'cloud_devops']
    },
    'Data Scientist': {
        'required': ['data_science_ml', 'programming_languages'],
        'preferred': ['databases', 'cloud_devops']
    },
    'Full Stack Developer': {
        'required': ['programming_languages', 'web_technologies', 'databases'],
        'preferred': ['cloud_devops']
    },
    'DevOps Engineer': {
        'required': ['cloud_devops', 'linux'],
        'preferred': ['programming_languages', 'databases']
    },
    'Frontend Developer': {
        'required': ['web_technologies', 'javascript'],
        'preferred': ['programming_languages']
    },
    'Backend Developer': {
        'required': ['programming_languages', 'databases'],
        'preferred': ['web_technologies', 'cloud_devops']
    },
    'Machine Learning Engineer': {
        'required': ['data_science_ml', 'programming_languages'],
        'preferred': ['cloud_devops', 'databases']
    },
    'Cloud Architect': {
        'required': ['cloud_devops'],
        'preferred': ['programming_languages', 'databases']
    },
    'Mobile Developer': {
        'required': ['mobile_dev'],
        'preferred': ['programming_languages', 'databases']
    },
    'Database Administrator': {
        'required': ['databases'],
        'preferred': ['cloud_devops', 'programming_languages']
    },
    'Security Engineer': {
        'required': ['security'],
        'preferred': ['networking', 'programming_languages']
    },
    'Product Manager': {
        'required': ['other_tech'],
        'preferred': ['programming_languages', 'web_technologies']
    },
    'UI/UX Designer': {
        'required': ['web_technologies'],
        'preferred': ['programming_languages']
    },
    'QA Engineer': {
        'required': ['testing'],
        'preferred': ['programming_languages', 'web_technologies']
    },
    'Systems Administrator': {
        'required': ['linux', 'networking'],
        'preferred': ['cloud_devops', 'security']
    },
    'Business Analyst': {
        'required': ['data_science_ml'],
        'preferred': ['databases', 'programming_languages']
    },
    'Technical Writer': {
        'required': ['other_tech'],
        'preferred': ['programming_languages', 'web_technologies']
    },
    'Blockchain Developer': {
        'required': ['blockchain'],
        'preferred': ['programming_languages', 'security']
    },
    'Game Developer': {
        'required': ['game dev'],
        'preferred': ['programming_languages', 'graphics']
    },
    'Embedded Systems Engineer': {
        'required': ['embedded'],
        'preferred': ['programming_languages', 'hardware']
    },
    'Data Engineer': {
        'required': ['data_science_ml', 'databases'],
        'preferred': ['cloud_devops', 'programming_languages']
    },
    'AI Researcher': {
        'required': ['data_science_ml'],
        'preferred': ['programming_languages', 'mathematics']
    },
    'Network Engineer': {
        'required': ['networking'],
        'preferred': ['security', 'cloud_devops']
    },
    'SRE': {
        'required': ['cloud_devops', 'linux'],
        'preferred': ['programming_languages', 'networking']
    }
}

def preprocess_text(text):
    """Preprocess the text by tokenizing, removing stopwords, and lemmatizing."""
    # Convert to lowercase
    text = text.lower()
    
    # Preserve technical terms before cleaning
    technical_terms = []
    for category in SKILLS.values():
        for skill, keywords in category.items():
            for keyword in keywords:
                if keyword in text:
                    technical_terms.append(keyword)
    
    # Remove special characters and numbers, but preserve technical terms
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    # Add back technical terms
    tokens.extend(technical_terms)
    
    return ' '.join(tokens)

def extract_skills(text):
    """Extract skills from the text using the skill dictionaries."""
    text = text.lower()
    skills = defaultdict(int)
    
    # Check each skill category
    for category, skills_dict in SKILLS.items():
        for skill, keywords in skills_dict.items():
            for keyword in keywords:
                # Use word boundary matching for more accurate skill detection
                pattern = r'\b' + re.escape(keyword) + r'\b'
                matches = re.findall(pattern, text)
                if matches:
                    skills[skill] += len(matches)
    
    # Sort skills by frequency
    sorted_skills = sorted(skills.items(), key=lambda x: x[1], reverse=True)
    return [skill for skill, _ in sorted_skills]

def get_skill_categories(skills):
    """Get the categories of the extracted skills."""
    categories = defaultdict(int)
    
    for skill in skills:
        for category, skills_dict in SKILLS.items():
            if skill in skills_dict:
                categories[category] += 1
    
    return dict(categories)

def calculate_role_score(skills, skill_categories, role):
    """Calculate a score for how well the skills match a role."""
    requirements = ROLE_REQUIREMENTS[role]
    score = 0
    
    # Check required categories
    for category in requirements['required']:
        if category in skill_categories:
            score += skill_categories[category] * 3  # Higher weight for required skills
    
    # Check preferred categories
    for category in requirements['preferred']:
        if category in skill_categories:
            score += skill_categories[category] * 2  # Medium weight for preferred skills
    
    # Add bonus for having multiple skills in required categories
    required_skills_count = sum(skill_categories.get(cat, 0) for cat in requirements['required'])
    if required_skills_count >= 2:
        score += required_skills_count
    
    return score

def get_recommendations(skills, skill_categories, top_n=3):
    """Get role recommendations based on skills and categories."""
    role_scores = {}
    
    # Calculate scores for each role
    for role in ROLE_REQUIREMENTS.keys():
        score = calculate_role_score(skills, skill_categories, role)
        role_scores[role] = score
    
    # Sort roles by score
    sorted_roles = sorted(role_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Get top N roles
    top_roles = [role for role, score in sorted_roles[:top_n] if score > 0]
    
    # If no roles have a positive score, return the top role anyway
    if not top_roles:
        return [sorted_roles[0][0]]
    
    return top_roles