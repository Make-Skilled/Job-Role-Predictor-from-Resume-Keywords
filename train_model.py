# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import LeaveOneOut, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
import joblib
import os
import logging
from datetime import datetime
from utils import preprocess_text, extract_skills, get_skill_categories
import json
import seaborn as sns
import matplotlib.pyplot as plt

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('model/training.log'),
        logging.StreamHandler()
    ]
)

def validate_data(data):
    """Validate and clean the training data."""
    logging.info("Validating training data...")
    
    # Check for missing values
    if data.isnull().any().any():
        logging.warning("Found missing values in the dataset")
        data = data.dropna()
    
    # Check for duplicate entries
    duplicates = data.duplicated()
    if duplicates.any():
        logging.warning(f"Found {duplicates.sum()} duplicate entries")
        data = data.drop_duplicates()
    
    # Check for empty texts
    empty_texts = data['resume_text'].str.strip().str.len() == 0
    if empty_texts.any():
        logging.warning(f"Found {empty_texts.sum()} empty texts")
        data = data[~empty_texts]
    
    # Check class distribution
    class_counts = data['job_role'].value_counts()
    logging.info("\nClass distribution:")
    for role, count in class_counts.items():
        logging.info(f"{role}: {count} examples")
    
    return data

def augment_training_data(data):
    """Augment training data by creating variations of existing examples."""
    logging.info("Augmenting training data...")
    augmented_data = []
    
    for _, row in data.iterrows():
        # Original example
        augmented_data.append(row)
        
        # Create variations by combining skills and categories
        skills = row['skills']
        categories = row['skill_categories']
        
        # Variation 1: Emphasize required skills
        if len(skills) > 1:
            new_row = row.copy()
            new_row['resume_text'] = f"{row['resume_text']} {' '.join(skills[:2])}"
            augmented_data.append(new_row)
        
        # Variation 2: Add category context
        if categories:
            new_row = row.copy()
            category_text = ' '.join(f"{k} {v}" for k, v in categories.items())
            new_row['resume_text'] = f"{row['resume_text']} {category_text}"
            augmented_data.append(new_row)
    
    return pd.DataFrame(augmented_data)

def create_model_pipeline():
    """Create a pipeline for text processing and classification."""
    return Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 3),
            min_df=1,  # Reduced from 2 to handle small dataset
            max_df=0.95,
            stop_words='english',
            sublinear_tf=True,
            use_idf=True,
            smooth_idf=True
        )),
        ('classifier', RandomForestClassifier(
            n_estimators=200,  # Reduced from 500 for small dataset
            max_depth=10,  # Reduced from 20
            min_samples_split=2,  # Reduced from 5
            min_samples_leaf=1,  # Reduced from 2
            max_features='sqrt',
            bootstrap=True,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        ))
    ])

def evaluate_model(model, X, y):
    """Evaluate the model using leave-one-out cross-validation."""
    logging.info("Evaluating model performance...")
    
    # Use leave-one-out cross-validation for small dataset
    loo = LeaveOneOut()
    scores = cross_val_score(model, X, y, cv=loo, scoring='accuracy')
    
    # Calculate metrics
    accuracy = scores.mean()
    std_dev = scores.std()
    
    # Log metrics
    logging.info(f"Cross-validation accuracy: {accuracy:.3f} (+/- {std_dev:.3f})")
    
    # Train final model on all data
    model.fit(X, y)
    
    # Get feature importance
    feature_names = model.named_steps['tfidf'].get_feature_names_out()
    importances = model.named_steps['classifier'].feature_importances_
    
    # Create DataFrame for feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    # Log top features
    logging.info("\nTop 20 most important features:")
    for _, row in feature_importance.head(20).iterrows():
        logging.info(f"{row['feature']}: {row['importance']:.3f}")
    
    # Calculate and log per-class metrics
    y_pred = model.predict(X)
    class_report = classification_report(y, y_pred)
    logging.info("\nClassification Report:\n" + class_report)
    
    # Save metrics
    metrics = {
        'accuracy': accuracy,
        'std_dev': std_dev,
        'top_features': feature_importance.head(20).to_dict('records'),
        'classification_report': class_report
    }
    
    return model, metrics

def plot_feature_importance(feature_importance, output_dir):
    """Plot and save feature importance visualization."""
    logging.info("Generating feature importance plot...")
    
    # Plot top 20 features
    plt.figure(figsize=(12, 6))
    sns.barplot(x='importance', y='feature', data=feature_importance.head(20))
    plt.title('Top 20 Most Important Features')
    plt.tight_layout()
    
    # Save plot
    plt.savefig(os.path.join(output_dir, 'feature_importance.png'))
    plt.close()

def save_model_artifacts(model, metrics, output_dir):
    """Save model artifacts with versioning."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    version_dir = os.path.join(output_dir, f'version_{timestamp}')
    os.makedirs(version_dir, exist_ok=True)
    
    # Save model
    joblib.dump(model, os.path.join(version_dir, 'model.pkl'))
    
    # Save metrics
    with open(os.path.join(version_dir, 'metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=4)
    
    # Update latest version
    with open(os.path.join(output_dir, 'latest_version.txt'), 'w', encoding='utf-8') as f:
        f.write(timestamp)
    
    logging.info(f"Model artifacts saved to {version_dir}")

def train_model():
    """Main function to train and evaluate the model."""
    logging.info("Starting model training process...")
    
    # Create model directory
    model_dir = 'model'
    os.makedirs(model_dir, exist_ok=True)
    
    try:
        # Load and validate data
        data = pd.read_csv('data/training_data.csv', encoding='utf-8')
        data = validate_data(data)
        
        # Preprocess the resume texts
        logging.info("Preprocessing resume texts...")
        data['processed_text'] = data['resume_text'].apply(preprocess_text)
        
        # Extract skills and categories
        logging.info("Extracting skills and categories...")
        data['skills'] = data['processed_text'].apply(extract_skills)
        data['skill_categories'] = data['skills'].apply(get_skill_categories)
        
        # Augment training data
        logging.info("Augmenting training data...")
        data = augment_training_data(data)
        logging.info(f"Training data size after augmentation: {len(data)} examples")
        
        # Recompute skill-based features for augmented data
        logging.info("Adding skill-based features...")
        data['skill_text'] = data['skills'].apply(lambda skills: ' '.join(skills))
        data['category_text'] = data['skill_categories'].apply(lambda cats: ' '.join(f"{k}_{v}" for k, v in cats.items()))
        combined_text = data['processed_text'] + ' ' + data['skill_text'] + ' ' + data['category_text']
        
        # Create model pipeline
        pipeline = create_model_pipeline()
        
        # Evaluate model using leave-one-out cross-validation
        best_model, metrics = evaluate_model(
            pipeline,
            combined_text,
            data['job_role']
        )
        
        # Generate and save visualizations
        feature_importance = pd.DataFrame({
            'feature': best_model.named_steps['tfidf'].get_feature_names_out(),
            'importance': best_model.named_steps['classifier'].feature_importances_
        }).sort_values('importance', ascending=False)
        
        plot_feature_importance(feature_importance, model_dir)
        
        # Save model artifacts
        save_model_artifacts(best_model, metrics, model_dir)
        
        logging.info("Model training completed successfully!")
        
    except Exception as e:
        logging.error(f"Error during model training: {str(e)}")
        raise

if __name__ == "__main__":
    train_model()
