from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pandas as pd
from collections import Counter
import re
import os
import joblib

class ClassicalMLEngine:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        self.classifier = LogisticRegression(max_iter=1000)
        self.metrics = {}
        self.conf_matrix = None
        self.save_dir = 'models/saved'
        os.makedirs(self.save_dir, exist_ok=True)

    def load_saved_model(self):
        """Loads models if they exist, returning True if successful."""
        try:
            self.vectorizer = joblib.load(f'{self.save_dir}/vectorizer.joblib')
            self.classifier = joblib.load(f'{self.save_dir}/classifier.joblib')
            self.metrics = joblib.load(f'{self.save_dir}/metrics.joblib')
            self.conf_matrix = joblib.load(f'{self.save_dir}/conf_matrix.joblib')
            return self.metrics, self.conf_matrix
        except FileNotFoundError:
            return None, None

    def train_and_evaluate(self, df):
        X = self.vectorizer.fit_transform(df['Text'])
        y = df['Label']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.classifier.fit(X_train, y_train)
        y_pred = self.classifier.predict(X_test)
        
        neg_text = " ".join(df[df['Label'] == 'Negative']['Text'].astype(str).tolist())
        words = re.findall(r'\w+', neg_text.lower())
        meaningful_words = [w for w in words if len(w) > 4 and w not in ['which', 'there', 'their', 'would', 'could']]
        top_neg_words = [word for word, count in Counter(meaningful_words).most_common(10)]
        
        pos_text = " ".join(df[df['Label'] == 'Positive']['Text'].astype(str).tolist())
        pos_words = re.findall(r'\w+', pos_text.lower())
        meaningful_pos = [w for w in pos_words if len(w) > 4 and w not in ['which', 'there', 'their', 'would', 'could']]
        top_pos_words = [word for word, count in Counter(meaningful_pos).most_common(10)]

        pos_label = 'Positive'
        self.metrics = {
            'Accuracy': round(accuracy_score(y_test, y_pred) * 100, 2),
            'Precision': round(precision_score(y_test, y_pred, pos_label=pos_label) * 100, 2),
            'Recall': round(recall_score(y_test, y_pred, pos_label=pos_label) * 100, 2),
            'F1_Score': round(f1_score(y_test, y_pred, pos_label=pos_label) * 100, 2),
            'Total_Analyzed': len(df),
            'Positive_Pct': round((len(df[df['Label'] == 'Positive']) / len(df)) * 100, 2),
            'Negative_Pct': round((len(df[df['Label'] == 'Negative']) / len(df)) * 100, 2),
            'Top_Negative_Words': top_neg_words,
            'Top_Positive_Words': top_pos_words,
            'Negative_Text_Corpus': neg_text,
            'Positive_Text_Corpus': pos_text
        }
        self.metrics['Customer_Satisfaction'] = self.metrics['Positive_Pct']
        self.conf_matrix = confusion_matrix(y_test, y_pred, labels=['Negative', 'Positive'])
        
        # Save models to disk
        joblib.dump(self.vectorizer, f'{self.save_dir}/vectorizer.joblib')
        joblib.dump(self.classifier, f'{self.save_dir}/classifier.joblib')
        joblib.dump(self.metrics, f'{self.save_dir}/metrics.joblib')
        joblib.dump(self.conf_matrix, f'{self.save_dir}/conf_matrix.joblib')
        
        return self.metrics, self.conf_matrix