import pandas as pd
import os

class DataLoader:
    """Handles parsing and normalizing diverse datasets into a unified format."""
    
    @staticmethod
    def load_amazon(filepath):
        # Amazon: No headers. Col1=Sentiment(1=Neg, 2=Pos), Col2=Title, Col3=Body
        df = pd.read_csv(filepath, header=None, names=['Label', 'Title', 'Body'], on_bad_lines='skip')
        df['Text'] = df['Title'].fillna('') + " " + df['Body'].fillna('')
        df['Label'] = df['Label'].map({1: 'Negative', 2: 'Positive'})
        df['Source'] = 'Amazon'
        return df[['Text', 'Label', 'Source']].dropna()

    @staticmethod
    def load_twitter(filepath):
        # Twitter: No headers. Col1=Sentiment(0=Neg, 4=Pos), Col6=Text
        cols = ['Label', 'ID', 'Date', 'Query', 'User', 'Text']
        df = pd.read_csv(filepath, header=None, names=cols, on_bad_lines='skip', encoding='latin-1')
        df['Label'] = df['Label'].map({0: 'Negative', 4: 'Positive'})
        df['Source'] = 'Twitter'
        return df[['Text', 'Label', 'Source']].dropna()

    @staticmethod
    def load_flipkart(filepath):
        # Flipkart: Headers exist. Combine Review + Summary.
        df = pd.read_csv(filepath, on_bad_lines='skip')
        df['Text'] = df['Review'].fillna('') + " " + df['Summary'].fillna('')
        df['Label'] = df['Sentiment'].str.capitalize() # 'positive' -> 'Positive'
        df['Source'] = 'Flipkart'
        return df[['Text', 'Label', 'Source']].dropna()

    @staticmethod
    def get_unified_dataset():
        """Loads available data and concatenates them into one uniform pipeline."""
        datasets = []
        if os.path.exists('data/Amazon_train.csv'):
            datasets.append(DataLoader.load_amazon('data/Amazon_train.csv'))
        if os.path.exists('data/Twitter.csv'):
            datasets.append(DataLoader.load_twitter('data/Twitter.csv'))
        if os.path.exists('data/Flipkart.csv'):
            datasets.append(DataLoader.load_flipkart('data/Flipkart.csv'))
            
        if not datasets:
            raise FileNotFoundError("No datasets found in the 'data/' folder.")

        unified_df = pd.concat(datasets, ignore_index=True)
        # Filter to binary sentiment for baseline ML model
        unified_df = unified_df[unified_df['Label'].isin(['Positive', 'Negative'])]
                
        # Sample for performance on standard laptops
        if len(unified_df) > 10000:
            unified_df = unified_df.sample(10000, random_state=42)
                    
        return unified_df