from transformers import pipeline

class TransformerLayer:
    def __init__(self):
        self.summarizer = None
        self.sentiment_analyzer = None
        
    def load_models(self):
        if not self.summarizer:
            self.summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
        if not self.sentiment_analyzer:
            self.sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

    def run_sentiment_comparison(self, sample_texts):
        """Runs HuggingFace sentiment on a small batch to compare with LogReg.
        We sample to avoid freezing the CPU on standard laptops."""
        self.load_models()
        try:
            results = self.sentiment_analyzer(sample_texts[:50]) # Limit to 50 for speed
            return results
        except Exception as e:
            return str(e)