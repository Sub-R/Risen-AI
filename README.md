# AI-Powered Business Intelligence Agent

**Disruptive Technologies Project**

This project is a complete end-to-end sentiment analysis pipeline and interactive chatbot designed to extract actionable business intelligence from customer reviews.

## 🚀 Features

1. **Sentiment Engine**: Analyzes textual reviews using NLP (`TextBlob`) to classify them into Positive, Negative, and Neutral.
2. **Business Insights**: Automatically identifies top complaints, positive keywords, and calculates a Customer Satisfaction KPI.
3. **Visualization**: Generates professional charts (Pie charts, Bar graphs, and Word Clouds).
4. **AI Chatbot**: A terminal-based virtual assistant that answers executive questions about the data.

## ⚙️ Installation & Setup

1. **Install Python**: Ensure Python 3.x is installed on your computer.
2. **Install Libraries**: Open your terminal and run:
   ```bash
   pip install -r requirements.txt
   ```
   
## 📊 Dataset Setup Instructions

To run this Business Intelligence application, you need to download the source datasets and place them in the local directory structure. The pipeline is built to handle three specific data formats automatically.

### 1. Required Directory Structure
Create a folder named `data` in the project root directory:
```text

### project/
├── data/
│   ├── Amazon_train.csv
│   ├── Twitter.csv
│   └── Flipkart.csv
```

###2. Dataset Sources & Formatting
Download the following datasets and place them into the data/ folder:
```
Amazon Dataset (Amazon_train.csv): https://www.kaggle.com/datasets/dongrelaxman/amazon-reviews-dataset

Format: No headers. Column 1: Sentiment (1 = Negative, 2 = Positive), Column 2: Title, Column 3: Review Body.

Twitter Dataset (Twitter.csv): https://www.kaggle.com/datasets/dongrelaxman/amazon-reviews-dataset

Format: No headers, latin-1 encoding. Column 1: Sentiment (0 = Negative, 4 = Positive), Column 6: Tweet Text.

Flipkart Dataset (Flipkart.csv): https://www.kaggle.com/datasets/dongrelaxman/amazon-reviews-dataset

Format: Headers must include Review, Summary, and Sentiment (with values labeled as 'positive' or 'negative').
```

Note: The application's DataLoader automatically handles downsampling to 10,000 records upon execution to ensure smooth performance on standard laptops.

Once you've updated the `README.md`, just save it, run `git add README.md`, `git commit -m "Update dataset setup instructions"`, and do one final quick `git push`!
