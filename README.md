# Financial-Market-News-Sentiment-Analysis-Project

#Introduction
Sentiment analysis is an essential tool for understanding market trends, investor sentiment, and aiding financial decision-making. In this project, we aim to predict the sentiment (positive or negative) of financial market news headlines using machine learning techniques. By analysing the content of daily financial news, this model can help investors and analysts better understand the emotional tone of the market and make informed decisions.
The project uses a dataset of financial news headlines and applies the Random Forest Classifier to classify the sentiment. With over 4,000 news entries, this model offers insights into how news impacts the financial market.

#Objective
To perform sentiment analysis on financial news headlines and classify them as either positive or negative using a Random Forest Classifier.
Data Source
The dataset used in this project consists of daily financial news headlines and can be found here.
________________________________________
#Project outline

1. Import Libraries
import pandas as pd
import numpy as np

3. Import Data
df = pd.read_csv('https://github.com/YBI-Foundation/Dataset/raw/main/Financial%20Market%20News.csv', encoding='ISO-8859-1')

5. Describe Data
df.info()
df.shape
df.head()

7. Data Preprocessing
•	Concatenate all news headlines for each day.
•	Convert the text data into a bag of words using CountVectorizer.

9. Define Target Variable (y) and Feature Variables (X)
•	Target Variable (y): 'Label' (indicating sentiment as 0 or 1).
•	Feature Variables (X): Bag of words representation of the concatenated news headlines.

11. Train Test Split
We use test split

13. Modeling
We use a Random Forest Classifier with 200 estimators for the model.

15. Model Evaluation
Evaluate the model's performance using a confusion matrix and classification report.

17. Prediction
The model can now predict whether new financial news headlines are positive or negative based on the training.

19. Explanation
This project uses a bag-of-words approach to represent text data from financial news headlines, and a Random Forest Classifier to predict sentiment. The accuracy of the model provides insights into how news impacts market sentiment, which is crucial for investors and analysts alike.

Dataset
The dataset used in this project is available at the following link: https://github.com/YBI-Foundation/Dataset/raw/main/Financial%20Market%20News.csv
________________________________________
Conclusion
By training the model on historical financial news data, we provide a framework for sentiment analysis that helps stakeholders make data-driven decisions in financial markets. The model's predictions can be integrated into trading algorithms, investment strategies, or risk assessments.

