# Spam Message Classifier

## Overview

The **Spam Message Classifier** is a machine learning project aimed at classifying text messages as "spam" or "non-spam" using **logistic regression**. This project demonstrates a ground-up approach to machine learning classification, from data preprocessing to model evaluation. The classification is carried out using the **logistic regression algorithm**, and various Python libraries are utilized to efficiently handle data, preprocess it, and perform vector operations.

Key technologies used in this project include:
- **NumPy** for fast and efficient vectorized operations
- **Pandas** for data manipulation and preprocessing
- **SciKit-Learn** for machine learning tasks, including model training and evaluation
- **TfidfVectorizer** for transforming text data into feature vectors suitable for machine learning

## Dataset

The primary dataset used in this project is the **SMS Spam Collection** dataset, which consists of SMS messages classified as either "spam" or "ham" (non-spam). The dataset is publicly available and can be accessed from the following link:

[SMS Spam Collection Dataset](https://archive.ics.uci.edu/dataset/228/sms+spam+collection)

The dataset contains:
- **Messages**: Text data in the form of SMS messages
- **Labels**: A binary classification label indicating whether a message is spam (`1`) or non-spam (`0`)

## Steps Involved

1. **Data Collection**:
   - Load the SMS Spam Collection dataset into a Pandas DataFrame.

2. **Data Preprocessing**:
   - Clean the text data by removing special characters, stop words, and performing tokenization.
   - Split the data into training and testing sets.

3. **Feature Extraction**:
   - Use **TfidfVectorizer** to convert text messages into numerical features that can be used by the logistic regression model.

4. **Model Training**:
   - Apply logistic regression using SciKit-Learn to classify the messages.

5. **Evaluation**:
   - Evaluate the performance of the model using common metrics such as accuracy, precision, recall, and F1 score.

6. **Model Optimization**:
   - Optionally, perform hyperparameter tuning to improve the performance of the logistic regression model.

## Installation

To run the Spam Message Classifier project, you will need the following Python libraries:

- **NumPy**: For numerical operations
- **Pandas**: For data manipulation
- **SciKit-Learn**: For machine learning tasks
- **Matplotlib** (optional, for visualizations)

You can install the required dependencies using `pip`:

```bash
pip install numpy pandas scikit-learn matplotlib
```
