"""
------- This is a refined version of the original version ------

Original file is located at
    https://colab.research.google.com/drive/1MFxWal0TbzP5FnrNoSKtygINWHfKn2DV

# Logistics Regression

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import pickle # to store the trained model

# Sigmoid function for predictions
def predict(x_array, w_array):
    z = np.dot(x_array, w_array)
    return 1 / (1 + np.exp(-z))

# Cost function
def cost_function(x_matrix, w_array, y_array):
    predictions = predict(x_matrix, w_array)
    predictions = np.clip(predictions, 1e-15, 1 - 1e-15)
    cost = - y_array * np.log(predictions) - (1 - y_array) * np.log(1 - predictions)
    return np.mean(cost)

# Gradient Descent Function
def gradient_descent(x_matrix, y_array, w_array, learning_rate):
    predictions = predict(x_matrix, np.transpose(w_array))
    errors = predictions - y_array
    gradients = learning_rate * (np.dot(np.transpose(x_matrix), errors) / len(y_array))
    return w_array - gradients

def main():
    # Reading the dataset
    df = pd.read_json("spam_dataset.json")

    # Feature extraction using TfidfVectorizer
    vectorizer = TfidfVectorizer(stop_words='english')
    x_matrix = vectorizer.fit_transform(df['Message']).toarray()

    # Target (Spamness)
    y_array = df['Spamness'].to_numpy()

    # Splitting into train and test sets
    x_train, x_test, y_train, y_test = train_test_split(x_matrix, y_array, test_size=0.2, random_state=42)

    # Scaling the data
    scaler = StandardScaler(with_mean=False)  # Avoids issues with sparse matrix
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    # Initializing weights
    w_array = np.zeros(x_train.shape[1])

    # Training parameters
    iterations = 1000
    learning_rate = 0.1
    cost_values = []

    # Training the model
    for i in range(iterations):
        cost = cost_function(x_train, w_array, y_train)
        cost_values.append(cost)
        if i % 100 == 0:
            print(f"Iteration: {i} | Cost: {cost}")
        w_array = gradient_descent(x_train, y_train, w_array, learning_rate)

    # Plotting the cost over iterations
    plt.plot(range(iterations), cost_values)
    plt.xlabel("Iteration")
    plt.ylabel("Cost")
    plt.show()

    # Accuracy testing on test set
    y_pred = [1 if predict(x, w_array) > 0.5 else 0 for x in x_test]
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Accuracy on test set: {accuracy * 100:.2f}%")

    # Storing the Model
    with open('trained_model.pkl', 'wb') as file:
        modal_data = {
            'vectorizer':vectorizer,
            'scaler':scaler,
            'w_array':w_array}
        pickle.dump(modal_data, file)

if __name__ == "__main__":
    main()