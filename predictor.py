import pickle
from model_trainer import predict

with open('trained_model.pkl', 'rb') as file:
    modal_data = pickle.load(file)
    vectorizer = modal_data['vectorizer']
    scaler = modal_data['scaler']
    w_array = modal_data['w_array']

user_input = None

while not user_input: # will prevent blank inputs
    user_input = input("Enter the message: ")

new_message = [user_input]

encoded_new_message = vectorizer.transform(new_message).toarray()
encoded_new_message = scaler.transform(encoded_new_message)

new_prediction = predict(encoded_new_message, w_array)
print('Spam' if new_prediction > 0.5 else 'Not Spam', new_prediction)