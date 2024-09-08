import pickle
from model_trainer import predict

def get_model_data(filename):
    try:
        model_data = {}
        with open(filename, 'rb') as file:
            model_data = pickle.load(file)
    except FileNotFoundError:
        print("Pickled file not found! Please train the model and try again!")
    finally:
        return model_data

def get_input(prompt):
    user_input = None
    while not user_input:
        user_input = input(prompt)
    return user_input

def main():
    user_input = get_input("Enter the message: ")
    model_data = get_model_data("trained_model.pkl")

    if not (user_input or model_data):
        return

    vectorizer = model_data["vectorizer"]
    scaler = model_data["scaler"]
    w_array = model_data["w_array"]

    new_message = [user_input]
    encoded_new_message = vectorizer.transform(new_message).toarray()
    encoded_new_message = scaler.transform(encoded_new_message)
    new_prediction = predict(encoded_new_message, w_array)

    print('Spam' if new_prediction > 0.5 else 'Not Spam', new_prediction)

if __name__ == "__main__":
    main()

