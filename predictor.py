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
    try:
        user_input = get_input("Enter the message: ")
        model_data = get_model_data("trained_models/sms_data_model.pkl")

        if not (user_input or model_data):
            return

        vectorizer = model_data["vectorizer"]
        scaler = model_data["scaler"]
        w_array = model_data["w_array"]

        new_message = [user_input]
        encoded_new_message = vectorizer.transform(new_message).toarray()
        encoded_new_message = scaler.transform(encoded_new_message)
        new_prediction = predict(encoded_new_message, w_array)[0]

        prediction_statement = "Spam"
        if new_prediction < 0.5:
            prediction_statement = "Not Spam"
        
        if new_prediction == 0.5:
            prediction_statement = "May be spam or not"

        print(f"Prediction Value     : {new_prediction *100 :.3f} %")
        print(f"Prediction Statement : {prediction_statement}")

    except KeyboardInterrupt:
        print("\nProcess Aborted!")

if __name__ == "__main__":
    main()
