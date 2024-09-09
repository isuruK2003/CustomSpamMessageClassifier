import json

data = []
lines = []

# Reading the original file
with open("data/original/SMSSpamCollection") as file:
    lines.extend(file.readlines())

# Extracting the data
for line in lines:
    line = line.strip("\n")
    spamness = 1 if line[0:4].strip() == "spam" else 0
    message = line[4:-1].strip()
    data.append({"Message":message, "Spamness":spamness})

    print(line)
    print(spamness, message)
    print("-----------------------------------------")

# Saving the data as JSON
with open("data/json/sms_spam_collection.json", "w") as file:
    json.dump(data, file, indent=2)
