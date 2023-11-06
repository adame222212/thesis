import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import Adam  # Changed optimizer to Adam

# Initialize the WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

# Load intents from a JSON file
with open('NLP/intents.json') as file:
    intents = json.load(file)

# Initialize empty lists for words, classes, and documents
words = []
classes = []
documents = []

# Define a set of characters to ignore
ignore_letters = ['!', '&', '?', '.', ',']

# Preprocess the intents and tokenize words
for intent in intents['intents']:
    for pattern in intent['patterns']:
        words.extend(nltk.word_tokenize(pattern))
        documents.append((nltk.word_tokenize(pattern), intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatize and remove ignored characters from words
words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignore_letters]
words = sorted(set(words))

# Save words and classes to pickle files
with open('words.pkl', 'wb') as f:
    pickle.dump(words, f)
with open('classes.pkl', 'wb') as f:
    pickle.dump(classes, f)

# Initialize training data
training = []
output_empty = [0] * len(classes)

# Create the bag of words for each pattern
for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row])

# Shuffle the training data
random.shuffle(training)
training = np.array(training)

# Separate features (X) and labels (y)
train_x = list(training[:, 0])
train_y = list(training[:, 1])

# Create a Sequential model
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Change optimizer to Adam
adam = Adam(learning_rate=0.01)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

# Train the model
model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=6, verbose=1)

# Save the model
model.save('chatbotmodel.h5')

print("Model training completed.")
