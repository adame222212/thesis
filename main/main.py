import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
from flask import Flask, render_template, request, jsonify
from io import BytesIO
from PIL import Image

app = Flask(__name__, static_url_path='/main/static')



# Load necessary data and models for the chatbot
lemmatizer = WordNetLemmatizer()
intents = json.loads(open('NLP/intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbotmodel.h5')

# Load necessary data and models for image classification
MODEL_PATH = "corns.h5"
MODEL = load_model(MODEL_PATH)

CLASS_NAMES = ["Common Rust", "Gray Spot", "Healthy", "Northern Blight"]
CLASS_DIF = [
    "This is  Common rust, Common rust is caused by the fungus Puccinia sorghi and occurs every growing season. It is seldom a concern in hybrid corn. Rust pustules usually first appear in late June. Early symptoms of common rust are chlorotic flecks on the leaf surface.",
    "This is Gray leaf spot, Gray leaf spot (GLS) is a common fungal disease in the United States caused by the pathogen Cercospora zeae-maydis in corn. Disease development is favored by warm temperatures, 80°F or 27 °C; and high humidity, relative humidity of 90% or higher for 12 hours or more.",
    "This is healthy, no worries",
    "This is Northern corn leaf blight, Northern corn leaf blight (NCLB) is caused by the fungus Setosphaeria turcica. Symptoms usually appear first on the lower leaves. Leaf lesions are long (1 to 6 inches) and elliptical, gray-green at first but then turn pale gray or tan."
]

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

def get_response(intents_list, intents_json):
    if intents_list:
        tag = intents_list[0]['intent']
        list_of_intents = intents_json['intents']
        for i in list_of_intents:
            if i['tag'] == tag:
                result = random.choice(i['responses'])
                break
        return result
    else:
        # Handle the case where intents_list is empty
        return "I'm sorry, I couldn't understand your message."

# Function to read and preprocess the file as an image for image classification
def read_file_as_image(data):
    image = Image.open(BytesIO(data))
    # Resize the image to the expected shape (256x256)
    image = image.resize((256, 256))
    image = np.array(image)
    return image

@app.route('/')
def home():
    return render_template('indexs.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.form['user_message']
    ints = predict_class(user_message)
    res = get_response(ints, intents)
    return res

@app.route('/image_classification', methods=['POST'])
def image_classification():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        image = read_file_as_image(file.read())
        img_batch = np.expand_dims(image, 0)

        predictions = MODEL.predict(img_batch)

        predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
        predicted_DIF = CLASS_DIF[np.argmax(predictions[0])]
        confidence = float(np.max(predictions[0]))
        return jsonify({
            'class': predicted_class,
            'Diff': predicted_DIF,
            'confidence': confidence
        })

if __name__ == '__main__':
    app.run(debug=True)
