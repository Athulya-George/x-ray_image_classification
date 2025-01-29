from flask import Flask, request, render_template, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.losses import CategoricalCrossentropy
import numpy as np
import os

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './uploads'

# Define the labels
labels = {'covid': 0, 'normal': 1, 'pneumonia': 2}

# Load the saved model (ensure the model path is correct)
model = load_model('C:\\Users\\athul\\OneDrive\\Desktop\\xray_img_clsification\\project\\model\\xray (1).h5', compile=False)  # Load without compiling
model.compile(optimizer='adam', loss=CategoricalCrossentropy(reduction='sum_over_batch_size'))


def model_predict(image_path):
    try:
        # Load and preprocess the image
        img = load_img(image_path, target_size=(256, 256))
        img = img_to_array(img)
        img = img.reshape(1, 256, 256, 3)  # Ensure proper shape (batch_size, height, width, channels)

        # Make predictions
        result = model.predict(img)
        preds1 = np.argmax(result, axis=1)[0]  # Get the predicted class

        # Interpret the prediction
        if preds1 == 0:
            prediction = "You are diagnosed with COVID. Please consult a doctor."
        elif preds1 == 1:
            prediction = "You are a healthy person."
        elif preds1 == 2:
            prediction = "You have pneumonia."
        else:
            prediction = "Unable to determine the diagnosis."

        return prediction

    except Exception as e:
        return f"Error processing the image: {e}"

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    
    # Save uploaded file
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    # Make prediction using the new model_predict function
    prediction = model_predict(file_path)

    return render_template('index.html', prediction=prediction)


if __name__ == '__main__':
    # Create upload folder if it doesn't exist
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    
    # Run the Flask app
    app.run(debug=True)
