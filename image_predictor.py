!pip install opencv-python-headless
import cv2
import numpy as np
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions

# Load pre-trained VGG16 model
model = VGG16(weights='imagenet')

# Upload an image file in Google Colab
from google.colab import files
uploaded = files.upload()

# Get the file name of the uploaded image
image_filename = list(uploaded.keys())[0]

# Load and preprocess the uploaded image for inference
image = cv2.imread(image_filename)
resized_image = cv2.resize(image, (224, 224))
preprocessed_image = preprocess_input(np.expand_dims(resized_image, axis=0))

# Make a prediction
predictions = model.predict(preprocessed_image)
decoded_predictions = decode_predictions(predictions, top=1)[0]

# Display the top predicted label (ImageNet class)
top_prediction = decoded_predictions[0]
print(f"Predicted class: {top_prediction[1]}")
