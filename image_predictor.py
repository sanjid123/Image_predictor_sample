import cv2
import numpy as np
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions

# Load pre-trained VGG16 model
model = VGG16(weights='imagenet')

# Replace with the path to your image
image_path = 'path/to/your/image.jpg'

# Load and preprocess the image for inference
image = cv2.imread(image_path)
resized_image = cv2.resize(image, (224, 224))
preprocessed_image = preprocess_input(np.expand_dims(resized_image, axis=0))

# Make a prediction
predictions = model.predict(preprocessed_image)
decoded_predictions = decode_predictions(predictions, top=1)[0]

# Display the top predicted label (ImageNet class)
top_prediction = decoded_predictions[0]
print(f"Predicted class: {top_prediction[1]}")
