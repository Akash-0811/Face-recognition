import tensorflow as tf
from tensorflow.keras import layers, models, applications
import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Set paths
dataset_path = 'path_to_your_dataset'

# Load and preprocess images
def load_images(dataset_path):
    images = []
    labels = []
    label_dict = {}
    current_label = 0
    for label_name in os.listdir(dataset_path):
        label_path = os.path.join(dataset_path, label_name)
        if os.path.isdir(label_path):
            for image_name in os.listdir(label_path):
                image_path = os.path.join(label_path, image_name)
                image = cv2.imread(image_path)
                image = cv2.resize(image, (224, 224))  # Resize to match input size for pre-trained models
                images.append(image)
                if label_name not in label_dict:
                    label_dict[label_name] = current_label
                    current_label += 1
                labels.append(label_dict[label_name])
    return np.array(images), np.array(labels), label_dict

images, labels, label_dict = load_images(dataset_path)
images = images / 255.0  # Normalize images

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Load pre-trained model and add custom layers
base_model = applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(len(label_dict), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=10, validation_split=0.2)

# Fine-tune the model
base_model.trainable = True
model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history_fine = model.fit(X_train, y_train, epochs=10, validation_split=0.2)

# Evaluate the model
y_pred = np.argmax(model.predict(X_test), axis=1)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')

# Save the model
model.save('face_recognition_model.h5')

# Load and use the model for prediction
def recognize_face(image_path, model, label_dict):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))
    image = np.expand_dims(image, axis=0) / 255.0
    prediction = np.argmax(model.predict(image))
    for label, idx in label_dict.items():
        if idx == prediction:
            return label
    return None

# Example usage
model = tf.keras.models.load_model('face_recognition_model.h5')
result = recognize_face('path_to_image.jpg', model, label_dict)
print(f'Recognized face: {result}')
