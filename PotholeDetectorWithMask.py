import numpy as np
import cv2
import glob
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, GlobalAveragePooling2D, Conv2D
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import os
import matplotlib.pyplot as plt

global size
size = 100

def create_model():
    model = Sequential()
    model.add(Conv2D(16, (8, 8), strides=(4, 4), padding='valid', input_shape=(size,size,1)))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (5, 5), padding="same"))
    model.add(Activation('relu'))
    model.add(GlobalAveragePooling2D())
    model.add(Dense(512))
    model.add(Dropout(0.1))
    model.add(Activation('relu'))
    model.add(Dense(2))
    model.add(Activation('softmax'))
    return model

# Create model with same architecture
model = create_model()

# Load weights if they exist
try:
    model.load_weights('new_model.weights.h5')
    print("Loaded model weights successfully")
except:
    print("Could not load model weights. Using uninitialized model.")

## Store original images and their paths
original_images = []
image_paths = []

## load Testing data : non-pothole
nonPotholeTestImages = glob.glob(os.path.join(os.getcwd(), "My Dataset/test/Plain/*.jpg"))
for img_path in nonPotholeTestImages:
    img = cv2.imread(img_path)
    if img is not None:
        original_images.append(img)
        image_paths.append(img_path)

## load Testing data : potholes
potholeTestImages = glob.glob(os.path.join(os.getcwd(), "My Dataset/test/Pothole/*.jpg"))
for img_path in potholeTestImages:
    img = cv2.imread(img_path)
    if img is not None:
        original_images.append(img)
        image_paths.append(img_path)

# Prepare images for prediction
X_test = []
for img in original_images:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (size, size))
    X_test.append(resized)

X_test = np.asarray(X_test)
X_test = X_test.reshape(X_test.shape[0], size, size, 1)

# Create labels
y_test1 = np.ones([len(potholeTestImages)], dtype=int)
y_test2 = np.zeros([len(nonPotholeTestImages)], dtype=int)
y_test = np.concatenate([y_test2, y_test1])
y_test = to_categorical(y_test)

# Get predictions
predictions = model.predict(X_test)
predicted_classes = np.argmax(predictions, axis=1)

# Display results
for i in range(len(original_images)):
    plt.figure(figsize=(8, 6))
    
    # Display original image
    rgb_img = cv2.cvtColor(original_images[i], cv2.COLOR_BGR2RGB)
    # Resize image for display
    display_size = (400, 300)  # Smaller display size
    display_img = cv2.resize(rgb_img, display_size)
    plt.imshow(display_img)
    
    # Get prediction and confidence
    pred_label = "Pothole" if predicted_classes[i] == 1 else "Plain Road"
    confidence = predictions[i][predicted_classes[i]] * 100
    
    # Add title with prediction info
    plt.title(f'Prediction: {pred_label}\nConfidence: {confidence:.1f}%\nImage: {os.path.basename(image_paths[i])}', 
              pad=20, fontsize=10)
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    # Print prediction details
    print(f"\nImage {i+1}/{len(original_images)}: {os.path.basename(image_paths[i])}")
    print(f"Prediction: {pred_label} | Confidence: {confidence:.1f}%")

# Print summary statistics
print("\n=== Summary ===")
print(f"Total images processed: {len(original_images)}")
print(f"Total potholes detected: {sum(predicted_classes == 1)}")
print(f"Total plain roads detected: {sum(predicted_classes == 0)}")
print(f"Detection rate: {(sum(predicted_classes == 1) / len(original_images)) * 100:.1f}%")
