import os
import cv2
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
from flask import Flask, request, render_template, redirect, url_for

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Ensure the uploads folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Path to dataset directory
dataset_dir = r'C:\Atharva\Study Material\Project\Dataset'

# Metadata file path
metadata_file = os.path.join(dataset_dir, 'HAM10000_metadata.csv')

# Load metadata
print("Loading metadata...")
metadata = pd.read_csv(metadata_file)
print("Metadata loaded. Number of records:", len(metadata))

# Map image IDs to their labels
id_to_label = dict(zip(metadata['image_id'], metadata['dx']))
print("Image IDs mapped to labels.")

# Preprocess function
img_size = (64, 64)  # Example size, adjust as needed

def preprocess_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        print(f"Warning: Unable to read image at path: {img_path}")
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    img = cv2.resize(img, img_size)  # Resize image
    img = img.flatten()  # Flatten image into a vector
    return img

# Load and preprocess all images
images = []
labels = []

print("Processing images...")
# Iterate over all images in the directory
for img_file in os.listdir(dataset_dir):
    if img_file.endswith('.jpg'):
        img_path = os.path.join(dataset_dir, img_file)
        img_id = img_file.split('.')[0]  # Extract image ID from filename
        img = preprocess_image(img_path)
        if img is None:
            continue  # Skip images that can't be read
        label = id_to_label.get(img_id, '')  # Get label from metadata using image ID
        if label:  # Only include images with valid labels
            images.append(img)
            labels.append(label)
        else:
            print(f"Warning: No label found for image ID: {img_id}")

print("Image processing complete. Number of processed images:", len(images))

# Check if any images were processed
if len(images) == 0:
    print("Error: No images were processed. Please check your image directory and metadata.")
    exit()

# Convert to numpy arrays
X = np.array(images)
y = np.array(labels)

# Encode labels
print("Encoding labels...")
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
print("Labels encoded. Number of unique labels:", len(label_encoder.classes_))

# Train-validation-test split
print("Splitting data...")
X_train, X_temp, y_train, y_temp = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
print("Data split complete.")

# Build and train the Decision Tree model
print("Training Decision Tree model...")
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)
print("Model training complete.")

# Evaluate the model on the validation set
print("Evaluating model on validation set...")
val_predictions = model.predict(X_val)
print("Validation Accuracy:", accuracy_score(y_val, val_predictions))
print("Validation Classification Report:\n", classification_report(y_val, val_predictions, target_names=label_encoder.classes_))

# Evaluate the model on the test set
print("Evaluating model on test set...")
test_predictions = model.predict(X_test)
print("Test Accuracy:", accuracy_score(y_test, test_predictions))
print("Test Classification Report:\n", classification_report(y_test, test_predictions, target_names=label_encoder.classes_))

# Save the model and label encoder
print("Saving model and label encoder...")
joblib.dump(model, 'dt_skin_disease_model.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')
print("Model and label encoder saved.")

# Load the model and make predictions
def predict_image(img_path):
    model = joblib.load('dt_skin_disease_model.pkl')
    label_encoder = joblib.load('label_encoder.pkl')
    new_image = preprocess_image(img_path)
    if new_image is None:
        print("Error: Image could not be preprocessed.")
        return None
    prediction = model.predict([new_image])
    predicted_label = label_encoder.inverse_transform(prediction)
    return predicted_label[0]

# Flask routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        predicted_class = predict_image(file_path)
        return render_template('result.html', predicted_class=predicted_class, image_path=file.filename)
    return redirect(url_for('index'))

if __name__ == "__main__":
    app.run(debug=True)

