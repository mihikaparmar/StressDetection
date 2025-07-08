# Import necessary libraries
import os
import pytesseract
from PIL import Image
from deepface import DeepFace
from transformers import DistilBertTokenizer, DistilBertModel
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Path setup for dataset and label file
# 'twitterDataset' contains image partitions and 'groundTruthLabel.txt' contains image-label mapping
dataset_dir = 'twitterDataset'
label_file = os.path.join(dataset_dir, 'groundTruthLabel.txt')

# Read labels from the groundTruthLabel.txt file
def read_labels(label_file):
    labels = {}
    with open(label_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                img, label = parts
                labels[img] = label  # Map image filename to its label
    return labels

# Extract text from an image using OCR (Tesseract)
def extract_text(image_path):
    try:
        img = Image.open(image_path)
        text = pytesseract.image_to_string(img)
        return text.strip()  # Return extracted text
    except Exception as e:
        return ''  # Return empty string if OCR fails

# Extract facial features from an image using DeepFace (Facenet model)
def extract_face_features(image_path):
    try:
        result = DeepFace.represent(img_path=image_path, model_name='Facenet', enforce_detection=False)
        if isinstance(result, list) and len(result) > 0:
            return np.array(result[0]['embedding'])  # Return facial embedding
        else:
            return np.zeros(128)  # Return zero vector if no face found
    except Exception as e:
        return np.zeros(128)  # Return zero vector if extraction fails

# Generate DistilBERT embedding for the given text
def get_text_embedding(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=64)
    with torch.no_grad():
        outputs = model(**inputs)
        return outputs.last_hidden_state[:,0,:].squeeze().numpy()  # Use [CLS] token embedding

# Main pipeline function
def main():
    labels = read_labels(label_file)  # Load image-label mapping
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')  # Load DistilBERT tokenizer
    bert_model = DistilBertModel.from_pretrained('distilbert-base-uncased')    # Load DistilBERT model
    X, y = [], []  # Feature and label lists
    # Iterate through each partition in the dataset
    for partition in os.listdir(dataset_dir):
        part_path = os.path.join(dataset_dir, partition)
        if os.path.isdir(part_path):
            for img_file in os.listdir(part_path):
                if img_file.endswith('.jpg'):
                    img_path = os.path.join(part_path, img_file)
                    text = extract_text(img_path)  # Extract text from image
                    text_emb = get_text_embedding(text, tokenizer, bert_model)  # Get text embedding
                    face_emb = extract_face_features(img_path)  # Get facial features
                    features = np.concatenate([text_emb, face_emb])  # Combine text and face features
                    X.append(features)
                    # Label: 1 for positive, 0 for negative (default to negative if not found)
                    y.append(1 if labels.get(img_file, 'negative') == 'positive' else 0)
    X = np.array(X)
    y = np.array(y)
    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = LogisticRegression(max_iter=1000)  # Initialize classifier
    clf.fit(X_train, y_train)  # Train classifier
    y_pred = clf.predict(X_test)  # Predict on test set
    acc = accuracy_score(y_test, y_pred)  # Calculate accuracy
    print(f'Accuracy: {acc:.4f}')  # Print accuracy

if __name__ == '__main__':
    main()  # Run the pipeline
