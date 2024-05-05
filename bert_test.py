import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from transformers import BertTokenizer, BertModel
import torch

def train_and_evaluate_with_bert(df, text_column, label_column):
    # Load pre-trained BERT model and tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    # Split the data into features (X) and target labels (y)
    X = df[text_column]
    y = df[label_column]

    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define a function to obtain BERT embeddings
    def get_bert_embeddings(text):
        input_ids = tokenizer(text, return_tensors='pt', padding=True, truncation=True)['input_ids']
        with torch.no_grad():
            outputs = model(input_ids)
            embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
        return embeddings

    # Obtain BERT embeddings for training and testing data
    x_train_embeddings = [get_bert_embeddings(text) for text in x_train]
    x_test_embeddings = [get_bert_embeddings(text) for text in x_test]

    # Initialize and train the model (RandomForestClassifier as default)
    classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    classifier.fit(x_train_embeddings, y_train)

    # Predict on test set
    y_pred = classifier.predict(x_test_embeddings)

    # Evaluate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

# Example usage:
df = pd.read_csv(r'C:\Users\DELL\Desktop\FinalYear\StressDetection\merged_data_with_clean_text.csv')
accuracy = train_and_evaluate_with_bert(df, 'clean_text', 'label')
print("Accuracy:", accuracy)
