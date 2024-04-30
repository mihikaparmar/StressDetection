import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from transformers import BertTokenizer, BertModel
import torch

def get_bert_embeddings(texts):
    # Load pre-trained BERT model and tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    # Define a function to obtain BERT embeddings for a single text
    def get_single_bert_embedding(text):
        input_ids = tokenizer(text, return_tensors='pt', padding=True, truncation=True)['input_ids']
        with torch.no_grad():
            outputs = model(input_ids)
            embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
        return embeddings

    # Obtain BERT embeddings for all texts
    embeddings_list = [get_single_bert_embedding(text) for text in texts]
    return embeddings_list
