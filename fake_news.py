import pandas as pd
import string
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from gensim.models import Word2Vec, Doc2Vec, FastText
from gensim.models.doc2vec import TaggedDocument
import numpy as np
from transformers import BertTokenizer, BertModel
import torch
import xgboost as xgb
import os
import joblib
import matplotlib.pyplot as plt
from datasets import load_dataset

# Load dataset
data = load_dataset('GonzaloA/fake_news')
df = pd.DataFrame(data['train'])

# Preprocessing

def preprocess_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    return text

df['text'] = df['text'].apply(preprocess_text)

# Feature Extraction Methods

# TF-IDF
vectorizer_tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
X_tfidf = vectorizer_tfidf.fit_transform(df['text']).toarray()

# Count Vectorizer
vectorizer_count = CountVectorizer(max_features=5000, stop_words='english')
X_count = vectorizer_count.fit_transform(df['text']).toarray()

# Word2Vec
sentences = [text.split() for text in df['text']]
word2vec_model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)
X_word2vec = np.array([np.mean([word2vec_model.wv[word] for word in words if word in word2vec_model.wv] or [np.zeros(100)], axis=0) for words in sentences])

# FastText
fasttext_model = FastText(sentences, vector_size=100, window=5, min_count=1, workers=4)
X_fasttext = np.array([np.mean([fasttext_model.wv[word] for word in words if word in fasttext_model.wv] or [np.zeros(100)], axis=0) for words in sentences])

# Doc2Vec
documents = [TaggedDocument(words, [i]) for i, words in enumerate(sentences)]
doc2vec_model = Doc2Vec(documents, vector_size=100, window=5, min_count=1, workers=4, epochs=20)
X_doc2vec = np.array([doc2vec_model.infer_vector(words) for words in sentences])

# BERT Embeddings
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert_model.to(device)

def get_bert_embeddings_batch(texts, batch_size=16):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        with torch.no_grad():
            outputs = bert_model(input_ids=input_ids, attention_mask=attention_mask)
        embeddings.extend(outputs.last_hidden_state[:, 0, :].cpu().numpy())
    return np.array(embeddings)

if os.path.exists("bert_embeddings.npy"):
    X_bert = np.load("bert_embeddings.npy")
else:
    X_bert = get_bert_embeddings_batch(df['text'].tolist(), batch_size=16)
    np.save("bert_embeddings.npy", X_bert)

# Labels
y = df['label']

# Train-Test Split
vectorization_methods = {
    "TF-IDF": X_tfidf,
    "CountVectorizer": X_count,
    "Word2Vec": X_word2vec,
    "FastText": X_fasttext,
    "Doc2Vec": X_doc2vec,
    "BERT": X_bert
}

models = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=200, random_state=42),
    "XGBoost": xgb.XGBClassifier(eval_metric='logloss', random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42)
}

# Evaluate Models
def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
    print(f"\n{model_name} Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Visualize and Save Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix: {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(f'confusion_matrix_{model_name}.png')  # Save figure
    plt.show(block=False)
    plt.pause(3)
    plt.close()


results = {}
for method_name, X in vectorization_methods.items():
    print(f"Evaluating Vectorization Method: {method_name}")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    results[method_name] = {}
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        results[method_name][model_name] = acc
        print(f"{model_name} using {method_name}: Accuracy = {acc:.4f}")
        evaluate_model(model, X_test, y_test, f"{model_name}_{method_name}")

# Plot Model Comparison
plt.figure(figsize=(12, 6))
for method_name, model_results in results.items():
    plt.plot(model_results.keys(), model_results.values(), marker='o', label=method_name)
plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.title("Comparison of Text Embedding Methods")
plt.legend()
plt.xticks(rotation=45)
plt.show()

# Save Models
for model_name, model in models.items():
    joblib.dump(model, f"{model_name.replace(' ', '_').lower()}_model.pkl")
