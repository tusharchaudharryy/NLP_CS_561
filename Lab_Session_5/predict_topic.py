import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def preprocess_text(text):
    """
    Cleans and tokenizes text. This function MUST be identical
    to the one used during training to ensure consistency.
    """
    stop_words = set(stopwords.words('english'))
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = word_tokenize(text)
    return " ".join([word for word in tokens if word not in stop_words and len(word) > 2])

def predict_sentence_topic(sentence):
    """
    Loads the trained vectorizer and classifiers to predict the topic of a sentence.
    """
    print(f"Analyzing new sentence: '{sentence}'")
    print("-" * 30)
    
    try:
        with open('tfidf_vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        with open('logistic_regression_model.pkl', 'rb') as f:
            lr_model = pickle.load(f)
        with open('svm_model.pkl', 'rb') as f:
            svm_model = pickle.load(f)
        print("Successfully loaded vectorizer and models.")
    except FileNotFoundError as e:
        print(f"ERROR: A required .pkl file was not found: {e}")
        print("Please make sure this script is in the same folder as your saved model files.")
        return

    processed_sentence = preprocess_text(sentence)
    sentence_vector = vectorizer.transform([processed_sentence])
    
    lr_prediction = lr_model.predict(sentence_vector)
    svm_prediction = svm_model.predict(sentence_vector)
    
    print("\n Prediction Results ")
    print(f"Logistic Regression Prediction: {lr_prediction[0].upper()}")
    print(f"SVM Prediction: {svm_prediction[0].upper()}")

if __name__ == "__main__":
    
    new_sentence_to_test = "The History Boys by Alan Bennett has been named best new play in the Critics' Circle Theatre Awards."
    predict_sentence_topic(new_sentence_to_test)