from flask import Flask, render_template, request, jsonify
import joblib
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
import string

# Download NLTK data if needed
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Define the SpamClassifier class (MUST be the same as when trained)
class BetterSpamClassifier:
    def __init__(self):
        self.vectorizer = None
        self.model = None
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
    
    def preprocess_text(self, text):
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove numbers
        text = re.sub(r'\d+', '', text)
        
        # Tokenize
        words = text.split()
        
        # Remove stopwords and stem
        words = [self.stemmer.stem(word) for word in words if word not in self.stop_words]
        
        return ' '.join(words)
    
    def predict(self, email):
        # Preprocess the email
        processed_email = self.preprocess_text(email)
        
        # Transform using trained vectorizer
        email_vector = self.vectorizer.transform([processed_email])
        
        # Predict
        prediction = self.model.predict(email_vector)[0]
        probability = self.model.predict_proba(email_vector)[0]
        
        return prediction, probability

app = Flask(__name__)

# Load the trained model
classifier = None
model_path = 'model/spam_classifier.pkl'

print("🚀 Starting Flask App...")
print(f"🔍 Looking for model at: {model_path}")

try:
    if os.path.exists(model_path):
        print("✅ Model file found! Loading...")
        classifier = joblib.load(model_path)
        print("✅ Model loaded successfully!")
    else:
        print(f"❌ Model file not found at: {model_path}")
        # Show available files for debugging
        if os.path.exists('model'):
            print(f"📁 Files in model directory: {os.listdir('model')}")
        
except Exception as e:
    print(f"❌ Error loading model: {e}")
    classifier = None

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    confidence = None
    email_text = ""
    
    print(f"📨 Request method: {request.method}")
    print(f"🤖 Classifier available: {classifier is not None}")
    
    if request.method == 'POST':
        email_text = request.form.get('email', '')
        print(f"📧 Received email text: '{email_text}'")
        
        if classifier is None:
            result = "Model not loaded. Please train the model first."
            print("❌ Classifier is None!")
        elif email_text.strip():
            try:
                prediction, probability = classifier.predict(email_text)
                
                # Get confidence percentage
                spam_confidence = probability[1] * 100
                ham_confidence = probability[0] * 100
                
                result = "SPAM" if prediction == 1 else "NOT SPAM"
                confidence = spam_confidence if prediction == 1 else ham_confidence
                
                print(f"✅ Classification result: {result}")
                print(f"📊 Confidence: {confidence:.2f}%")
                print(f"🎯 Probabilities: Spam: {probability[1]:.4f}, Ham: {probability[0]:.4f}")
                
            except Exception as e:
                result = f"Error in classification: {str(e)}"
                print(f"❌ Classification error: {e}")
        else:
            result = "Please enter some email text"
            print("❌ Empty email text")
    
    return render_template('index.html', 
                         result=result, 
                         confidence=confidence,
                         email_text=email_text)

@app.route('/status')
def status():
    """Check model status"""
    return {
        'model_loaded': classifier is not None,
        'model_path': model_path,
        'model_exists': os.path.exists(model_path)
    }

if __name__ == '__main__':
    app.run(debug=True)