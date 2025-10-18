import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import joblib
import string
import os

def download_nltk_data():
    try:
        nltk.data.find('corpora/stopwords')
        print("✓ Stopwords already downloaded")
    except LookupError:
        print("Downloading stopwords...")
        nltk.download('stopwords')

download_nltk_data()

class BetterSpamClassifier:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        self.model = MultinomialNB()
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
    
    def preprocess_text(self, text):
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = re.sub(r'\d+', '', text)
        
        words = text.split()
        words = [self.stemmer.stem(word) for word in words if word not in self.stop_words]
        
        return ' '.join(words)
    
    def train(self, emails, labels):
        processed_emails = [self.preprocess_text(email) for email in emails]
        X = self.vectorizer.fit_transform(processed_emails)
        self.model.fit(X, labels)
        
        y_pred = self.model.predict(X)
        accuracy = accuracy_score(labels, y_pred)
        
        print(f"Training Accuracy: {accuracy:.4f}")
        print(classification_report(labels, y_pred))
    
    def predict(self, email):
        processed_email = self.preprocess_text(email)
        email_vector = self.vectorizer.transform([processed_email])
        prediction = self.model.predict(email_vector)[0]
        probability = self.model.predict_proba(email_vector)[0]
        return prediction, probability

def create_smart_training_data():
    """Create realistic training data that catches tricky spam"""
    
    spam_examples = [
        # Money-related spam
        "Send me money using this link: http://bit.ly/payme",
        "Urgent: I need money, please send via PayPal",
        "Emergency cash needed, click link to help",
        "Quick loan available, apply now at our website",
        "You inherited money, claim now at this link",
        
        # Link + Urgency combinations
        "Your account will be suspended, verify now: secure-link.com",
        "Package delivery failed, track at this link immediately",
        "Security alert: login to verify at http://bank-secure.com",
        "Limited time offer: 90% off at our store, shop-now.com",
        
        # Personal + Suspicious combinations (like your example)
        "Hi mom, I need money please send via this link",
        "Hey dad, emergency situation send funds using PayPal",
        "Dear friend, urgent help needed transfer money here",
        "Hello, I'm in trouble please wire money to this account",
        
        # Phishing attempts
        "Your Netflix account expired, update payment at netflix-secure.com",
        "Amazon: unusual activity detected, verify at amazon-verify.net",
        "Bank security update required, login at bank-official.site",
        
        # More sophisticated spam
        "Congratulations you won a prize, claim at winner-gift.com",
        "Investment opportunity guaranteed returns, sign up now",
        "Work from home earning $5000 weekly, apply today",
        "Your computer has viruses, download antivirus at secure-pc.com"
    ]
    
    ham_examples = [
        # Normal personal messages
        "Hi mom, happy mother's day! Love you",
        "Hey dad, how are you doing?",
        "Hello friend, let's meet for coffee tomorrow",
        "Hi, thanks for your email I'll respond soon",
        
        # Work communications
        "Meeting scheduled for 3 PM tomorrow in conference room",
        "Please review the attached document and provide feedback",
        "Team lunch this Friday at 12:30 PM",
        "Project deadline extended to next week",
        
        # Normal online communications
        "Your package has been delivered to your doorstep",
        "Flight confirmation for your trip next month",
        "Dinner reservation confirmed for 7 PM tonight",
        "Weather forecast shows rain tomorrow bring umbrella",
        
        # Genuine money conversations
        "Can you send me the money you owe me when you get a chance",
        "I'll transfer the rent money to your account tomorrow",
        "Thanks for sending the payment for the concert tickets",
        "Let me know your PayPal so I can send you the money",
        
        # Normal link sharing
        "Check out this interesting article I found: news-site.com/article",
        "Here's the recipe I mentioned: cooking-blog.com/recipe",
        "Join our meeting using this Zoom link: zoom.us/meeting123",
        "Track your order here: amazon.com/your-orders"
    ]
    
    emails = spam_examples + ham_examples
    labels = [1] * len(spam_examples) + [0] * len(ham_examples)
    
    return emails, labels

if __name__ == "__main__":
    # Create better classifier
    classifier = BetterSpamClassifier()
    
    # Use smarter training data
    emails, labels = create_smart_training_data()
    
    print(f"Training with {len(emails)} examples ({sum(labels)} spam, {len(labels)-sum(labels)} ham)")
    
    # Train the model
    classifier.train(emails, labels)
    
    # Create model directory if needed
    model_dir = 'model'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    # Save the trained model
    model_path = os.path.join(model_dir, 'spam_classifier.pkl')
    joblib.dump(classifier, model_path)
    print(f"✅ Better model saved at: {model_path}")
    
    # Test tricky examples
    test_emails = [
        "Hi mom, Happy Mother's day send me money using this link.",
        "Congratulations you won $1000 click here to claim",
        "Meeting tomorrow at 2 PM in the main conference room",
        "Your bank account needs verification click this link",
        "Hey dad, can you send me some money for groceries?",
        "Security alert: your account will be suspended verify now",
        "Hi team, project review meeting scheduled for Friday",
        "I need emergency funds please send via PayPal immediately"
    ]
    
    print("\n" + "="*60)
    print("TESTING THE SMARTER MODEL:")
    print("="*60)
    
    for test_email in test_emails:
        prediction, probability = classifier.predict(test_email)
        spam_prob = probability[1] * 100
        
        print(f"\n📧 '{test_email}'")
        print(f"🔍 Result: {'🚩 SPAM' if prediction == 1 else '✅ NOT SPAM'}")
        print(f"📊 Spam confidence: {spam_prob:.1f}%")
        
        # Show why it made this decision
        if "money" in test_email.lower() and "link" in test_email.lower():
            print("💡 Detected: Money request + Link = Likely spam!")
        if "verify" in test_email.lower() and "link" in test_email.lower():
            print("💡 Detected: Verification + Link = Likely spam!")