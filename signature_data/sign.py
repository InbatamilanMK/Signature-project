import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

class SignatureVerificationSystem:
    def __init__(self):
        self.model = None
        self.label_encoder = {'genuine': 1, 'forged': 0}
        
    def load_signatures(self, data_dir):
        """Load signature images from directory structure:
        data_dir/
            genuine/
                person1/
                    sample1.jpg
                    sample2.jpg
                person2/
                    ...
            forged/
                person1/
                    fake1.jpg
                    ...
        """
        features = []
        labels = []
        
        for label_type in ['genuine', 'forged']:
            label_path = os.path.join(data_dir, label_type)
            for person in os.listdir(label_path):
                person_path = os.path.join(label_path, person)
                for sample in os.listdir(person_path):
                    img_path = os.path.join(person_path, sample)
                    img = self.preprocess_signature(img_path)
                    features.append(self.extract_features(img))
                    labels.append(self.label_encoder[label_type])
        
        return np.array(features), np.array(labels)
    
    def preprocess_signature(self, img_path):
        """Basic image preprocessing"""
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (200, 100))  # Standard size
        _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        return img
    
    def extract_features(self, img):
        """Extract simple features from signature image"""
        # Horizontal and vertical projections
        h_proj = np.sum(img, axis=1) / 255
        v_proj = np.sum(img, axis=0) / 255
        
        # Center of mass
        y, x = np.where(img == 0)
        if len(x) > 0 and len(y) > 0:
            com_x = np.mean(x)
            com_y = np.mean(y)
        else:
            com_x, com_y = img.shape[1]//2, img.shape[0]//2
        
        # Aspect ratio
        aspect_ratio = img.shape[1] / img.shape[0]
        
        # Combine all features
        features = np.concatenate([
            h_proj, v_proj, 
            [com_x, com_y, aspect_ratio]
        ])
        
        return features
    
    def train_model(self, X_train, y_train):
        """Train a simple SVM classifier"""
        self.model = SVC(kernel='rbf', probability=True)
        self.model.fit(X_train, y_train)
        return self.model
    
    def evaluate_model(self, X_test, y_test):
        """Evaluate model performance"""
        if self.model is None:
            raise ValueError("Model not trained yet")
            
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model Accuracy: {accuracy:.2f}")
        return accuracy
    
    def manual_verification(self, img_path):
        """Manual verification interface"""
        img = self.preprocess_signature(img_path)
        features = self.extract_features(img)
        
        # Model prediction
        proba = self.model.predict_proba([features])[0]
        prediction = "genuine" if proba[1] > 0.5 else "forged"
        confidence = max(proba)
        
        # Display results
        plt.imshow(img, cmap='gray')
        plt.title(f"Prediction: {prediction} (Confidence: {confidence:.2f})")
        plt.axis('off')
        plt.show()
        
        print(f"\nSignature Verification Result:")
        print(f"Prediction: {prediction}")
        print(f"Confidence: {confidence:.2%}")
        print("\nPlease verify manually:")
        print("1. Compare stroke patterns")
        print("2. Check overall shape consistency")
        print("3. Look for natural variations")
        
        # Get final manual decision
        manual_decision = input("Final decision (g=genuine/f=forged): ").lower()
        while manual_decision not in ['g', 'f']:
            manual_decision = input("Invalid input. Enter 'g' or 'f': ")
            
        final_decision = "genuine" if manual_decision == 'g' else "forged"
        print(f"\nFinal Verification Result: {final_decision}")
        return final_decision

# Example usage
if __name__ == "__main__":
    # Initialize system
    svs = SignatureVerificationSystem()
    
    # Load and prepare data
    X, y = svs.load_signatures(r"D:\github project\signature_data")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # Train model
    svs.train_model(X_train, y_train)
    
    # Evaluate model
    svs.evaluate_model(X_test, y_test)
    
    # Manual verification example
    test_signature = r"D:\github project\signature_data\fake1.jpg"
    svs.manual_verification(test_signature)