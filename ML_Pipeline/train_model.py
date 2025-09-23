import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import cv2
from .feature_extraction import TomatoFeatureExtractor, extract_features_from_dataset
from .dataset_loader import LaboroDatasetLoader

class TomatoSVMTrainer:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = None
        self.class_names = ['unripe', 'turning', 'ripe', 'rotten']
        
    def load_or_extract_features(self, dataset_dir, features_file="features.csv"):
        """Load existing features or extract them from dataset"""
        features_path = Path(features_file)
        dataset_path = Path(dataset_dir)
        
        if features_path.exists():
            print(f"Loading existing features from {features_file}")
            features_df = pd.read_csv(features_file)
        elif dataset_path.exists():
            print(f"Extracting features from dataset at {dataset_dir}")
            features_df = extract_features_from_dataset(dataset_dir, features_file)
        else:
            # Prepare dataset first
            print("Dataset not found. Preparing dataset...")
            loader = LaboroDatasetLoader()
            dataset_dir = loader.prepare_dataset()
            if dataset_dir:
                features_df = extract_features_from_dataset(dataset_dir, features_file)
            else:
                raise ValueError("Could not prepare dataset")
        
        return features_df
    
    def prepare_data(self, features_df):
        """Prepare features and labels for training"""
        # Separate features and labels
        X = features_df.drop('label', axis=1)
        y = features_df['label']
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        # Handle missing values
        X = X.fillna(X.mean())
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=self.random_state, stratify=y_encoded
        )
        
        print(f"Training set size: {X_train.shape}")
        print(f"Test set size: {X_test.shape}")
        print(f"Feature dimensions: {X_train.shape[1]}")
        print(f"Classes: {self.label_encoder.classes_}")
        
        return X_train, X_test, y_train, y_test
    
    def create_svm_pipeline(self):
        """Create SVM pipeline with preprocessing"""
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('svm', SVC(random_state=self.random_state, probability=True))
        ])
        
        return pipeline
    
    def tune_hyperparameters(self, X_train, y_train):
        """Perform hyperparameter tuning using GridSearch"""
        print("Tuning SVM hyperparameters...")
        
        pipeline = self.create_svm_pipeline()
        
        # Parameter grid for SVM
        param_grid = {
            'svm__C': [0.1, 1, 10, 100],
            'svm__kernel': ['rbf', 'poly', 'sigmoid'],
            'svm__gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1]
        }
        
        # Perform grid search
        grid_search = GridSearchCV(
            pipeline, 
            param_grid, 
            cv=5,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_
    
    def train_model(self, dataset_dir=None, features_file="features.csv", tune_hyperparams=True):
        """Train the SVM model"""
        print("Starting SVM training pipeline...")
        
        # Load or extract features
        if dataset_dir:
            features_df = self.load_or_extract_features(dataset_dir, features_file)
        else:
            features_df = pd.read_csv(features_file)
        
        # Prepare data
        X_train, X_test, y_train, y_test = self.prepare_data(features_df)
        
        # Train model
        if tune_hyperparams:
            self.model = self.tune_hyperparameters(X_train, y_train)
        else:
            # Use default parameters
            self.model = self.create_svm_pipeline()
            self.model.fit(X_train, y_train)
        
        # Evaluate model
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)
        
        print(f"Training accuracy: {train_score:.4f}")
        print(f"Test accuracy: {test_score:.4f}")
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=5)
        print(f"Cross-validation accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Detailed evaluation
        y_pred = self.model.predict(X_test)
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=self.label_encoder.classes_))
        
        # Confusion matrix
        self.plot_confusion_matrix(y_test, y_pred)
        
        return {
            'train_accuracy': train_score,
            'test_accuracy': test_score,
            'cv_scores': cv_scores,
            'X_test': X_test,
            'y_test': y_test,
            'y_pred': y_pred
        }
    
    def plot_confusion_matrix(self, y_true, y_pred):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.label_encoder.classes_,
                   yticklabels=self.label_encoder.classes_)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    def analyze_feature_importance(self, X_train, y_train):
        """Analyze feature importance using Random Forest"""
        print("Analyzing feature importance...")
        
        # Train Random Forest for feature importance
        rf = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
        rf.fit(X_train, y_train)
        
        # Get feature importances
        importances = rf.feature_importances_
        feature_importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        # Plot top 20 features
        plt.figure(figsize=(10, 8))
        top_features = feature_importance_df.head(20)
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Feature Importance')
        plt.title('Top 20 Most Important Features')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        return feature_importance_df
    
    def predict_single_image(self, image_path):
        """Predict ripeness class for a single image"""
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")
        
        # Extract features from image
        extractor = TomatoFeatureExtractor()
        result = extractor.process_image_for_analysis(image_path)
        
        if result is None:
            return None
        
        # Prepare features
        features_dict = result['features']
        features_array = np.array([features_dict[name] for name in self.feature_names]).reshape(1, -1)
        
        # Handle missing features
        features_array = np.nan_to_num(features_array, nan=0.0)
        
        # Predict
        prediction = self.model.predict(features_array)[0]
        probabilities = self.model.predict_proba(features_array)[0]
        
        # Convert back to class names
        predicted_class = self.label_encoder.inverse_transform([prediction])[0]
        
        # Create probability dictionary
        prob_dict = {}
        for i, class_name in enumerate(self.label_encoder.classes_):
            prob_dict[class_name] = probabilities[i]
        
        return {
            'predicted_class': predicted_class,
            'probabilities': prob_dict,
            'confidence': np.max(probabilities),
            'cielab_analysis': result['cielab_analysis'],
            'spoilage_score': result['spoilage_score'],
            'spoilage_features': result['spoilage_features']
        }
    
    def save_model(self, model_path="tomato_svm_model.pkl"):
        """Save trained model"""
        if self.model is None:
            raise ValueError("No model to save. Train model first.")
        
        model_data = {
            'model': self.model,
            'label_encoder': self.label_encoder,
            'feature_names': self.feature_names,
            'class_names': self.class_names
        }
        
        joblib.dump(model_data, model_path)
        print(f"Model saved to {model_path}")
    
    def load_model(self, model_path="tomato_svm_model.pkl"):
        """Load trained model"""
        model_data = joblib.load(model_path)
        
        self.model = model_data['model']
        self.label_encoder = model_data['label_encoder']
        self.feature_names = model_data['feature_names']
        self.class_names = model_data['class_names']
        
        print(f"Model loaded from {model_path}")
    
    def compare_algorithms(self, X_train, X_test, y_train, y_test):
        """Compare SVM with other algorithms"""
        print("Comparing different algorithms...")
        
        algorithms = {
            'SVM (RBF)': Pipeline([
                ('scaler', StandardScaler()),
                ('svm', SVC(kernel='rbf', random_state=self.random_state))
            ]),
            'SVM (Linear)': Pipeline([
                ('scaler', StandardScaler()),
                ('svm', SVC(kernel='linear', random_state=self.random_state))
            ]),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=self.random_state),
            'Logistic Regression': Pipeline([
                ('scaler', StandardScaler()),
                ('lr', LogisticRegression(random_state=self.random_state, max_iter=1000))
            ])
        }
        
        results = {}
        
        for name, algorithm in algorithms.items():
            # Train
            algorithm.fit(X_train, y_train)
            
            # Evaluate
            train_acc = algorithm.score(X_train, y_train)
            test_acc = algorithm.score(X_test, y_test)
            cv_scores = cross_val_score(algorithm, X_train, y_train, cv=5)
            
            results[name] = {
                'train_accuracy': train_acc,
                'test_accuracy': test_acc,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std()
            }
            
            print(f"{name}:")
            print(f"  Train Accuracy: {train_acc:.4f}")
            print(f"  Test Accuracy: {test_acc:.4f}")
            print(f"  CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            print()
        
        return results

def main():
    """Main training pipeline"""
    trainer = TomatoSVMTrainer()
    
    # Check if we have a dataset
    dataset_dir = "./data/crops"
    if not Path(dataset_dir).exists():
        print("Dataset not found. Preparing Laboro dataset...")
        loader = LaboroDatasetLoader()
        dataset_dir = loader.prepare_dataset()
    
    if dataset_dir:
        # Train the model
        print("Training SVM model for tomato ripeness classification...")
        results = trainer.train_model(dataset_dir, tune_hyperparams=True)
        
        # Analyze features
        # Load features for analysis
        features_df = pd.read_csv("features.csv")
        X = features_df.drop('label', axis=1)
        y = features_df['label']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, trainer.label_encoder.transform(y), 
            test_size=0.2, random_state=42
        )
        
        trainer.analyze_feature_importance(X_train, y_train)
        
        # Compare algorithms
        trainer.compare_algorithms(X_train, X_test, y_train, y_test)
        
        # Save model
        trainer.save_model("tomato_svm_model.pkl")
        
        print("Training complete! Model saved as 'tomato_svm_model.pkl'")
        print(f"Final test accuracy: {results['test_accuracy']:.4f}")
        
        # Test prediction on a sample image if available
        sample_images = list(Path(dataset_dir).rglob("*.jpg"))
        if sample_images:
            test_image = sample_images[0]
            prediction = trainer.predict_single_image(str(test_image))
            if prediction:
                print(f"\nTest prediction on {test_image.name}:")
                print(f"Predicted class: {prediction['predicted_class']}")
                print(f"Confidence: {prediction['confidence']:.3f}")
                print(f"Probabilities: {prediction['probabilities']}")
    else:
        print("Could not prepare dataset. Please check your Kaggle API configuration.")

if __name__ == "__main__":
    main()