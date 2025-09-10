"""
Iris Flower Classification - Supervised Learning Example for College Students
================================================================================

This example demonstrates supervised learning using the famous Iris dataset.
We'll use multiple algorithms to classify iris flowers into three species:
- Iris-setosa
- Iris-versicolor  
- Iris-virginica

The dataset contains 4 features (sepal length, sepal width, petal length, petal width)
and we'll predict the species based on these measurements.

Author: Educational Example
Date: 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class IrisClassifier:
    """
    A comprehensive class for Iris flower classification using multiple algorithms.
    Perfect for learning supervised machine learning concepts!
    """
    
    def __init__(self, data_path):
        """
        Initialize the classifier with the dataset path.
        
        Args:
            data_path (str): Path to the Iris CSV file
        """
        self.data_path = data_path
        self.data = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.results = {}
        
    def load_and_explore_data(self):
        """
        Load the dataset and perform basic exploration.
        This is the first step in any machine learning project!
        """
        print("🌺 Loading Iris Dataset...")
        print("=" * 50)
        
        # Load the data
        self.data = pd.read_csv(self.data_path)
        
        # Display basic information
        print(f"Dataset shape: {self.data.shape}")
        print(f"Features: {list(self.data.columns[:-1])}")  # All except Species
        print(f"Target variable: {self.data.columns[-1]}")
        print()
        
        # Display first few rows
        print("First 5 rows of the dataset:")
        print(self.data.head())
        print()
        
        # Check for missing values
        print("Missing values:")
        print(self.data.isnull().sum())
        print()
        
        # Basic statistics
        print("Dataset statistics:")
        print(self.data.describe())
        print()
        
        # Check class distribution
        print("Species distribution:")
        print(self.data['Species'].value_counts())
        print()
        
    def visualize_data(self):
        """
        Create beautiful visualizations to understand the data better.
        Visualization is crucial for understanding patterns in data!
        """
        print("📊 Creating Data Visualizations...")
        print("=" * 50)
        
        # Set up the plotting area
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Iris Dataset Analysis', fontsize=16, fontweight='bold')
        
        # 1. Pairplot to see relationships between features
        plt.subplot(2, 2, 1)
        # Create a simplified pairplot for one pair of features
        sns.scatterplot(data=self.data, x='SepalLengthCm', y='SepalWidthCm', 
                       hue='Species', s=60, alpha=0.8)
        plt.title('Sepal Length vs Sepal Width')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 2. Petal measurements
        plt.subplot(2, 2, 2)
        sns.scatterplot(data=self.data, x='PetalLengthCm', y='PetalWidthCm', 
                       hue='Species', s=60, alpha=0.8)
        plt.title('Petal Length vs Petal Width')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 3. Feature distributions
        plt.subplot(2, 2, 3)
        feature_cols = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
        self.data[feature_cols].boxplot(ax=plt.gca())
        plt.title('Feature Distributions')
        plt.xticks(rotation=45)
        
        # 4. Species distribution
        plt.subplot(2, 2, 4)
        species_counts = self.data['Species'].value_counts()
        colors = ['#ff9999', '#66b3ff', '#99ff99']
        plt.pie(species_counts.values, labels=species_counts.index, autopct='%1.1f%%', 
                colors=colors, startangle=90)
        plt.title('Species Distribution')
        
        plt.tight_layout()
        plt.savefig('/Users/vignesh/Documents/GitHub/Generative_ai/Machine Learning/iris_analysis.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Visualizations saved as 'iris_analysis.png'")
        print()
        
    def prepare_data(self):
        """
        Prepare the data for machine learning.
        This includes separating features from target and encoding labels.
        """
        print("🔧 Preparing Data for Machine Learning...")
        print("=" * 50)
        
        # Separate features (X) and target (y)
        # We'll use all columns except 'Id' and 'Species'
        feature_columns = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
        self.X = self.data[feature_columns]
        self.y = self.data['Species']
        
        print(f"Features shape: {self.X.shape}")
        print(f"Target shape: {self.y.shape}")
        print()
        
        # Encode the target variable (convert text to numbers)
        # This is necessary because most ML algorithms work with numbers
        label_encoder = LabelEncoder()
        self.y_encoded = label_encoder.fit_transform(self.y)
        
        print("Species encoding:")
        for i, species in enumerate(label_encoder.classes_):
            print(f"  {species} -> {i}")
        print()
        
        # Split the data into training and testing sets
        # This is crucial for evaluating our model's performance
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y_encoded, test_size=0.3, random_state=42, stratify=self.y_encoded
        )
        
        print(f"Training set size: {self.X_train.shape[0]} samples")
        print(f"Testing set size: {self.X_test.shape[0]} samples")
        print()
        
    def train_models(self):
        """
        Train multiple machine learning models.
        We'll compare different algorithms to see which works best!
        """
        print("🤖 Training Multiple Machine Learning Models...")
        print("=" * 50)
        
        # Define the models we want to try
        self.models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Decision Tree': DecisionTreeClassifier(random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Support Vector Machine': SVC(random_state=42),
            'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=3)
        }
        
        # Train each model
        for name, model in self.models.items():
            print(f"Training {name}...")
            model.fit(self.X_train, self.y_train)
            print(f"✅ {name} training completed!")
        
        print()
        
    def evaluate_models(self):
        """
        Evaluate all trained models and compare their performance.
        This helps us choose the best model for our problem!
        """
        print("📈 Evaluating Model Performance...")
        print("=" * 50)
        
        # Create a results dictionary to store performance metrics
        self.results = {}
        
        for name, model in self.models.items():
            # Make predictions on the test set
            y_pred = model.predict(self.X_test)
            
            # Calculate accuracy
            accuracy = accuracy_score(self.y_test, y_pred)
            self.results[name] = accuracy
            
            print(f"{name}:")
            print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
            print()
        
        # Find the best model
        best_model_name = max(self.results, key=self.results.get)
        best_accuracy = self.results[best_model_name]
        
        print("🏆 BEST MODEL:")
        print(f"  {best_model_name} with {best_accuracy:.4f} accuracy ({best_accuracy*100:.2f}%)")
        print()
        
        return best_model_name
        
    def detailed_analysis(self, model_name):
        """
        Perform detailed analysis of the best performing model.
        """
        print(f"🔍 Detailed Analysis of {model_name}...")
        print("=" * 50)
        
        model = self.models[model_name]
        y_pred = model.predict(self.X_test)
        
        # Classification report
        print("Classification Report:")
        print(classification_report(self.y_test, y_pred, 
                                  target_names=['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']))
        print()
        
        # Confusion Matrix
        print("Confusion Matrix:")
        cm = confusion_matrix(self.y_test, y_pred)
        print(cm)
        print()
        
        # Visualize confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'],
                   yticklabels=['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'])
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('/Users/vignesh/Documents/GitHub/Generative_ai/Machine Learning/confusion_matrix.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Confusion matrix saved as 'confusion_matrix.png'")
        print()
        
    def feature_importance_analysis(self):
        """
        Analyze which features are most important for classification.
        This helps us understand what makes each species unique!
        """
        print("🎯 Feature Importance Analysis...")
        print("=" * 50)
        
        # Get feature importance from Random Forest (it provides this information)
        rf_model = self.models['Random Forest']
        feature_names = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']
        importances = rf_model.feature_importances_
        
        # Create a DataFrame for better visualization
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        print("Feature Importance (from Random Forest):")
        for _, row in importance_df.iterrows():
            print(f"  {row['Feature']}: {row['Importance']:.4f}")
        print()
        
        # Visualize feature importance
        plt.figure(figsize=(10, 6))
        sns.barplot(data=importance_df, x='Importance', y='Feature', palette='viridis')
        plt.title('Feature Importance for Iris Classification')
        plt.xlabel('Importance Score')
        plt.tight_layout()
        plt.savefig('/Users/vignesh/Documents/GitHub/Generative_ai/Machine Learning/feature_importance.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Feature importance plot saved as 'feature_importance.png'")
        print()
        
    def make_predictions(self, sepal_length, sepal_width, petal_length, petal_width):
        """
        Make predictions on new data using the best model.
        This shows how to use the trained model for real predictions!
        """
        print("🔮 Making Predictions on New Data...")
        print("=" * 50)
        
        # Find the best model
        best_model_name = max(self.results, key=self.results.get)
        best_model = self.models[best_model_name]
        
        # Create new data point
        new_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        
        # Make prediction
        prediction = best_model.predict(new_data)[0]
        prediction_proba = best_model.predict_proba(new_data)[0]
        
        # Convert back to species name
        species_names = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
        predicted_species = species_names[prediction]
        
        print(f"Input measurements:")
        print(f"  Sepal Length: {sepal_length} cm")
        print(f"  Sepal Width: {sepal_width} cm")
        print(f"  Petal Length: {petal_length} cm")
        print(f"  Petal Width: {petal_width} cm")
        print()
        print(f"Predicted Species: {predicted_species}")
        print()
        print("Prediction Probabilities:")
        for i, species in enumerate(species_names):
            print(f"  {species}: {prediction_proba[i]:.4f} ({prediction_proba[i]*100:.2f}%)")
        print()
        
        return predicted_species
        
    def run_complete_analysis(self):
        """
        Run the complete machine learning pipeline.
        This is the main method that orchestrates everything!
        """
        print("🚀 Starting Complete Iris Classification Analysis")
        print("=" * 60)
        print()
        
        # Step 1: Load and explore data
        self.load_and_explore_data()
        
        # Step 2: Visualize data
        self.visualize_data()
        
        # Step 3: Prepare data
        self.prepare_data()
        
        # Step 4: Train models
        self.train_models()
        
        # Step 5: Evaluate models
        best_model = self.evaluate_models()
        
        # Step 6: Detailed analysis
        self.detailed_analysis(best_model)
        
        # Step 7: Feature importance
        self.feature_importance_analysis()
        
        # Step 8: Example predictions
        print("📝 Example Predictions:")
        print("-" * 30)
        
        # Example 1: Typical Iris-setosa
        print("Example 1:")
        self.make_predictions(5.1, 3.5, 1.4, 0.2)
        
        # Example 2: Typical Iris-versicolor
        print("Example 2:")
        self.make_predictions(6.0, 2.2, 4.0, 1.0)
        
        # Example 3: Typical Iris-virginica
        print("Example 3:")
        self.make_predictions(6.3, 3.3, 6.0, 2.5)
        
        print("🎉 Analysis Complete!")
        print("=" * 60)
        print("Key Takeaways:")
        print("1. Supervised learning uses labeled data to make predictions")
        print("2. Data visualization helps understand patterns")
        print("3. Multiple algorithms can be compared to find the best one")
        print("4. Feature importance shows which measurements matter most")
        print("5. Model evaluation ensures our predictions are reliable")
        print()


def main():
    """
    Main function to run the Iris classification example.
    """
    # Path to the Iris dataset
    data_path = "/Users/vignesh/Documents/GitHub/Generative_ai/Datasets/neural_networks/Iris.csv"
    
    # Create and run the classifier
    classifier = IrisClassifier(data_path)
    classifier.run_complete_analysis()


if __name__ == "__main__":
    main()
