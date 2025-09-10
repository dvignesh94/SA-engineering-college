"""
Asset Links for Machine Learning Examples
=========================================

This file contains links and references to all the assets
created for the Machine Learning examples in this directory.
"""

# Dataset paths
IRIS_DATASET_PATH = "/Users/vignesh/Documents/GitHub/Generative_ai/Datasets/neural_networks/Iris.csv"

# Generated visualization files
GENERATED_FILES = {
    "iris_analysis": "/Users/vignesh/Documents/GitHub/Generative_ai/Machine Learning/iris_analysis.png",
    "confusion_matrix": "/Users/vignesh/Documents/GitHub/Generative_ai/Machine Learning/confusion_matrix.png", 
    "feature_importance": "/Users/vignesh/Documents/GitHub/Generative_ai/Machine Learning/feature_importance.png",
    "simple_iris_plot": "/Users/vignesh/Documents/GitHub/Generative_ai/Machine Learning/simple_iris_plot.png"
}

# Main example files
EXAMPLE_FILES = {
    "comprehensive_example": "/Users/vignesh/Documents/GitHub/Generative_ai/Machine Learning/iris_classification.py",
    "simple_example": "/Users/vignesh/Documents/GitHub/Generative_ai/Machine Learning/simple_iris_example.py",
    "requirements": "/Users/vignesh/Documents/GitHub/Generative_ai/Machine Learning/requirements.txt",
    "readme": "/Users/vignesh/Documents/GitHub/Generative_ai/Machine Learning/README.md"
}

# Learning objectives
LEARNING_OBJECTIVES = [
    "Understand supervised learning fundamentals",
    "Learn data exploration and visualization techniques", 
    "Compare multiple machine learning algorithms",
    "Evaluate model performance using appropriate metrics",
    "Analyze feature importance for classification",
    "Make predictions on new, unseen data",
    "Understand the complete ML pipeline from data to predictions"
]

# Algorithms covered
ALGORITHMS = [
    "Logistic Regression",
    "Decision Tree", 
    "Random Forest",
    "Support Vector Machine",
    "K-Nearest Neighbors"
]

# Key concepts taught
KEY_CONCEPTS = [
    "Supervised vs Unsupervised Learning",
    "Classification vs Regression",
    "Training vs Testing Data",
    "Overfitting vs Underfitting", 
    "Feature Engineering",
    "Model Evaluation",
    "Cross-Validation",
    "Confusion Matrix",
    "Feature Importance"
]

def get_asset_info():
    """
    Return information about all assets in the Machine Learning directory.
    """
    return {
        "dataset_path": IRIS_DATASET_PATH,
        "generated_files": GENERATED_FILES,
        "example_files": EXAMPLE_FILES,
        "learning_objectives": LEARNING_OBJECTIVES,
        "algorithms": ALGORITHMS,
        "key_concepts": KEY_CONCEPTS
    }

if __name__ == "__main__":
    info = get_asset_info()
    
    print("Machine Learning Assets Information")
    print("=" * 40)
    print()
    
    print("Dataset:")
    print(f"  Iris Dataset: {info['dataset_path']}")
    print()
    
    print("Example Files:")
    for name, path in info['example_files'].items():
        print(f"  {name}: {path}")
    print()
    
    print("Generated Visualizations:")
    for name, path in info['generated_files'].items():
        print(f"  {name}: {path}")
    print()
    
    print("Algorithms Covered:")
    for algorithm in info['algorithms']:
        print(f"  - {algorithm}")
    print()
    
    print("Key Concepts:")
    for concept in info['key_concepts']:
        print(f"  - {concept}")
