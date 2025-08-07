# MNIST_Digit_Classifier
Image classification using the MNIST dataset — built with Scikit-Learn, tuned with hyperparameters, and powered by pipelines. A foundational machine learning project focusing on supervised classification

🔢 MNIST Digit Classification
This project demonstrates the classification of handwritten digits using the MNIST dataset. It covers model training, evaluation, and sets the foundation for building machine learning pipelines and applying hyperparameter tuning.

# 📂 Contents
- mnist_digit_classification.ipynb: Full Colab notebook covering model training, evaluation, and visualization
- other_metrics.ipynb: Full Colab notebook with updates- Other Performance Metrics: Confusion Matrix,Precision,Recall,F1 Score.

#  Features
- Dataset loading using fetch_openml("mnist_784") from Scikit-learn
- Data reshaping for model readiness
- Data normalization for faster and better model performance
- Train-Test Split using train_test_split from Scikit-learn
- Model Training & Evaluation with:
    - SGDClassifier (Stochastic Gradient Descent for large-scale learning)
- Custom Cross-Validation:
    - Manual implementation of Stratified K-Fold CV
    - Accuracy evaluation on each fold
- Metrics Used:
    - Accuracy Score
    - Confusion Matrix
    - Precision Score
    - Recall Score
    - F1 Score
- Visualizations:
    - Plotting sample digit images using matplotlib
- Insightful experiments for understanding performance tradeoffs

# 📊 Technologies Used
- Python 3
- NumPy
- Matplotlib
- Scikit-learn
- Google Colab

# 🧠 What I Learned
- How to work with image classification datasets
- How to preprocess image data (reshaping, scaling)
- How to train and evaluate a classification model

#  Author
Subhranil Dutta
CSE | GenAI & DSA Learner | Python Developer  
🔗 [GitHub Profile](https://github.com/subhranil-gen-ai)
