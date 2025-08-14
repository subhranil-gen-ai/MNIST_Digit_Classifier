# MNIST_Digit_Classifier
Image classification using the MNIST dataset — built with Scikit-Learn. A foundational machine learning project focusing on supervised classification

🔢 MNIST Digit Classification
This project demonstrates the classification of handwritten digits using the MNIST dataset.

# 📂 Contents
- mnist_digit_classification.ipynb: Full Colab notebook covering model training, evaluation, and visualization
- other_metrics.ipynb: Full Colab notebook with updates- Other Performance Metrics: Confusion Matrix,Precision,Recall,F1 Score.
- decision_function.ipynb- Full Colab notebook with updates- Demonstrating Decision Function, Custom Decision Thresholds, Precision-Recall Curves with visualizations, and performance trade-offs.
- roc_auc_random_forest_classifier.ipynb- Full Colab notebook with updates- ROC Curve, ROC AUC Score, and visual representations for both SGD Classifier and Random Forest Classifier models, implementing Multi-output Binary Classification using K-Neighbors Classifier.

#  Features
- Dataset loading using fetch_openml("mnist_784") from Scikit-learn
- Data reshaping for model readiness
- Data normalization for faster and better model performance
- Train-Test Split using train_test_split from Scikit-learn
- Model Training & Evaluation with:
    - SGDClassifier (Stochastic Gradient Descent for large-scale learning)
    - Random Forest Classifier
    - Multi-output Binary Classification using K-Neighbors Classifier
- Custom Cross-Validation:
    - Manual implementation of Stratified K-Fold CV
    - Accuracy evaluation on each fold
- Metrics Used:
    - Accuracy Score
    - Confusion Matrix
    - Precision Score
    - Recall Score
    - F1 Score
- Decision Function & Threshold Tuning:
  - Using decision_function() for score-based classification
  - Implementing custom decision thresholds to optimize for precision or recall
  - Visualization of how thresholds affect performance metrics
- Precision-Recall Analysis:
  - Precision-Recall Curves and area under curve
  - Visual representations for understanding precision-recall trade-offs
- Advanced Evaluation Metrics:
  - ROC Curve & ROC AUC Score (SGD & Random Forest)
  - Visualization of ROC Curves for multiple models
  - predict_proba() method usage for probabilistic outputs
- Extended Decision Analysis:
    - Custom decision thresholds with ROC and Precision-Recall visualizations
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
- How to calculate and interpret model performance scores (e.g., accuracy, precision, recall, F1-score, ROC AUC)
- How to use decision_function() for detailed model output scores
- How to adjust classification thresholds for specific goals (e.g., higher precision or recall)
- How to generate and interpret Precision-Recall Curves
- How to visualize the relationship between threshold choice and model performance
- How to interpret and visualize ROC Curves & AUC scores
- How to implement and evaluate ensemble methods like Random Forest
- How to use predict_proba() for probabilistic classification
- How to handle and evaluate multi-output binary classification tasks

#  Author
Subhranil Dutta
CSE | GenAI & DSA Learner | Python Developer  
🔗 [GitHub Profile](https://github.com/subhranil-gen-ai)
