# Rapid-Internship
📌 Instagram Fake Account Detection Using Machine Learning
A comprehensive machine learning project to detect fake Instagram accounts using multi-modal features including biography text, profile images, and numerical account metadata.

📂 Project Overview
This project utilizes state-of-the-art Natural Language Processing (BERT), Computer Vision (ResNet50), and Machine Learning (XGBoost) techniques to classify Instagram accounts as real or fake with high accuracy.

✅ Key Objectives
Extract meaningful insights from biography, followers/following counts, and profile pictures.

Classify Instagram accounts using XGBoost, combining textual, image, and numerical features.

Achieve high accuracy with model explainability using SHAP analysis.

📝 Dataset Details
Total Records: 5,130 Instagram profiles

Classes: Real (0), Fake (1)

Core Features:

        username
        
        followersCount, followsCount
        
        biography (including emojis)
        
        profilePicUrl
        
        joinedRecently, private, verified
        
        fake_account (target label)

⚙ Feature Engineering
Feature Type	                Methodology	                                Tools/Models
Text (Biography & Username)	BERT Embeddings (768-dim)	        bert-base-uncased, Hugging Face Transformers
Profile Images	                ResNet50 Feature Extraction	        TorchVision Pre-trained ResNet50
Numerical	                Scaled Followers & Following Counts	StandardScaler (Scikit-learn)
Metadata	                Direct inclusion of boolean features	pandas, NumPy

🏆 Model Details
   Model	                       Purpose	                        Remarks
XGBoost Classifier	        Final Classifier	        Best performance (~99% F1-score)
Random Forest, SVM, AdaBoost	Comparison Models	        Baseline benchmarking
SHAP Explainability	        Model Interpretability	        Feature importance visualization

📊 Model Performance
Metric	                          Value
Cross-Validation F1-Score	~99.6%
Test Accuracy	                 99%
Explainability	           SHAP plots highlight text + followers features
