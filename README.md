##Thyroid Disorder Detection and Classification Using Deep CNN
This project presents an intelligent system for classifying thyroid disorders using deep convolutional neural networks (CNNs). It provides a user-friendly Streamlit web interface and employs DenseNet121 for image-based classification of six types of thyroid conditions.

##Features
📂 Image classification across 6 thyroid categories:

Thyroid Cancer

Thyroiditis

Hyperthyroidism

Hypothyroidism

Thyroid Nodule

Normal

🧠 Deep Learning model using DenseNet121

🖼️ Otsu's method for segmentation

📊 Performance metrics: Accuracy, Confusion Matrix, and Classification Report

🗂️ Model training, evaluation, and prediction on custom images

🌐 Streamlit-based web UI with login/signup

🔐 SQLite-based user authentication system

🛠️ Tech Stack
Python 3.x

TensorFlow & Keras

OpenCV

Matplotlib

Scikit-learn

Streamlit

SQLite

NumPy

📁 Folder Structure
├── Dataset/
│   ├── thyroid_cancer/
│   ├── thyroid_ditis/
│   └── ...
├── model.py             # Model building and training
├── Proposed.py          # Image classification with DenseNet + UI logic
├── app.py               # Streamlit app with authentication
├── model.h5             # Trained model
└── Background/          # Background images and GIFs
🧪 Getting Started
1. Clone the repository
git clone https://github.com/your-username/thyroid-classification.git
cd thyroid-classification
2. Install dependencies
pip install -r requirements.txt
3. Run the application
streamlit run app.py
📸 Sample Results
Model Accuracy: ~X% (based on training)

Loss/Accuracy plots generated after training

Real-time prediction on uploaded thyroid images
