# Intelligent-Modelling-for-thyroid-disorder-diagnosis
This repository contains a Streamlit-based web application designed to detect and classify thyroid disorders from medical images. The system leverages a Deep Neural Network (DNN) enhanced with a Hybrid Meta-Heuristic optimization approach and Long Short-Term Memory (LSTM) architecture for accurate classification.

Features
User authentication (Sign Up and Login) using SQLite

Secure password storage and account creation interface

Upload and analysis of thyroid-related medical images

Classification of six thyroid conditions:

thyroid_cancer

thyroid_ditis

thyroid_hyper

thyroid_hypo

thyroid_nodule

thyroid_normal

Visual display of original and segmented images

Personalized textual medical recommendations based on classification results

Audio output of predictions using text-to-speech (gTTS)

Custom background styling and animation support

Project Structure
bash
Copy
Edit
thyroid-detection-app/
├── app.py                # Main Streamlit app for classification and visualization
├── main.py               # Streamlit app with user registration and login
├── model.h5              # Pre-trained DNN+LSTM model file
├── users.db              # SQLite database (created automatically)
├── Background/
│   ├── 1.png             # Background image
│   └── 1.gif             # Home page animation
├── sample.mp3            # Generated audio output for predictions
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
Installation
Clone the Repository

bash
Copy
Edit
git clone https://github.com/yourusername/thyroid-detection-app.git
cd thyroid-detection-app
Install the Required Packages

bash
Copy
Edit
pip install -r requirements.txt
Add the Model File
Place your trained model.h5 in the root directory. The model should support classification of the following six categories:

thyroid_cancer

thyroid_ditis

thyroid_hyper

thyroid_hypo

thyroid_nodule

thyroid_normal

Run the Application
Launch the main application using:

bash
Copy
Edit
streamlit run main.py
Usage
Register a new user account or log in using an existing one.

Upload thyroid medical images through the interface.

Receive immediate classification results and recommendations.

View both the original and segmented versions of the image.

Listen to the prediction through automatically generated audio.

Technologies Used
Frontend: Streamlit

Backend: Python, SQLite

Machine Learning Framework: TensorFlow (Keras)

Image Processing: OpenCV, Pillow, NumPy

Visualization: Matplotlib

Text-to-Speech: gTTS



Acknowledgements
This application was developed as part of a research and academic initiative focusing on the application of deep learning in medical image analysis. The integration of hybrid meta-heuristic optimization and LSTM is aimed at enhancing diagnostic performance for thyroid disease classification.

