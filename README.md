This project presents a Streamlit-based web application for detecting and classifying various thyroid disorders using a deep neural network (DNN) model enhanced by Hybrid Meta-Heuristic algorithms and LSTM techniques. It features secure user authentication and an intelligent prediction interface that classifies uploaded medical images and provides spoken and visual feedback.

## Features
🔐 User Authentication (Login & Sign Up) using SQLite

🧬 Thyroid disorder prediction using a trained DNN + LSTM model

🖼️ Image preprocessing and segmentation (thresholding)

🗣️ Text-to-speech feedback for predictions (gTTS)

📊 Visualization of original and segmented images

🎨 Custom background and UI styling with embedded media

🧠 Six-class classification:

thyroid_cancer

thyroid_ditis

thyroid_hyper

thyroid_hypo

thyroid_nodule

thyroid_normal

## Project Structure

📦thyroid-detection-app/
├── app.py                # Main prediction logic with Streamlit
├── main.py               # Entry point for user authentication
├── model.h5              # Pre-trained DNN+LSTM model
├── users.db              # SQLite database for users (auto-created)
├── Background/
│   └── 1.png             # Background image for UI
│   └── 1.gif             # Animation displayed on home
├── sample.mp3            # Generated voice output for prediction
├── requirements.txt      # Python dependencies
└── README.md             # This file
🛠️ Installation

Clone the repository

git clone https://github.com/yourusername/thyroid-detection-app.git
cd thyroid-detection-app

Install dependencies

pip install -r requirements.txt
Add your model
Place the model.h5 file in the root directory. Make sure it matches the training classes:
['thyroid_cancer','thyroid_ditis','thyroid_hyper','thyroid_hypo','thyroid_nodule','thyroid_normal']

Run the app

streamlit run main.py

## Sample Usage
Sign up or log in.

Upload thyroid-related medical images.

View predictions, segmented images, and personalized medical advice.

Listen to the classification result through generated audio.

## Tech Stack
Frontend/UI: Streamlit, HTML/CSS

Backend: Python, SQLite

ML/DL Frameworks: TensorFlow/Keras

Others: OpenCV, NumPy, gTTS (Text-to-Speech), Matplotlib, Pillow

##  Model Highlights
The classification model combines DNN with LSTM layers and is optimized using Hybrid Meta-Heuristic techniques to enhance accuracy in medical image classification, especially for thyroid disease detection.


##  Acknowledgements
Developed as part of an academic project on medical image diagnosis

Inspired by real-world applications in AI-driven healthcare

