# MelvinSujay-Data-Science-Project-Facial-Emotion-Recognition-Using-Deep-Learning
Facial emotion recognition is an advanced field within computer vision and artificial intelligence, where the goal is to identify human emotions based on facial expressions. This project uses deep learning techniques, particularly Convolutional Neural Networks (CNNs), to classify emotions such as happiness, sadness, anger, surprise, and others from facial images. By analyzing various facial features, such as the eyes, mouth, and overall facial expressions, the system can predict the emotional state of individuals.

Emotions Detected
Angry
Disgust
Fear
Happy
Neutral
Sad
Surprise

Features
CNN-based model trained on grayscale 48x48 images.
Real-time emotion recognition using webcam.
Static image emotion analysis.
Evaluation through accuracy/loss plots and confusion matrix.
Clean modular codebase with separate files for training, testing, and evaluation.

Technologies Used
Python
TensorFlow / Keras
OpenCV
NumPy, Pandas
Matplotlib, Seaborn
Haar Cascade Classifier for face detection

Project Structure Facial-Emotion-Recognition/ │
Data Preprocessing – Enhances training data using augmentation, normalization, and grayscale conversion.
CNN Model Training – Uses a Convolutional Neural Network (CNN) to train a model and to classify emotions into seven categories.
Model Evaluation – Measures performance using accuracy, loss, confusion matrix, and classification report.
Static Image Recognition – Predicts emotions from uploaded facial images.
Real-Time Emotion Detection – Uses a webcam to detect and classify emotions in live video.
model_file_final.h5 # Trained model
README.md
Install Requirements pip install tensorflow opencv-python matplotlib seaborn numpy

Results • Test Accuracy: 62.1% • Confusion matrix and plots saved as image files.

Applications • Human-computer interaction • Mood-based recommendation systems • Mental health monitoring • Smart surveillance systems

Dataset link: https://www.kaggle.com/datasets/msambare/fer2013
