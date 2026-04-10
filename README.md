# GAN-Based Model for Real-Time Detection of Synthetic Audio

This project focuses on detecting **synthetic and manipulated (deepfake) audio** in real time using **deep learning and Generative Adversarial Networks (GANs)**. With the rapid rise of AI-generated audio, it has become important to distinguish between real and fake audio to prevent misuse in security-sensitive applications.

The system extracts meaningful audio features and applies multiple neural network models to accurately classify audio as **real or synthetic**.

---

## 📌 Problem Statement

Advancements in AI have made it easy to generate realistic synthetic audio that can be misused for fraud, impersonation, and misinformation. Traditional audio detection techniques often fail to identify such manipulations effectively.

This project aims to build a **real-time system** that can reliably detect synthetic audio using deep learning techniques.

---

## 🎯 Objectives

- Detect synthetic (AI-generated) audio in real time  
- Use GAN-based techniques for improved detection accuracy  
- Extract meaningful audio features using MFCCs and spectrograms  
- Compare performance of different neural network models  
- Provide a simple interface for real-time prediction  

---

## 🧠 Methodology

1. **Audio Preprocessing**
   - Audio files are preprocessed to remove noise and standardize input  
   - Features such as **MFCCs** and **spectrograms** are extracted  

2. **Data Augmentation**
   - GANs are used to generate synthetic audio samples to improve training data diversity  

3. **Model Training**
   - Multiple models are trained and evaluated:
     - Artificial Neural Network (ANN)
     - Convolutional Neural Network (CNN)
     - Recurrent Neural Network (RNN)

4. **Classification**
   - Audio is classified into **Real** or **Synthetic** categories  

5. **Real-Time Detection**
   - The trained model is integrated into a web application for live predictions  

---

## 🛠️ Technologies Used

- **Programming Language:** Python  
- **Deep Learning:** TensorFlow, Keras  
- **Models:** GAN, CNN, ANN, RNN  
- **Audio Processing:** Librosa (MFCC, Spectrogram)  
- **Web Framework:** Flask  
- **Frontend:** HTML, CSS, JavaScript  
- **Tools:** VS Code  

---

## 📊 Results

- Achieved reliable accuracy in detecting synthetic audio  
- CNN and RNN models performed better on spectrogram-based features  
- GAN-based augmentation improved model generalization  
- Real-time predictions successfully displayed through the web interface  

---

## 🚀 Features

- Real-time audio input and prediction  
- Multiple model comparison  
- Visual display of prediction results  
- Scalable and modular architecture  

---

## 📁 Project Structure

├── data/
│ ├── real_audio/
│ └── synthetic_audio/
├── models/
│ ├── cnn_model.h5
│ ├── ann_model.h5
│ └── rnn_model.h5
├── preprocessing/
│ ├── feature_extraction.py
│ └── audio_preprocessing.py
├── app.py
├── templates/
├── static/
├── requirements.txt
└── README.md


---

## ▶️ How to Run the Project

1. Clone the repository  
   ```bash
   git clone https://github.com/MansiRana3/gan-synthetic-audio-detection.git
Install dependencies

pip install -r requirements.txt
Run the application

python app.py
Open browser and go to

http://127.0.0.1:5000
