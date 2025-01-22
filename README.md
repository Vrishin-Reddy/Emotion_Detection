
# Emotion Recognition Chatbot

This project, **"Text-Based Mood Detection Using Machine Learning and Emotion Identification"**, analyzes and predicts emotions from text inputs using advanced machine learning techniques. The project includes several components, such as Jupyter notebooks, a Streamlit-based web application, and a `requirements.txt` file for dependencies.



## Table of Contents
- [About the Project](#about-the-project)
- [Key Features](#key-features)
- [Dataset Information](#dataset-information)
- [File Descriptions](#file-descriptions)
- [Setup Instructions](#setup-instructions)
- [How to Use](#how-to-use)
- [System Requirements](#system-requirements)
- [Authors](#authors)
- [Acknowledgments](#acknowledgments)

---

## About the Project
This project uses deep learning techniques such as **BERT embeddings** for text feature extraction, alongside traditional machine learning models (e.g., **Lasso Regression**) for emotion detection. The web interface provides users with predictions and interactive visualizations.


## Key Features
- **Emotion Detection:** Classifies text into emotions such as Joy, Sadness, Anger, Fear, Love, and Surprise.
- **BERT Model:** Fine-tuned BERT for superior accuracy.
- **Interactive Interface:** Real-time predictions and visualizations via Streamlit.
- **Comprehensive Analysis:** Detailed evaluation metrics and cross-validation results.



## Dataset Information
The dataset used for training and evaluating the emotion detection models is the **[Emotion Dataset by dair.ai](https://huggingface.co/datasets/dair-ai/emotion)**, available on Hugging Face. 

### Key Features of the Dataset:
- **Text Inputs:** A collection of sentences and short texts.
- **Emotion Labels:** Six emotions - Joy, Sadness, Anger, Fear, Love, and Surprise.
- **Use Case:** Ideal for text classification tasks, including emotion detection.
- **License:** Open for research and non-commercial use.

For more details, visit the [dataset page](https://huggingface.co/datasets/dair-ai/emotion).

---

## File Descriptions

### 1. **8_Mod.ipynb**
Contains preprocessing, training, and evaluation of baseline machine learning models like SVM and Naïve Bayes. Includes feature extraction using TF-IDF.

### 2. **BERT_Mod.ipynb**
Focuses on the fine-tuning and evaluation of BERT-based models for emotion detection. Includes hyperparameter tuning and performance analysis.

### 3. **app_Streamlit.py**
A Streamlit application for deploying the emotion recognition system. Key features include:  
- Predicting emotions from user-provided text.  
- Visualizing model insights such as cross-validation results and metrics comparisons.

### 4. **requirements.txt**
Lists all required dependencies for the project, including libraries for machine learning, data preprocessing, visualization, and the web interface.

---

## Setup Instructions

### Prerequisites
1. Python 3.10 or higher.
2. NVIDIA GPU for training (optional but recommended).

### Installation Steps
1. Download the project files and navigate to the directory where they are stored.

2. Install dependencies:
   ```command prompt
   pip install -r requirements.txt
   ```

3. Install additional resources for NLP (e.g., SpaCy models):
   ```command prompt
   python -m spacy download en_core_web_sm
   ```

4. Set up the directory structure:
   - Place saved models and results in the `model_saves` directory.
   - Ensure the `static` folder contains necessary images for the app.

---

## How to Use

### Run Notebooks
1. Use `8_Mod.ipynb` and `BERT_Mod.ipynb` for training and evaluation:
   ```command promt
   jupyter notebook
   ```

### Launch the Streamlit Application
1. Start the Streamlit app:
   ```command prompt
   streamlit run app_Streamlit.py
   ```
2. Open the web interface in your browser:
   - Default: `http://localhost:8501/`

3. Use the navigation sidebar to explore various features, including emotion prediction and model insights.

---

## System Requirements
- **Operating System:** Windows, macOS, or Linux
- **Hardware:**
  - GPU: NVIDIA with CUDA support (optional but recommended for training)
  - RAM: 8GB or more
- **Software:** Python 3.10, Jupyter Notebook, Streamlit

---

## Authors
This project was developed by:
- Vrishin Reddy Minkuri  
- Rohith Reddy Marlapally  
- Rishitha Reddy Bitla  
- Keerthana Reddy Kalva  

---

## Acknowledgments
- Professor Dr. Duoduo Liao for guidance and support.
- Open-source libraries, including Hugging Face, TensorFlow, PyTorch, and Streamlit.

--- 
"# Emotion_Detection" 
