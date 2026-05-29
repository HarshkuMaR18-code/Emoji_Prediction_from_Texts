# 😊 Emoji Prediction from Text using NLP and Deep Learning

## 📌 Overview

Emoji Prediction from Text is a Natural Language Processing (NLP) project that automatically predicts the most appropriate emoji for a given sentence. The model uses pre-trained GloVe word embeddings and LSTM neural networks to understand the semantic meaning and context of text before assigning an emoji label.

This project demonstrates how deep learning can enhance text-based communication by making emoji recommendations based on sentence meaning rather than simple keyword matching.

---

## 🚀 Features

* Predict emojis directly from text sentences.
* Uses pre-trained GloVe 50-dimensional word embeddings.
* Stacked LSTM architecture for sequence learning.
* Experiment tracking with MLflow and DagsHub.
* End-to-end NLP and Deep Learning workflow.
* Handles unseen words through semantic word representations.

---

## 🏗️ Model Architecture

```text
Input Sentence
      ↓
Text Preprocessing
      ↓
Word Index Conversion
      ↓
GloVe Embedding Layer
      ↓
LSTM (128 Units)
      ↓
Dropout (0.5)
      ↓
LSTM (128 Units)
      ↓
Dropout (0.5)
      ↓
Dense Layer (5 Classes)
      ↓
Softmax
      ↓
Predicted Emoji
```

---

## 🧠 Technologies Used

* Python
* NumPy
* Pandas
* TensorFlow / Keras
* GloVe Word Embeddings
* MLflow
* DagsHub

---

## 📂 Dataset

The dataset consists of sentences paired with emoji labels. The model learns the relationship between sentence meaning and corresponding emojis.

Example:

| Text                | Emoji |
| ------------------- | ----- |
| I love you          | ❤️    |
| Let's play football | ⚽     |
| Congratulations!    | 👍    |
| Want some coffee?   | ☕     |
| I am very happy     | 😄    |

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/Emoji_Prediction_from_Text.git
cd Emoji_Prediction_from_Text
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Download GloVe embeddings from:

https://nlp.stanford.edu/projects/glove/

Place `glove.6B.50d.txt` in the required directory before running the project.

---

## ▶️ Running the Project

```bash
python mlflow_tracking.py
```

The script will:

* Load the dataset
* Load GloVe embeddings
* Train the LSTM model
* Evaluate model performance
* Log experiments using MLflow and DagsHub

---

## 📊 Experiment Tracking

This project integrates:

* **MLflow** for logging metrics and parameters
* **DagsHub** for experiment management and model tracking

Tracked information includes:

* Training accuracy
* Hyperparameters
* Model artifacts
* Experiment history

---

## 🔮 Future Improvements

* Support additional emoji classes
* Use Bidirectional LSTM models
* Experiment with BERT and Transformer models
* Deploy as a web application
* Build a real-time emoji recommendation system

---

## 👨‍💻 Author

**Harsh Kumar**

Machine Learning | Deep Learning | MLOps Enthusiast

⭐ If you found this project useful, consider giving it a star on GitHub!
