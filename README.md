# 😊 Emoji Prediction from Text

## 📌 Overview

Emoji Prediction from Text is a Deep Learning and Natural Language Processing (NLP) project that predicts the most suitable emoji for a given sentence. The model uses pre-trained GloVe word embeddings and LSTM networks to understand sentence meaning and classify text into emoji categories.

This project demonstrates how word embeddings and sequence models can be used to build intelligent and context-aware emoji recommendation systems.

---

## 🚀 Features

* Predict emojis from text sentences.
* Uses pre-trained GloVe 50-dimensional embeddings.
* Stacked LSTM architecture for sequence learning.
* Experiment tracking with MLflow and DagsHub.
* End-to-end NLP pipeline from preprocessing to prediction.

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
LSTM (128)
      ↓
Dropout (0.5)
      ↓
LSTM (128)
      ↓
Dropout (0.5)
      ↓
Dense Layer
      ↓
Softmax
      ↓
Emoji Prediction
```

---

## 🛠️ Technologies Used

* Python
* NumPy
* Pandas
* TensorFlow / Keras
* GloVe Word Embeddings
* MLflow
* DagsHub

---

## 📂 Dataset

The dataset contains text sentences paired with emoji labels.

Examples:

| Sentence            | Emoji |
| ------------------- | ----- |
| I love you          | ❤️    |
| Let's play football | ⚽     |
| Congratulations!    | 👍    |
| Want some coffee?   | ☕     |
| I am very happy     | 😄    |

---

## 📥 Download GloVe Embeddings

This repository does not include the GloVe file due to its large size.

1. Download GloVe from:
   https://nlp.stanford.edu/projects/glove/

2. Extract:

```text
glove.6B.50d.txt
```

3. Place it in:

```text
data/Files/glove.6B.50d.txt
```

---

## ▶️ Run the Project

Install dependencies:

```bash
pip install -r requirements.txt
```

Run:

```bash
python mlflow_tracking.py
```

---

## 📊 Experiment Tracking

Experiments are tracked using:

* MLflow
* DagsHub

Logged information includes:

* Accuracy
* Parameters
* Model artifacts
* Training runs

---

## 🔮 Future Improvements

* Support more emoji classes
* Use Bidirectional LSTMs
* Experiment with Transformer models (BERT)
* Deploy as a web application

---

## 👨‍💻 Author

**Harsh Kumar**

Machine Learning | Deep Learning | MLOps

⭐ If you found this project useful, consider giving it a star on GitHub.

