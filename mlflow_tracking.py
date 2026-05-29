import pandas as pd
import numpy as np
import dagshub
import mlflow

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Dropout, LSTM, Activation
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.utils import to_categorical


def read_glove_vecs(glove_file):
    with open(glove_file, 'r', encoding='utf8') as f:
        words = set()
        word_to_vec_map = {}
        for line in f:
            line = line.strip().split()
            curr_word = line[0]
            words.add(curr_word)
            word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float64)
    
    i = 1
    words_to_index = {}
    index_to_words = {}
    for w in sorted(words):
        words_to_index[w] = i
        index_to_words[i] = w
        i = i + 1
    return words_to_index, index_to_words, word_to_vec_map


def sentences_to_indices(X, word_to_index, max_len):
    m = X.shape[0]
    X_indices = np.zeros((m, max_len))

    for i in range(m):
        words = str(X[i]).lower().split()
        for j, w in enumerate(words):
            if w in word_to_index and j < max_len:
                X_indices[i, j] = word_to_index[w]
    return X_indices


def pretrained_embedding_layer(word_to_vec_map, word_to_index):
    vocab_size = len(word_to_index) + 1
    any_word = next(iter(word_to_vec_map))
    emb_dim = word_to_vec_map[any_word].shape[0]

    emb_matrix = np.zeros((vocab_size, emb_dim))

    for word, idx in word_to_index.items():
        emb_matrix[idx] = word_to_vec_map[word]

    embedding_layer = Embedding(vocab_size, emb_dim, trainable=False)

    embedding_layer.build((None,))
    embedding_layer.set_weights([emb_matrix])

    return embedding_layer


dagshub.init(repo_owner="HarshkuMaR18-code", repo_name="Emoji_Prediction_from_Text", mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/HarshkuMaR18-code/Emoji_Prediction_from_Text.mlflow")

train_df = pd.read_csv(r'C:\mlops\mlflow\Emoji_Prediction_from_Texts\data\Files\train_emoji.csv', header=None)
X_train = train_df[0].values
Y_train = train_df[1].values

test_df = pd.read_csv(r'C:\mlops\mlflow\Emoji_Prediction_from_Texts\data\Files\tesss.csv', header=None)
X_test = test_df[0].values
Y_test = test_df[1].values

word_to_index, index_to_word, word_to_vec_map = read_glove_vecs(r'C:\mlops\mlflow\glove.6B.50d.txt')

maxLen = len(max(X_train, key=lambda x: len(str(x).split())).split())


mlflow.autolog()
mlflow.set_experiment('Emojify')

with mlflow.start_run():

    sentence_indices = Input(shape=(maxLen,), dtype="int32")

    embedding_layer = pretrained_embedding_layer(word_to_vec_map, word_to_index)
    X = embedding_layer(sentence_indices)

    X = LSTM(128, return_sequences=True)(X)
    X = Dropout(0.5)(X)

    X = LSTM(128)(X)
    X = Dropout(0.5)(X)

    X = Dense(5)(X)
    X = Activation("softmax")(X)

    model = Model(sentence_indices, X)

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    X_train_indices = sentences_to_indices(X_train, word_to_index, maxLen)
    Y_train_oh = to_categorical(Y_train, num_classes=5)

    model.fit(X_train_indices, Y_train_oh, epochs=50, batch_size=32, shuffle=True)

    X_test_indices = sentences_to_indices(X_test, word_to_index, max_len=maxLen)
    Y_test_oh = to_categorical(Y_test, num_classes=5)
    loss, acc = model.evaluate(X_test_indices, Y_test_oh)