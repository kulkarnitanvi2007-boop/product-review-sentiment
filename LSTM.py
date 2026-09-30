from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


# Create tokenizer
tok = Tokenizer(
    num_words=10000,
    oov_token="<OOV>"
)

tok.fit_on_texts(X_train)


# Convert text to sequences
seq_tr = pad_sequences(
    tok.texts_to_sequences(X_train),
    maxlen=120
)

seq_te = pad_sequences(
    tok.texts_to_sequences(X_test),
    maxlen=120
)


# Create LSTM model
model = Sequential([
    Embedding(
        10000,
        128,
        input_length=120
    ),
    LSTM(
        64,
        dropout=0.3
    ),
    Dense(
        3,
        activation='softmax'
    )  # 3 classes: neg / neutral / pos
])


# Compile the model
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)


# Train the model
model.fit(
    seq_tr,
    y_train,
    epochs=5,
    batch_size=64,
    validation_data=(seq_te, y_test)
)