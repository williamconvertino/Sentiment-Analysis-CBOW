import os

from datasets import load_dataset

import tensorflow as tf
import tensorflow.keras as keras

import numpy as np

# === Dataset Initialization ===

def get_dataset_split():

    dataset = load_dataset("yelp_polarity")

    training_texts = np.array(dataset['train']['text'])
    training_labels = np.array(dataset['train']['label'])

    testing_texts = np.array(dataset['test']['text'])
    testing_labels = np.array(dataset['test']['label'])

    all_text = np.concatenate([training_texts, testing_texts])

    return training_texts, training_labels, testing_texts, testing_labels, all_text

# === Model Construction ===

def build_model(all_text, model_path = 'empty_model.keras'):

    if os.path.exists(model_path):
        model = keras.models.load_model(model_path)
    else:
        text_vectorization = keras.layers.TextVectorization(max_tokens=20000)

        text_vectorization.adapt(all_text)

        vocabulary = text_vectorization.get_vocabulary()
        vocabulary_size = len(vocabulary)

        embedding_dimensions = 50

        model = keras.Sequential([
            text_vectorization,
            keras.layers.Embedding(input_dim=vocabulary_size, output_dim=embedding_dimensions),
            keras.layers.GlobalAveragePooling1D(),
            keras.layers.Dense(1, activation='sigmoid')
        ])

        model.save(model_path)

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# === Model Training ===

def train_model(model, texts, labels):
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, verbose=2)
    model.fit(texts, labels, batch_size=2048, epochs=50, validation_split=0.2, callbacks=[early_stopping], verbose=2)

# === Model Evaluation ===

def evaluate_model(model, texts, labels):
    loss, accuracy = model.evaluate(texts, labels)

    print(f'Loss: {loss}')
    print(f'Accuracy: {accuracy}')

# === Main ===
    
def main():
    training_tests, training_labels, testing_texts, testing_labels, all_text = get_dataset_split()

    model = build_model(all_text)

    train_model(model, training_tests, training_labels)

    model.save('sentiment_analysis_model.keras')

    # model = keras.models.load_model('sentiment_analysis_model.keras')

    evaluate_model(model, testing_texts, testing_labels)

if __name__ == '__main__':
    main()