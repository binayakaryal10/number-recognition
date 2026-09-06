"""
Train and Save CNN Digit Recognition Model
Reads MNIST dataset, trains the CNN architecture, and exports the model to disk.
"""

import os
import sys
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

MODEL_PATH = "mnist_cnn_model.keras"
TRAIN_CSV = "mnist_train.csv"
TEST_CSV = "mnist_test.csv"

def load_data():
    if os.path.exists(TRAIN_CSV) and os.path.exists(TEST_CSV):
        print(f"Loading data from local CSV files: {TRAIN_CSV} and {TEST_CSV}...")
        df_train = pd.read_csv(TRAIN_CSV)
        df_test = pd.read_csv(TEST_CSV)
        
        X_train = df_train.drop("label", axis=1).values.astype("float32") / 255.0
        y_train = df_train["label"].values
        
        X_test = df_test.drop("label", axis=1).values.astype("float32") / 255.0
        y_test = df_test["label"].values
    else:
        print("CSV files not found. Falling back to tf.keras.datasets.mnist...")
        (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
        X_train = X_train.astype("float32") / 255.0
        X_test = X_test.astype("float32") / 255.0

    X_train = X_train.reshape(-1, 28, 28, 1)
    X_test = X_test.reshape(-1, 28, 28, 1)
    
    y_train_cat = to_categorical(y_train, 10)
    y_test_cat = to_categorical(y_test, 10)
    
    print(f"Training samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")
    return X_train, y_train_cat, X_test, y_test_cat

def build_cnn_model():
    model = Sequential([
        Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(10, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def main():
    print("=" * 60)
    print("Starting CNN Model Training on MNIST")
    print("=" * 60)
    
    X_train, y_train, X_test, y_test = load_data()
    
    model = build_cnn_model()
    model.summary()
    
    print("\nTraining CNN for 5 epochs...")
    history = model.fit(
        X_train, y_train,
        epochs=5,
        batch_size=64,
        validation_data=(X_test, y_test),
        verbose=1
    )
    
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print("\n" + "=" * 60)
    print(f"Test Accuracy: {test_acc * 100:.2f}% | Test Loss: {test_loss:.4f}")
    print("=" * 60)
    
    # Save the model
    print(f"Saving model to {MODEL_PATH}...")
    model.save(MODEL_PATH)
    print(f"Model successfully saved to {os.path.abspath(MODEL_PATH)}!")

if __name__ == "__main__":
    main()
