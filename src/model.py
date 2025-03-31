#!/usr/bin/env python3

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, GlobalMaxPooling1D
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.regularizers import l2



def build_cnn_model(vocab_size, max_length, num_classes, embedding_dim=64):
    """
    Build a CNN model for language identification.
    
    Args:
        vocab_size: Size of the vocabulary (number of unique characters)
        max_length: Maximum sequence length
        num_classes: Number of languages to classify
        embedding_dim: Dimension of character embeddings
    
    Returns:
        Compiled Keras model
    """
    model = Sequential([
        # Character embedding layer
        Embedding(input_dim=vocab_size + 1,  # +1 for padding token
                 output_dim=embedding_dim,
                 input_length=max_length),
        
        Conv1D(64, 3, activation='relu', padding='same', kernel_regularizer=l2(0.01)),
        MaxPooling1D(pool_size=2),
        
        Conv1D(128, 3, activation='relu', padding='same', kernel_regularizer=l2(0.01)),
        MaxPooling1D(pool_size=2),
        
        Conv1D(128, 3, activation='relu', padding='same', kernel_regularizer=l2(0.01)),
        GlobalMaxPooling1D(),
        
        Dense(64, activation='relu'),
        Dropout(0.6),
        Dense(num_classes, activation='softmax')
    ])
    
    # compile model
    model.compile(loss='categorical_crossentropy',
                  optimizer='adamw',
                  metrics=['accuracy'])
    
    return model


def train_model(model, X_train, y_train, X_val, y_val, batch_size=64, epochs=10, patience=3):
    """
    Train the CNN model with early stopping.
    """
    # Define callbacks
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=patience,
        restore_best_weights=True
    )
    
    # Train model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=[early_stopping],
        verbose=1
    )
    
    return model, history
