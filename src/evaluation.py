#!/usr/bin/env python3

import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import os

def evaluate_model(model, X_test, y_test, language_codes):
    """
    Evaluate the model and generate comprehensive metrics.
    
    Args:
        model: Trained Keras model
        X_test: Test data
        y_test: One-hot encoded test labels
        language_codes: List of language codes in the same order as labels
    
    Returns:
        Dictionary of evaluation metrics
    """
    # Make predictions
    y_pred_proba = model.predict(X_test)
    y_pred = np.argmax(y_pred_proba, axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_true, y_pred)
    
    # Calculate precision, recall, and F1 for each language
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=range(len(language_codes))
    )
    
    # Create a confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Prepare metrics dictionary
    metrics = {
        "accuracy": float(accuracy),
        "language_metrics": {}
    }
    
    # Add per-language metrics
    for i, lang in enumerate(language_codes):
        metrics["language_metrics"][lang] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1_score": float(f1[i]),
            "support": int(support[i])
        }
    
    return metrics, cm

def save_metrics(metrics, output_dir="results"):
    """Save metrics to a JSON file."""
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, "evaluation_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

def plot_confusion_matrix(cm, language_codes, output_dir="results"):
    """Plot and save the confusion matrix."""
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=language_codes, 
                yticklabels=language_codes)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
    plt.close()

def plot_training_history(history, output_dir="results"):
    """Plot and save the training history."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot accuracy
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_history.png"))
    plt.close()