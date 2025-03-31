# Language Identification Model using MasakhaNEWS Dataset

## Overview

This project implements a language identification model using the [MasakhaNEWS dataset](https://huggingface.co/datasets/masakhane/masakhanews), a large publicly available dataset for news topic classification in 16 African languages. The model utilizes a Convolutional Neural Network (CNN) architecture to classify the language of news articles based on their text content.


The model is however trained to identify the following languages from the corpus:

- **Amharic** (amh)
- **Igbo** (ibo)
- **Oromo** (orm)
- **Nigerian Pidgin** (pcm)
- **Kirundi**  (run)
- **chiShona**  (sna)


## Project Structure

Language_ID/
├── data/
│   └── Language_ID/masakhanews/  # Contains the original downloaded dataset
│       ├── amh/
│       ├── ibo/
│       ├── orm/
│       └── ... (other language folders)
│
├── models/
│   ├── language_id_model.h5
│   ├── metadata.json         
│   ├── tokenizer.json   
│
├── results/
│   ├── evaluation_metrics.json 
│   ├── confusion_matrix.png 
│   ├── training_history.png    
│
├── src/
│   ├── preprocessing.py  
│   ├── model.py          
│   ├── evaluation.py     
│   ├── prediction.py  
│
├── notebooks/
│   ├── language_id_training.ipynb
│
├── README.md    
├── requirements.txt 


## Getting Started

### Prerequisites

Before running the project, ensure you have the following installed:

* **Python:** Version 3.8 or higher is recommended.
* **pip:** Python package installer.
* **TensorFlow/Keras:** For building and training the CNN model.
* **pandas:** For data manipulation.
* **scikit-learn:** For evaluation metrics.
* **matplotlib and seaborn:** For plotting.
* **Hugging Face Datasets:** (Though the project is configured for local data, mentioning it for context is good).


You can install the necessary dependencies using pip:

```bash
pip install -r requirements.txt


