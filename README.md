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
```
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
```

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
```

### Installation
#### 1. Clone the repository
```bash
git clone https://github.com/lindahnyambura/Language-ID.git
cd Language-ID
```

#### 2. Download the MasakhaNEWS dataset:

As this project is configured to use locally downloaded data, you need to manually download the desired language splits of the MasakhaNEWS dataset from the Hugging Face Datasets repository. Download the ```train.csv```, ```validation.csv```, and ```test.csv``` files for each target language and place them in the ```data/masakhanews/<language_code>/``` directory as shown in the Project Structure.

For example, for Amharic (amh), you should have:
```
Language_ID/data/masakhanews/amh/train.csv
Language_ID/data/masakhanews/amh/validation.csv
Language_ID/data/masakhanews/amh/test.csv
```

### Usage

The primary way to run the project is through the provided Jupyter Notebook.

**1. Navigate to the notebooks directory:**
```
cd notebooks
```

**2. Start Jupyter Notebook:**
```
jupyter notebook
```

**3. Open and run the ```language_id_training.ipynb``` notebook.**
The notebook contains the following steps:

- **Data Loading and Preprocessing:** Loads the data from the local CSV files, cleans the text, and prepares it for the model.
- **Model Building:** Defines and builds the CNN model architecture.
- **Model Training:** Trains the model on the preprocessed training data, using the validation set for monitoring and early stopping.
- **Model Evaluation:** Evaluates the trained model on the test set and generates performance metrics (accuracy, precision, recall, F1-score, confusion matrix).
- **Saving Results:** Saves the evaluation metrics as a JSON file, the confusion matrix as a PNG image, and the training history as a PNG image in the results directory.
- **Model Saving:** Saves the trained model in the models directory as an .h5 file.

Alternatively, you can run the individual Python scripts in the src directory


### Model Details

The model architecture is a Convolutional Neural Network (CNN) designed for text classification. It consists of:

- **Embedding Layer:** Converts characters into dense vector representations.
- **Convolutional Layers:** Extracts features from sequences of characters.
- **Max Pooling Layers:** Reduces the dimensionality of the feature maps.
- **Global Max Pooling Layer:** Aggregates the features across the entire sequence.
- **Dense Layers:** Performs the final classification.
- **Dropout Layer:** Helps prevent overfitting.

The model is trained using the **Adamw** optimizer and **categorical cross-entropy loss**. Early stopping is used during training to prevent overfitting based on the validation set accuracy.


### Evaluation
The model is evaluated on the test set, and the following metrics are calculated:

- **Accuracy:** The overall percentage of correctly classified language samples.
- **Precision:** The ability of the classifier not to label as positive a sample that is negative, for each language.
- **Recall:** The ability of the classifier to find all the positive samples, for each language.
- **F1-Score:** The harmonic mean of precision and recall, for each language.
- **Confusion Matrix:** A table showing the number of correct and incorrect predictions for each language, allowing for visualization of which languages are often confused.


The evaluation metrics are saved in ```results/evaluation_metrics.json```, and the confusion matrix and training history plots are saved as PNG images in the ```results``` directory.

### Configuration
You can configure the following parameters within the ```language_id_training.ipynb``` notebook or in your training script:

- ```target_languages```: A list of the language codes you want to include in the model training and evaluation. Ensure you have downloaded the data for these languages.
- ```embedding_dim```: The dimensionality of the character embeddings in the CNN model.
- ```batch_size```: The number of samples processed in each training batch.
- ```epochs```: The maximum number of training epochs.
- ```patience```: The number of epochs to wait for improvement in validation accuracy before stopping training early.
- **Paths**: The paths to the dataset and results directories can be adjusted if needed.

### Acknowledgements

- The MasakhaNEWS dataset for providing valuable data for this project and the project that retrains the model built here. The project can be found here:


https://github.com/lindahnyambura/language_id 
