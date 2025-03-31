import re
import os
import numpy as py
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical


def clean_text(text):
  """cleans text data"""
  if text is None:
    return ""
  
  # convert into lowercase
  text = text.lower()

  #remove urls
  text = re.sub(r'https?://\S+|www\.\S+', '', text)

  #remove html tags
  txt = re.sub(r'<.*?>', '', text)
  return text.strip()

def load_and_preprocess(processed_path, target_languages):
  """
  load and preprocess data from csvs, uisng the 'text' column
  returns:
   preprocessed train, val and test data and labes, tokenizer, max length, num classes
  """
  train_df = pd.read_csv(os.path.join(processed_path, "train.csv"), usecols=["text", "lang"]).dropna()
  val_df = pd.read_csv(os.path.join(processed_path, "validation.csv"), usecols=["text", "lang"]).dropna()
  test_df = pd.read_csv(os.path.join(processed_path, "test.csv"), usecols=["text", "lang"]).dropna()

  # apply cleaning to 'text' column
  train_df['text'] = train_df['text'].apply(clean_text)
  val_df['text'] = val_df['text'].apply(clean_text)
  test_df['text'] = test_df['text'].apply(clean_text)

  #prepare text for tokenizer fitting from the 'text' column
  all_train_texts = train_df['text'].tolist()

  #create and fit tokenizer on training data
  tokenizer = Tokenizer(char_level=True, oov_token='<unk>')
  tokenizer.fit_on_texts(all_train_texts)

  # max sequence length
  max_length = min(max([len(text) for text in all_train_texts]), 500)

  # convert texts to sequences and pad
  X_train = tokenizer.texts_to_sequences(train_df['text'].tolist())
  X_train = pad_sequences(X_train, maxlen=max_length)

  X_val = tokenizer.texts_to_sequences(val_df['text'].tolist())
  X_val = pad_sequences(X_val, maxlen=max_length)

  X_test = tokenizer.texts_to_sequences(test_df['text'].tolist())
  X_test = pad_sequences(X_test, maxlen=max_length)

  # convert language labels into numerical format
  language_to_index = {lang: i for i, lang in enumerate(target_languages)}
  y_train = to_categorical(train_df['lang'].map(language_to_index), num_classes=len(target_languages))
  y_val = to_categorical(val_df['lang'].map(language_to_index), num_classes=len(target_languages))
  y_test = to_categorical(test_df['lang'].map(language_to_index), num_classes=len(target_languages))

  return (X_train, y_train), (X_val, y_val), (X_test, y_test), tokenizer, max_length, len(target_languages)

