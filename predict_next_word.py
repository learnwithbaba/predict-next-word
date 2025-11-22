import streamlit as st
import numpy as np
import pickle
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

#Load LSTM Model
model = load_model("next_word_LSTM.h5")

#Load the tokenizer
with open("tokenizer.pickle","rb") as handle:
    tokenizer = pickle.load(handle)

#Function to predict
def predict_next_word(model, tokenizer, text, max_sequence_len):
  token_list = tokenizer.texts_to_sequences([text])[0]
  print(token_list)
  if len(token_list) >= max_sequence_len:
    token_list = token_list[-(max_sequence_len-1):]
  
  token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding="pre")
  predicted = model.predict(token_list, verbose=0)
  predicted_word_index = np.argmax(predicted,axis=1)

  for word, index in tokenizer.word_index.items():
    if index == predicted_word_index:
      return word

  return None

#Streamlit App
st.title("Next word prediction with LSTM RNN")
input_text = st.text_input("Enter the sequence of words","As far as I")

if st.button("Predict next word"):
  max_sequence_len = model.input_shape[1]+1

#Retrieve the maximum sequence Length
  next_word = predict_next_word(model, tokenizer, input_text, max_sequence_len)
  st.write(f"next word: {next_word}")