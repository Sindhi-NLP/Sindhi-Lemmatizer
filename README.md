# Sindhi POS Tagging and Lemmatization

This project focuses on building a POS tagging and lemmatization system for the Sindhi language using sequence-to-sequence models. The models are trained using LSTM networks and the Keras library, and include both training and inference scripts.

## Project Structure

- `SindhiPosDataset.csv`: The dataset used for training the models.
- `config.pickle`: Configuration file containing parameters for preprocessing and model inference.
- `decoder_model.h5`: The trained decoder model for sequence-to-sequence inference.
- `encoder_model.h5`: The trained encoder model for sequence-to-sequence inference.
- `full_model.h5`: The complete trained model.
- `input_tokenizer.pickle`: The tokenizer for input sequences.
- `target_tokenizer.pickle`: The tokenizer for target sequences.
- `model_training.ipynb`: Jupyter notebook containing the code for training the models.
- `model_inference.ipynb`: Jupyter notebook containing the code for running inference with the trained models.

## Setup Instructions

1. **Clone the repository**:
    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

2. **Install required packages**:
    Make sure you have Python installed, and install the necessary packages using pip:
    ```bash
    pip install numpy scikit-learn tensorflow
    ```

3. **Download and extract dataset**:
    Ensure `SindhiPosDataset.csv` is in the project directory.

4. **Run the training script**:
    Open the `model_training.ipynb` notebook and run all cells to train the models. This will save the trained models and tokenizers in the project directory.

5. **Run the inference script**:
    Open the `model_inference.ipynb` notebook and run all cells to load the models and test the lemmatization function. This will print the lemmatized output for test words.

## Lemmatization Function

The lemmatization function takes Sindhi words as input and returns their lemmas using the trained sequence-to-sequence models.

### Example Code
```encoder_model, decoder_model, input_tokenizer, target_tokenizer, config = load_models_and_tokenizers()
# Test the lemmatizer
test_words = ['ڪرڻ', 'اڪثر', 'کائيندو']
for word in test_words:
  lemma = lemmatize(word, encoder_model, decoder_model, input_tokenizer, target_tokenizer, config)
  print(f"Word: {word}, Lemma: {lemma}")
```

```Word: ڪرڻ, Lemma: ڪر
Word: اڪثر, Lemma: اڪثر
Word: کائيندو, Lemma: ناه
