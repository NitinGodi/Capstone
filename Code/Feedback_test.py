# Import libraries
import pandas as pd
import nltk
import pickle
from transformers import AutoTokenizer, RobertaTokenizerFast
import json
import torch
import Feedback_util as util
from Feedback_model_selection import get_model

# Load and prepare configuration variables
f = open('feedback_config.json')
CONF = json.load(f)
ID2CLASS = {v: k for k, v in CONF['CLASS2ID'].items()}
LABELS_TO_IDS = {v:k for k,v in enumerate(CONF['OUTPUT_LABELS'])}
IDS_TO_LABELS = {k:v for k,v in enumerate(CONF['OUTPUT_LABELS'])}

# Naive-Bayes model
def naive_bayes(text):
    # Preprocessing the data from the app
    lines = nltk.sent_tokenize(text)
    start = 0
    discourse_text = []
    pred_string = []
    for line in lines:
        discourse_text.append(line)
        length = len(line.split())
        end = start + length
        l = list(range(start, end))
        l = ' '.join([str(j) for j in l])
        pred_string.append(l)
        start += length
    # Loading the naive-bayes pipeline
    nb_model = pickle.load(open(f'{CONF['MODEL_PATH']}/naive_bayes_model.pkl', 'rb'))
    # Making predictions
    y_pred = nb_model.predict(lines)
    # Formatting and returning the predictions
    df = pd.DataFrame({'class': y_pred, 'predictionstring': pred_string})
    for i in range(len(df['predictionstring'])):
        df['predictionstring'][i] = list(map(int, df['predictionstring'][i].split()))
    df['class'] = df['class'].replace(ID2CLASS)
    return df

# LSTM model
def lstm(text):
    # Initializing the word and label vocabulary
    tokenizer = RobertaTokenizerFast.from_pretrained(CONF['MODEL_NAME'], add_prefix_space=True)
    word_to_id = tokenizer.vocab
    word_vocab = util.WordVocabulary(word_to_id)
    label_vocab = util.LabelVocabulary(CONF['OUTPUT_LABELS'])
    # Building the word embedding matrix
    pretrain_word_embedding = util.build_pretrain_embedding(CONF['pretrain_embed_path'], word_vocab,
                                                       CONF['word_embed_dim'])
    # Initializing the lstm model
    lstm_model = get_model('lstm', word_vocab, label_vocab, pretrain_word_embedding)
    # Loading the state dictionary of the saved lstm model
    lstm_model.load_state_dict(torch.load(f'{CONF['MODEL_PATH']}/best_lstm_model.bin', map_location=torch.device('cpu')))
    # Preprocessing the data from the app
    text_split = text.split()
    input = []
    for word in text_split:
        input.append(word_vocab.word_to_id(word))
    input = torch.tensor(input).long()
    # Making predictions
    tag_seq = lstm_model.predict(input)
    # Formatting and return ing the predictions
    tag_seq = [[label_vocab.id_to_label(j) for j in i] for i in tag_seq.numpy()]
    print(tag_seq)
    tag_seq = tag_seq[0]
    final_preds = []
    j = 0
    # Recording the chunks, with more than 4 words, having same labels
    while j < len(tag_seq):
        cls = tag_seq[j]
        if cls == 'O':
            pass
        else:
            cls = cls.replace('B', 'I')
        end = j + 1
        while end < len(tag_seq) and tag_seq[end] == cls:
            end += 1
        if cls != 'O' and cls != '' and end - j > 4:
            final_preds.append((cls.replace('I-', ''),
                                list(range(j, end))))
        j = end

    df_pred = pd.DataFrame(final_preds)
    df_pred.columns = ['class', 'predictionstring']
    return df_pred

# Transformer model
def roberta(essay):
    # Preprocessing the data from the app
    df_test = [essay]
    df_test = pd.DataFrame({'id': '01', 'text': df_test})
    df_test['text_split'] = df_test.text.str.split()
    tokenizer = AutoTokenizer.from_pretrained(CONF['MODEL_NAME'], add_prefix_space=True)
    tokenized_text = util.tokenize(df_test, tokenizer, with_labels=False)
    # Getting predictions
    df_sub = util.app_get_predictions(tokenized_text)
    return df_sub

# Calls the functions based on model and returns the final predictions to the app
def get_result(model, essay):
    if model == 'naive-bayes':
        return naive_bayes(essay)
    elif model == 'lstm':
        return lstm(essay)
    elif model == 'roberta':
        return roberta(essay)
