import os
import pandas as pd
from collections import defaultdict
import nltk
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM
import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForTokenClassification

ID2CLASS = {0:'No Class', 1:'Claim' , 2:'Evidence' ,  3:'Position' , 4:'Concluding Statement' , 5:'Lead', 6:'Counterclaim', 7:'Rebuttal' }
CLASS2ID = {v: k for k, v in ID2CLASS.items()}
output_labels = ['O', 'B-Lead', 'I-Lead', 'B-Position', 'I-Position', 'B-Claim', 'I-Claim', 'B-Counterclaim', 'I-Counterclaim',
          'B-Rebuttal', 'I-Rebuttal', 'B-Evidence', 'I-Evidence', 'B-Concluding Statement', 'I-Concluding Statement']
LABELS_TO_IDS = {v:k for k,v in enumerate(output_labels)}
IDS_TO_LABELS = {k:v for k,v in enumerate(output_labels)}
max_words = 10000
EMBEDDING_DIM = 300
max_len = 512
MODEL_PATH = 'model'
MODEL_CHECKPOINT = 'pytorch_model_e4.bin'
DOC_STRIDE = 200
MODEL_NAME = 'roberta-base'
DEVICE = 'cpu'

def get_labels(word_ids, word_labels):
    label_ids = []
    for word_idx in word_ids:
        if word_idx is None:
            label_ids.append(-100)
        else:
            label_ids.append(LABELS_TO_IDS[word_labels[word_idx]])
    return label_ids

def tokenize(df, tokenizer, to_tensor=True, with_labels=True):
    # This is what's different from a longformer
    # Read the parameters with attention
    encoded = tokenizer(df['text_split'].tolist(),
                        is_split_into_words=True,
                        return_overflowing_tokens=True,
                        stride=DOC_STRIDE,
                        max_length=max_len,
                        padding="max_length",
                        truncation=True)

    if with_labels:
        encoded['labels'] = []

    encoded['wids'] = []
    n = len(encoded['overflow_to_sample_mapping'])
    for i in range(n):

        # Map back to original row
        text_idx = encoded['overflow_to_sample_mapping'][i]

        # Get word indexes (this is a global index that takes into consideration the chunking :D )
        word_ids = encoded.word_ids(i)

        if with_labels:
            # Get word labels of the full un-chunked text
            word_labels = df['entities'].iloc[text_idx]

            # Get the labels associated with the word indexes
            label_ids = get_labels(word_ids, word_labels)
            encoded['labels'].append(label_ids)
        encoded['wids'].append([w if w is not None else -1 for w in word_ids])

    if to_tensor:
        encoded = {key: torch.as_tensor(val) for key, val in encoded.items()}
    return encoded

def download_model():
    # https://www.kaggle.com/cdeotte/pytorch-bigbird-ner-cv-0-615
    if os.path.exists(MODEL_PATH) != True:
        os.mkdir(MODEL_PATH)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, add_prefix_space=True)
    tokenizer.save_pretrained(MODEL_PATH)

    config_model = AutoConfig.from_pretrained(MODEL_NAME)
    config_model.num_labels = 15
    config_model.save_pretrained(MODEL_PATH)

    backbone = AutoModelForTokenClassification.from_pretrained(MODEL_NAME,
                                                               config=config_model)
    backbone.save_pretrained(MODEL_PATH)
    print(f"Model downloaded to {MODEL_PATH}/")

def load_model():
    model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH)
    model.to(DEVICE)
    model.load_state_dict(torch.load(MODEL_CHECKPOINT, map_location='cpu'))
    model.eval()
    print('Model loaded.')
    return model

def inference(batch):
    # These 2 dictionaries will hold text-level data
    # Helping in the merging process by accumulating data
    # Through all the chunks
    predictions = defaultdict(list)
    seen_words_idx = defaultdict(list)

    roberta_model = load_model()

    ids = batch["input_ids"].to(DEVICE)
    mask = batch["attention_mask"].to(DEVICE)
    outputs = roberta_model(ids, attention_mask=mask, return_dict=False)

    batch_preds = torch.argmax(outputs[0], axis=-1).cpu().numpy()

    for k, (chunk_preds, text_id) in enumerate(zip(batch_preds, batch['overflow_to_sample_mapping'].tolist())):
        word_ids = batch['wids'][k].numpy()
        chunk_preds = [IDS_TO_LABELS[i] for i in chunk_preds]

        for idx, word_idx in enumerate(word_ids):
            if word_idx == -1:
                pass
            elif word_idx not in seen_words_idx[text_id]:
                predictions[text_id].append(chunk_preds[idx])
                seen_words_idx[text_id].append(word_idx)

    final_predictions = [predictions[k] for k in sorted(predictions.keys())]
    return final_predictions

def get_predictions(tokenized_text):
    pred = inference(tokenized_text)[0]
    final_preds = []
    j = 0
    while j < len(pred):
        cls = pred[j]
        if cls == 'O':
            pass
        else:
            cls = cls.replace('B', 'I')
        end = j + 1
        while end < len(pred) and pred[end] == cls:
            end += 1
        if cls != 'O' and cls != '' and end - j > 4:
            final_preds.append((cls.replace('I-', ''),
                                list(range(j, end))))
        j = end

    df_pred = pd.DataFrame(final_preds)
    df_pred.columns = ['class', 'predictionstring']
    return df_pred

def naive_bayes(text):
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
    nb_model = pickle.load(open('naive_bayes_model.pkl', 'rb'))
    y_pred = nb_model.predict(lines)
    df = pd.DataFrame({'class': y_pred, 'predictionstring': pred_string})
    for i in range(len(df['predictionstring'])):
        df['predictionstring'][i] = list(map(int, df['predictionstring'][i].split()))
    df['class'] = df['class'].replace(ID2CLASS)
    return df

def lstm(text):
    discourse_text = []
    discourse_start =[]
    discourse_end = []
    split_text = []
    predictionstring = []
    lines = nltk.sent_tokenize(text)
    start = 0
    for line in lines:
        discourse_text.append(line)
        split_text.append(line.split())
        length = len(line.split())
        end = start + length
        l = list(range(start,end))
        l = ' '.join([str(j) for j in l])
        predictionstring.append(l)
        discourse_start.append(start)
        discourse_end.append(end-1)
        start += length
    testing_data =pd.DataFrame()
    testing_data['discourse_text'] = discourse_text
    testing_data['split_text'] = split_text
    testing_data['discourse_start'] = discourse_start
    testing_data['discourse_end'] = discourse_end
    testing_data['predictionstring'] = predictionstring
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(testing_data['discourse_text'])
    X_test = tokenizer.texts_to_sequences(testing_data['split_text'].values)
    X_test = pad_sequences(X_test, maxlen=max_len)

    lstm_model = Sequential()
    lstm_model.add(Embedding(max_words, EMBEDDING_DIM, input_length=max_len))
    lstm_model.add(LSTM(32, return_sequences=True))
    lstm_model.add(LSTM(32, return_sequences=True))
    lstm_model.add(LSTM(32, return_sequences=True))
    lstm_model.add(LSTM(32, return_sequences=True))
    lstm_model.add(LSTM(32))
    lstm_model.add(Dense(8, activation='softmax'))
    lstm_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    lstm_model.load_weights('lstm_model.h5')

    y_pred = lstm_model.predict(X_test)
    output = pd.DataFrame(y_pred ,columns= CLASS2ID.keys())
    output = list(output.idxmax(axis = 1))
    submission_df = pd.DataFrame()
    submission_df['class'] = output
    submission_df['predictionstring'] = testing_data['predictionstring']
    for i in range(len(submission_df['predictionstring'])):
        submission_df['predictionstring'][i] = list(map(int,submission_df['predictionstring'][i].split()))
    submission_df['class']= submission_df['class'].replace(ID2CLASS)
    return submission_df

def roberta(essay):
    df_test = [essay]
    df_test = pd.DataFrame({'id': '01', 'text': df_test})
    df_test['text_split'] = df_test.text.str.split()
    download_model()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    tokenized_text = tokenize(df_test, tokenizer, with_labels=False)
    df_sub = get_predictions(tokenized_text)
    return df_sub

def get_result(model, essay):
    if model == 'naive-bayes':
        return naive_bayes(essay)
    elif model == 'lstm':
        return lstm(essay)
    elif model == 'roberta':
        return roberta(essay)

# text = "How should kids be today: at home all day doing chores or outside doing sports and having fun? Playing sports is healthy for your mind, body, and strength. It's also fun to hang out with friends and take your mind off of things at home. Last but not least, it's great for extra credit for future classes. Therefore; I agree with requiring at least one extracurricular activity like sports, for it is fun, healthy, and provides extra credit. Sports could be very for kids like me. Sports are definitely great for hanging out with friends. You can practice for games with your best of friends. Also, you can take a break from home like chores and babysitting. It can also take away from some of your stress by not even thinking about it. Everyday life could be taken a break from just by focusing on practicing and playing. So you dont have to worry about everything you have to do at home. Playing sports is also very healthy. It requires a lot of exercise, making you physically active. Practicing is exercise by planning and getting ready for everything for the next game you play a game. Games require the most energy of all, for you are doing everything planned but ten times harder. It can also be a big stress reliever. Better sleep schedules is an effect from doing a lot of practice and exercise. While practicing or playing games, it can also take your mind off of things at home. Extra credit is a great effect of playing sports. It helps you get into certain schools that are tough to be accepted in to. High school specialty programs are tough to be accepted in to but playing sports can help. Also, playing sports looks great on any college resame. Teachers could also have an interest in the sports you play. They might be impressed and agree to have you as a student in their class. It could also be pleasing knowing that sports help you sleep better to be better focused on classwork. I believe sports should be required as an extracurricular activity. Great effects are it could be fun, healthy, and you could receive extra credit. As kids we should be outside more, be active, and have fun. Sports could also prepare us for future jobs or side activities like hobbies. "
#
# result = get_result('lstm', text)
