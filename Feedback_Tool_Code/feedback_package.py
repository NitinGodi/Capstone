import os
import gc
import time
import pandas as pd
import numpy as np
from collections import defaultdict
from tqdm.auto import tqdm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import nltk
from sklearn.metrics import accuracy_score, f1_score
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM
from tensorflow.keras.callbacks import ModelCheckpoint
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoConfig, AutoTokenizer, AutoModelForTokenClassification

os.environ['TOKENIZERS_PARALLELISM'] = 'false'
TRAIN_CSV = "../train.csv"
TRAIN_PATH = "../train/"
ID2CLASS = {0:'No Class', 1:'Claim' , 2:'Evidence' ,  3:'Position' , 4:'Concluding Statement' , 5:'Lead', 6:'Counterclaim', 7:'Rebuttal' }
CLASS2ID = {v: k for k, v in ID2CLASS.items()}
output_labels = ['O', 'B-Lead', 'I-Lead', 'B-Position', 'I-Position', 'B-Claim', 'I-Claim', 'B-Counterclaim', 'I-Counterclaim',
          'B-Rebuttal', 'I-Rebuttal', 'B-Evidence', 'I-Evidence', 'B-Concluding Statement', 'I-Concluding Statement']
LABELS_TO_IDS = {v:k for k,v in enumerate(output_labels)}
IDS_TO_LABELS = {k:v for k,v in enumerate(output_labels)}
model = 'lstm'
max_words = 10000
EMBEDDING_DIM = 300
max_len = 512
MODEL_PATH = 'model'
DOC_STRIDE = 200
MODEL_NAME = 'roberta-base'
config = {'train_batch_size': 8,
          'valid_batch_size': 8,
          'epochs': 5,
          'learning_rates': [2.5e-5, 2.5e-5, 2.5e-6, 2.5e-6, 2.5e-7],
          'max_grad_norm': 10,
          'device': 'cuda'}

df = pd.read_csv(TRAIN_CSV)
print(df.head())
print(df.shape)
unique_ids = df['id'].unique()

def fill_gaps(elements, text):
    """Add "No Class" elements to a list of elements (see get_elements) """
    initial_idx = 0
    final_idx = len(text)

    # Add element at the beginning if it doesn't in index 0
    new_elements = []
    if elements[0][0] != initial_idx:
        starting_element = (0, elements[0][0] - 1, 'No Class')
        new_elements.append(starting_element)

    # Add element at the end if it doesn't in index "-1"
    if elements[-1][1] != final_idx:
        closing_element = (elements[-1][1] + 1, final_idx, 'No Class')
        new_elements.append(closing_element)

    elements += new_elements
    elements = sorted(elements, key=lambda x: x[0])

    # Add "No class" elements inbetween separated elements
    new_elements = []
    for i in range(1, len(elements) - 1):
        if elements[i][0] != elements[i - 1][1] + 1 and elements[i][0] != elements[i - 1][1]:
            new_element = (elements[i - 1][1] + 1, elements[i][0] - 1, 'No Class')
            new_elements.append(new_element)

    elements += new_elements
    elements = sorted(elements, key=lambda x: x[0])
    return elements

def get_text(path, a_id):
    a_file = f"{path}/{a_id}.txt"
    with open(a_file, "r") as fp:
        txt = fp.read()
    return txt

def get_samples(df, path, text_id, do_fill_gaps=True):
    """Create a dataframe of the sentences of the text_id, with columns text, label """
    text = get_text(path, text_id)
    elements = df[df['id'] == text_id][['discourse_start', 'discourse_end', 'discourse_type']].to_records(index=False).tolist()
    elements = fill_gaps(elements, text)
    sentences = []
    for start, end, class_ in elements:
        elem_sentences = nltk.sent_tokenize(text[int(start):int(end)])
        sentences += [(text_id, sentence, class_) for sentence in elem_sentences]
    df = pd.DataFrame(sentences, columns=['id', 'text', 'label'])
    df['label'] = df['label'].map(CLASS2ID).astype('int')
    return df

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

class FeedbackPrizeDataset(Dataset):
    def __init__(self, tokenized_ds):
        self.data = tokenized_ds

    def __getitem__(self, index):
        item = {k: self.data[k][index] for k in self.data.keys()}
        return item

    def __len__(self):
        return len(self.data['input_ids'])

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

def preprocessing():
    if model == 'naive-bayes' or model == 'lstm':
        x = []
        for text_id in tqdm(unique_ids):
            x.append(get_samples(df, TRAIN_PATH, text_id))
        df_temp = pd.concat(x)
        df_temp = df_temp[df_temp.text.str.split().str.len() >= 3]
        print(df_temp.head())
        train_ids = np.random.choice(unique_ids, int(len(unique_ids) * 0.9), replace=False)
        val_ids = np.setdiff1d(unique_ids, train_ids)
        df_train = df_temp.loc[df_temp['id'].isin(train_ids)].reset_index(drop=True)
        df_val = df_temp.loc[df_temp['id'].isin(val_ids)].reset_index(drop=True)
        if model == 'naive-bayes':
            return df_train, df_val
        elif model == 'lstm':
            lstm_tokenizer = Tokenizer()
            lstm_tokenizer.fit_on_texts(df_train['text'])
            y_train = pd.get_dummies(df_train['label'])
            X_train = lstm_tokenizer.texts_to_sequences(df_train['text'])
            X_train = pad_sequences(X_train, maxlen=max_len)
            y_val = pd.get_dummies(df_val['label'])
            X_val = lstm_tokenizer.texts_to_sequences(df_val['text'])
            X_val = pad_sequences(X_val, maxlen=max_len)
            return X_train, y_train, X_val, y_val
    elif model == 'roberta':
        train_names, train_texts = [], []
        for f in tqdm(list(os.listdir(TRAIN_PATH))):
            train_names.append(f.replace('.txt', ''))
            train_texts.append(open(TRAIN_PATH + f, 'r').read())
            df_texts = pd.DataFrame({'id': train_names, 'text': train_texts})
        df_texts['text_split'] = df_texts.text.str.split()
        all_entities = []
        for _, row in tqdm(df_texts.iterrows(), total=len(df_texts)):
            total = len(row['text_split'])
            entities = ["O"] * total
            for _, row2 in df[df['id'] == row['id']].iterrows():
                discourse = row2['discourse_type']
                list_ix = [int(x) for x in row2['predictionstring'].split(' ')]
                entities[list_ix[0]] = f"B-{discourse}"
                for k in list_ix[1:]: entities[k] = f"I-{discourse}"
            all_entities.append(entities)
        df_texts['entities'] = all_entities
        np.random.seed(42)
        train_idx = np.random.choice(np.arange(len(unique_ids)), int(0.9 * len(unique_ids)), replace=False)
        valid_idx = np.setdiff1d(np.arange(len(unique_ids)), train_idx)
        np.random.seed(None)
        df_train = df_texts.loc[df_texts['id'].isin(unique_ids[train_idx])].reset_index(drop=True)
        df_val = df_texts.loc[df_texts['id'].isin(unique_ids[valid_idx])].reset_index(drop=True)
        download_model()
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        tokenized_train = tokenize(df_train, tokenizer)
        tokenized_val = tokenize(df_val, tokenizer)
        ds_train = FeedbackPrizeDataset(tokenized_train)
        dl_train = DataLoader(ds_train, batch_size=config['train_batch_size'],
                              shuffle=True, num_workers=1)

        ds_val = FeedbackPrizeDataset(tokenized_val)
        dl_val = DataLoader(ds_val, batch_size=config['valid_batch_size'],
                            num_workers=1)
        return df_val, dl_train, dl_val, valid_idx

def get_model():
    if model == 'naive-bayes':
        nb_pipeline = Pipeline([('vect', CountVectorizer()),
                                ('tfidf', TfidfTransformer()),
                                ('clf', MultinomialNB())])
        return nb_pipeline
    elif model == 'lstm':
        lstm_model = Sequential()
        lstm_model.add(Embedding(max_words, EMBEDDING_DIM, input_length=max_len))
        lstm_model.add(LSTM(32, return_sequences=True))
        lstm_model.add(LSTM(32, return_sequences=True))
        lstm_model.add(LSTM(32, return_sequences=True))
        lstm_model.add(LSTM(32, return_sequences=True))
        lstm_model.add(LSTM(32))
        lstm_model.add(Dense(8, activation='softmax'))
        lstm_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return lstm_model
    elif model == 'roberta':
        config_model = AutoConfig.from_pretrained(MODEL_PATH + '/config.json')
        roberta_model = AutoModelForTokenClassification.from_pretrained(
            MODEL_PATH + '/pytorch_model.bin', config=config_model)
        roberta_model.to(config['device'])
        optimizer = torch.optim.Adam(params=roberta_model.parameters(), lr=config['learning_rates'][0])
        return roberta_model, optimizer

def train(roberta_model, optimizer, dl_train, epoch):
    time_start = time.time()

    # Set learning rate to the one in config for this epoch
    for g in optimizer.param_groups:
        g['lr'] = config['learning_rates'][epoch]
    lr = optimizer.param_groups[0]['lr']

    epoch_prefix = f"[Epoch {epoch + 1:2d} / {config['epochs']:2d}]"
    print(f"{epoch_prefix} Starting epoch {epoch + 1:2d} with LR = {lr}")

    # Put model in training mode
    roberta_model.train()

    # Accumulator variables
    tr_loss, tr_accuracy = 0, 0
    nb_tr_examples, nb_tr_steps = 0, 0
    for idx, batch in enumerate(dl_train):

        ids = batch['input_ids'].to(config['device'], dtype=torch.long)
        mask = batch['attention_mask'].to(config['device'], dtype=torch.long)
        labels = batch['labels'].to(config['device'], dtype=torch.long)

        loss, tr_logits = roberta_model(input_ids=ids, attention_mask=mask, labels=labels,
                                return_dict=False)
        tr_loss += loss.item()

        nb_tr_steps += 1
        nb_tr_examples += labels.size(0)

        if idx % 200 == 0:
            loss_step = tr_loss / nb_tr_steps
            print(f"{epoch_prefix}     Steps: {idx:4d} --> Loss: {loss_step:.4f}")

        # compute training accuracy
        flattened_targets = labels.view(-1)  # shape (batch_size * seq_len,)
        active_logits = tr_logits.view(-1, roberta_model.config.num_labels)  # shape (batch_size * seq_len, num_labels)
        flattened_predictions = torch.argmax(active_logits, axis=1)  # shape (batch_size * seq_len,)

        # only compute accuracy at active labels
        active_accuracy = labels.view(-1) != -100  # shape (batch_size, seq_len)

        labels = torch.masked_select(flattened_targets, active_accuracy)
        predictions = torch.masked_select(flattened_predictions, active_accuracy)

        tmp_tr_accuracy = accuracy_score(labels.cpu().numpy(), predictions.cpu().numpy())
        tr_accuracy += tmp_tr_accuracy

        # gradient clipping
        torch.nn.utils.clip_grad_norm_(
            parameters=roberta_model.parameters(), max_norm=config['max_grad_norm']
        )

        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    epoch_loss = tr_loss / nb_tr_steps
    tr_accuracy = tr_accuracy / nb_tr_steps

    torch.save(roberta_model.state_dict(), f'pytorch_model_e{epoch}.bin')
    torch.cuda.empty_cache()
    gc.collect()

    elapsed = time.time() - time_start

    print(epoch_prefix)
    print(f"{epoch_prefix} Training loss    : {epoch_loss:.4f}")
    print(f"{epoch_prefix} Training accuracy: {tr_accuracy:.4f}")
    print(f"{epoch_prefix} Model saved to pytorch_model_e{epoch}.bin  [{elapsed / 60:.2f} mins]")
    print(epoch_prefix)
    del ids, mask, labels

def inference(dl):
    # These 2 dictionaries will hold text-level data
    # Helping in the merging process by accumulating data
    # Through all the chunks
    predictions = defaultdict(list)
    seen_words_idx = defaultdict(list)

    for batch in dl:
        ids = batch["input_ids"].to(config['device'])
        mask = batch["attention_mask"].to(config['device'])
        outputs = roberta_model(ids, attention_mask=mask, return_dict=False)
        del ids, mask

        batch_preds = torch.argmax(outputs[0], axis=-1).cpu().numpy()

        # Go over each prediction, getting the text_id reference
        for k, (chunk_preds, text_id) in enumerate(zip(batch_preds, batch['overflow_to_sample_mapping'].tolist())):

            # The word_ids are absolute references in the original text
            word_ids = batch['wids'][k].numpy()

            # Map from ids to labels
            chunk_preds = [IDS_TO_LABELS[i] for i in chunk_preds]

            for idx, word_idx in enumerate(word_ids):
                if word_idx == -1:
                    pass
                elif word_idx not in seen_words_idx[text_id]:
                    # Add predictions if the word doesn't have a prediction from a previous chunk
                    predictions[text_id].append(chunk_preds[idx])
                    seen_words_idx[text_id].append(word_idx)

    final_predictions = [predictions[k] for k in sorted(predictions.keys())]
    return final_predictions

def get_predictions(df, dl):
    all_labels = inference(dl)
    final_preds = []

    for i in range(len(df)):
        idx = df.id.values[i]
        pred = all_labels[i]
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
            if cls != 'O' and cls != '' and end - j > 7:
                final_preds.append((idx, cls.replace('I-', ''),
                                    ' '.join(map(str, list(range(j, end))))))
            j = end

    df_pred = pd.DataFrame(final_preds)
    df_pred.columns = ['id', 'class', 'predictionstring']
    return df_pred

def calc_overlap(row):
    """
    Calculates the overlap between prediction and
    ground truth and overlap percentages used for determining
    true positives.
    """
    set_pred = set(row.predictionstring_pred.split(' '))
    set_gt = set(row.predictionstring_gt.split(' '))
    # Length of each and intersection
    len_gt = len(set_gt)
    len_pred = len(set_pred)
    inter = len(set_gt.intersection(set_pred))
    overlap_1 = inter / len_gt
    overlap_2 = inter / len_pred
    return [overlap_1, overlap_2]

def score_feedback_comp(pred_df, gt_df):
    """
    A function that scores for the kaggle
        Student Writing Competition

    Uses the steps in the evaluation page here:
        https://www.kaggle.com/c/feedback-prize-2021/overview/evaluation
    """
    gt_df = gt_df[['id', 'discourse_type', 'predictionstring']].reset_index(drop=True).copy()
    pred_df = pred_df[['id', 'class', 'predictionstring']].reset_index(drop=True).copy()
    pred_df['pred_id'] = pred_df.index
    gt_df['gt_id'] = gt_df.index
    # Step 1. all ground truths and predictions for a given class are compared.
    joined = pred_df.merge(gt_df,
                           left_on=['id', 'class'],
                           right_on=['id', 'discourse_type'],
                           how='outer',
                           suffixes=('_pred', '_gt')
                           )
    joined['predictionstring_gt'] = joined['predictionstring_gt'].fillna(' ')
    joined['predictionstring_pred'] = joined['predictionstring_pred'].fillna(' ')

    joined['overlaps'] = joined.apply(calc_overlap, axis=1)

    # 2. If the overlap between the ground truth and prediction is >= 0.5,
    # and the overlap between the prediction and the ground truth >= 0.5,
    # the prediction is a match and considered a true positive.
    # If multiple matches exist, the match with the highest pair of overlaps is taken.
    joined['overlap1'] = joined['overlaps'].apply(lambda x: eval(str(x))[0])
    joined['overlap2'] = joined['overlaps'].apply(lambda x: eval(str(x))[1])

    joined['potential_TP'] = (joined['overlap1'] >= 0.5) & (joined['overlap2'] >= 0.5)
    joined['max_overlap'] = joined[['overlap1', 'overlap2']].max(axis=1)
    tp_pred_ids = joined.query('potential_TP').sort_values('max_overlap', ascending=False).groupby(['id', 'predictionstring_gt']).first()['pred_id'].values

    # 3. Any unmatched ground truths are false negatives
    # and any unmatched predictions are false positives.
    fp_pred_ids = [p for p in joined['pred_id'].unique() if p not in tp_pred_ids]

    matched_gt_ids = joined.query('potential_TP')['gt_id'].unique()
    unmatched_gt_ids = [c for c in joined['gt_id'].unique() if c not in matched_gt_ids]

    # Get numbers of each type
    TP = len(tp_pred_ids)
    FP = len(fp_pred_ids)
    FN = len(unmatched_gt_ids)
    TN = len(matched_gt_ids) + len(unmatched_gt_ids) - TP - FP - FN
    # calc microf1
    my_f1_score = TP / (TP + 0.5 * (FP + FN))
    my_acc = (TP + TN)/(TP+TN+FP+FN)
    return my_f1_score, my_acc

def validate(roberta_model, df_all, df_val, dl_val, epoch, valid_idx):
    time_start = time.time()

    # Put model in eval model
    roberta_model.eval()

    # Valid targets: needed because df_val has a subset of the columns
    df_valid = df_all.loc[df_all['id'].isin(unique_ids[valid_idx])]

    # OOF predictions
    oof = get_predictions(df_val, dl_val)

    # Compute F1-score
    f1s = []
    accs = []
    classes = oof['class'].unique()

    epoch_prefix = f"[Epoch {epoch + 1:2d} / {config['epochs']:2d}]"
    print(f"{epoch_prefix} Validation F1 scores")

    for c in classes:
        pred_df = oof.loc[oof['class'] == c].copy()
        gt_df = df_valid.loc[df_valid['discourse_type'] == c].copy()
        f1, acc = score_feedback_comp(pred_df, gt_df)
        print(f"{epoch_prefix}   * {c:<10} \t F1: {f1:4f} \t Accuracy: {acc:4f}")
        f1s.append(f1)
        accs.append(acc)

    elapsed = time.time() - time_start
    print(epoch_prefix)
    print(f'{epoch_prefix} Overall Validation F1: {np.mean(f1s):.4f} [{elapsed:.2f} secs]')
    print(f'{epoch_prefix} Overall Validation Accuracy: {np.mean(accs):.4f} [{elapsed:.2f} secs]')
    print(epoch_prefix)

if model == 'naive-bayes':
    df_train, df_val = preprocessing()
    nb_pipeline = get_model()
    nb_model = nb_pipeline.fit(df_train["text"], df_train["label"])
    pickle.dump(nb_model, open('naive_bayes_model.pkl','wb'))
    df_val['predictions'] = nb_model.predict(df_val['text'])
    df_val['pred_class'] = df_val['predictions'].map(ID2CLASS)
    print(f'Validation accuracy= {accuracy_score(df_val["predictions"], df_val["label"])}')
    print(f'Validation F1 score= {f1_score(df_val["predictions"], df_val["label"], average="macro")}')

elif model == 'lstm':
    X_train, y_train, X_val, y_val = preprocessing()
    lstm_model = get_model()
    batch_size = 1024
    lstm_model.fit(X_train, y_train,
              epochs=50,
              batch_size=batch_size,
              validation_split=0.1,
              callbacks=[ModelCheckpoint('lstm_model.h5', monitor='val_loss', save_best_only=True, save_weights_only=True)])
    y_pred = lstm_model.predict(X_val)
    y_val = list(y_val.idxmax(axis = 1))
    y_pred = list(pd.DataFrame(y_pred).idxmax(axis = 1))
    print(f'Validation accuracy= {accuracy_score(y_pred, y_val)}')
    print(f'Validation F1 score= {f1_score(y_pred, y_val, average="macro")}')
elif model == 'roberta':
    df_val, dl_train, dl_val, valid_idx = preprocessing()
    roberta_model, optimizer = get_model()
    for epoch in range(config['epochs']):
        train(roberta_model, optimizer, dl_train, epoch)
        validate(roberta_model, df, df_val, dl_val, epoch, valid_idx)
print()

