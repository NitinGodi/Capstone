# Import packages
import os
import gc
import time
import pandas as pd
import numpy as np
from collections import defaultdict
from tqdm.auto import tqdm
import nltk
from sklearn.metrics import accuracy_score, f1_score
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForTokenClassification, RobertaTokenizerFast
import json
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_

# Load and prepare configuration variables
f = open('Feedback_config.json')
CONF = json.load(f)
LABELS_TO_IDS = {v:k for k,v in enumerate(CONF['OUTPUT_LABELS'])}
IDS_TO_LABELS = {k:v for k,v in enumerate(CONF['OUTPUT_LABELS'])}
device = torch.device("cuda")

# Assign label to sections of essay that do not belong to any class
def fill_gaps(elements, text):
    """Function to Assign label to sections of essay that do not belong to any class"""
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

# Load the essay in a txt file
def get_text(path, a_id):
    a_file = f"{path}/{a_id}.txt"
    with open(a_file, "r") as fp:
        txt = fp.read()
    return txt

# Create a dataframe of the sentences of the text_id, with columns text, label
def get_samples(df, path, text_id):
    """Create a dataframe of the sentences of the text_id, with columns text, label """
    text = get_text(path, text_id)
    elements = df[df['id'] == text_id][['discourse_start', 'discourse_end', 'discourse_type']].to_records(index=False).tolist()
    elements = fill_gaps(elements, text)
    sentences = []
    for start, end, class_ in elements:
        elem_sentences = nltk.sent_tokenize(text[int(start):int(end)])
        sentences += [(text_id, sentence, class_) for sentence in elem_sentences]
    df = pd.DataFrame(sentences, columns=['id', 'text', 'label'])
    df['label'] = df['label'].map(CONF['CLASS2ID']).astype('int')
    return df

# Assign label to each word
def get_labels(word_ids, word_labels):
    label_ids = []
    for word_idx in word_ids:
        if word_idx is None:
            label_ids.append(-100)
        else:
            label_ids.append(LABELS_TO_IDS[word_labels[word_idx]])
    return label_ids

# Tokenization of data for the Transform model
def tokenize(df, tokenizer, to_tensor=True, with_labels=True):
    # Encode discourses
    encoded = tokenizer(df['text_split'].tolist(),
                        is_split_into_words=True,
                        return_overflowing_tokens=True,
                        stride=CONF['DOC_STRIDE'],
                        max_length=CONF['MAX_LEN'],
                        padding="max_length",
                        truncation=True)

    # Encoding corresponding labels
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

    # Converting all key value pair into tensors
    if to_tensor:
        encoded = {key: torch.as_tensor(val) for key, val in encoded.items()}
    return encoded

# Dataset class for Transformer model
class FeedbackPrizeDataset(Dataset):
    def __init__(self, tokenized_ds):
        self.data = tokenized_ds

    def __getitem__(self, index):
        item = {k: self.data[k][index] for k in self.data.keys()}
        return item

    def __len__(self):
        return len(self.data['input_ids'])

# Loading and preprocessing data for different models
def preprocessing(model):
    # Loading the train.csv file
    df = pd.read_csv(CONF['TRAIN_CSV'])
    unique_ids = df['id'].unique()
    # Data preprocessing for Naive-Bayes model
    if model == 'naive-bayes':
        x = []
        # Loading and processing data
        for text_id in tqdm(unique_ids):
            x.append(get_samples(df, CONF['TRAIN_PATH'], text_id))
        df_temp = pd.concat(x)
        df_temp = df_temp[df_temp.text.str.split().str.len() >= 3]
        print(df_temp.head())
        # Spiting data into training and validation sets
        train_ids = np.random.choice(unique_ids, int(len(unique_ids) * 0.9), replace=False)
        val_ids = np.setdiff1d(unique_ids, train_ids)
        df_train = df_temp.loc[df_temp['id'].isin(train_ids)].reset_index(drop=True)
        df_val = df_temp.loc[df_temp['id'].isin(val_ids)].reset_index(drop=True)
        return df_train, df_val
    # Data Preprocessing for LSTM model
    elif model == 'lstm':
        # Getting word vocabulary and their ids
        tokenizer = RobertaTokenizerFast.from_pretrained(CONF['MODEL_NAME'], add_prefix_space=True)
        word_to_id = tokenizer.vocab
        # Mapper classes for words and labels
        word_vocab = WordVocabulary(word_to_id)
        label_vocab = LabelVocabulary(CONF['OUTPUT_LABELS'])
        # Loading glove word embeddings
        pretrain_word_embedding = build_pretrain_embedding(CONF['pretrain_embed_path'], word_vocab,
                                                           CONF['word_embed_dim'])
        # Splitting data into training and validation sets
        train_ids = np.random.choice(unique_ids, int(len(unique_ids) * 0.9), replace=False)
        val_ids = np.setdiff1d(unique_ids, train_ids)
        # Initializing training and validation datasets
        train_dataset = MyDataset(CONF['TRAIN_PATH'], word_vocab, label_vocab, train_ids, df)
        val_dataset = MyDataset(CONF['TRAIN_PATH'], word_vocab, label_vocab, val_ids, df)
        # Initializing training and validation data loaders
        train_dataloader = DataLoader(train_dataset, batch_size=CONF['batch_size'], shuffle=True,
                                      collate_fn=my_collate_fn)
        val_dataloader = DataLoader(val_dataset, batch_size=CONF['batch_size'], shuffle=False, collate_fn=my_collate_fn)
        return pretrain_word_embedding, train_dataloader, val_dataloader, word_vocab, label_vocab
    # Data Preprocessing for Transformer model
    elif model == 'roberta':
        # Loading essays from txt files
        train_names, train_texts = [], []
        for f in tqdm(list(os.listdir(CONF['TRAIN_PATH']))):
            train_names.append(f.replace('.txt', ''))
            train_texts.append(open(CONF['TRAIN_PATH'] + f, 'r').read())
            df_texts = pd.DataFrame({'id': train_names, 'text': train_texts})
        # Splitting essays into words
        df_texts['text_split'] = df_texts.text.str.split()
        all_entities = []
        # Assigning labels to all the words of the essays
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
        # Splitting training and validation sets
        np.random.seed(42)
        train_idx = np.random.choice(np.arange(len(unique_ids)), int(0.9 * len(unique_ids)), replace=False)
        valid_idx = np.setdiff1d(np.arange(len(unique_ids)), train_idx)
        np.random.seed(None)
        df_train = df_texts.loc[df_texts['id'].isin(unique_ids[train_idx])].reset_index(drop=True)
        df_val = df_texts.loc[df_texts['id'].isin(unique_ids[valid_idx])].reset_index(drop=True)
        # Initializing tokenizer
        tokenizer = AutoTokenizer.from_pretrained(CONF['MODEL_NAME'], add_prefix_space=True)
        # Tokenizing training and validation data
        tokenized_train = tokenize(df_train, tokenizer)
        tokenized_val = tokenize(df_val, tokenizer)
        # Initializing training and validation datasets
        ds_train = FeedbackPrizeDataset(tokenized_train)
        dl_train = DataLoader(ds_train, batch_size=CONF['CONFIG']['train_batch_size'],
                              shuffle=True, num_workers=1)
        # Initializing training and validation data loaders
        ds_val = FeedbackPrizeDataset(tokenized_val)
        dl_val = DataLoader(ds_val, batch_size=CONF['CONFIG']['valid_batch_size'],
                            num_workers=1)
        return {'df':df, 'df_val':df_val, 'dl_train':dl_train, 'dl_val':dl_val, 'valid_idx':valid_idx, 'unique_ids':unique_ids}

# Training Transformer model
def train(roberta_model, optimizer, dl_train, epoch):
    time_start = time.time()

    # Set learning rate to the one in config for this epoch
    for g in optimizer.param_groups:
        g['lr'] = CONF['CONFIG']['learning_rates'][epoch]
    lr = optimizer.param_groups[0]['lr']

    epoch_prefix = f"[Epoch {epoch + 1:2d} / {CONF['CONFIG']['epochs']:2d}]"
    print(f"{epoch_prefix} Starting epoch {epoch + 1:2d} with LR = {lr}")

    # Put model in training mode
    roberta_model.train()

    # Accumulator variables
    tr_loss, tr_accuracy = 0, 0
    nb_tr_examples, nb_tr_steps = 0, 0
    for idx, batch in enumerate(dl_train):
        # Load input ids, attention mask and labels into the device
        ids = batch['input_ids'].to(CONF['CONFIG']['device'], dtype=torch.long)
        mask = batch['attention_mask'].to(CONF['CONFIG']['device'], dtype=torch.long)
        labels = batch['labels'].to(CONF['CONFIG']['device'], dtype=torch.long)
        # Forward Pass
        loss, tr_logits = roberta_model(input_ids=ids, attention_mask=mask, labels=labels,
                                return_dict=False)
        tr_loss += loss.item()

        nb_tr_steps += 1
        nb_tr_examples += labels.size(0)
        # Calculating loss after every 200 steps
        if idx % 200 == 0:
            loss_step = tr_loss / nb_tr_steps
            print(f"{epoch_prefix}     Steps: {idx:4d} --> Loss: {loss_step:.4f}")

        # Computing training accuracy at active labels
        flattened_targets = labels.view(-1)
        active_logits = tr_logits.view(-1, roberta_model.config.num_labels)
        flattened_predictions = torch.argmax(active_logits, axis=1)
        active_accuracy = labels.view(-1) != -100
        labels = torch.masked_select(flattened_targets, active_accuracy)
        predictions = torch.masked_select(flattened_predictions, active_accuracy)
        tmp_tr_accuracy = accuracy_score(labels.cpu().numpy(), predictions.cpu().numpy())
        tr_accuracy += tmp_tr_accuracy

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            parameters=roberta_model.parameters(), max_norm=CONF['CONFIG']['max_grad_norm']
        )

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # Computing epoch loss and accuracy
    epoch_loss = tr_loss / nb_tr_steps
    tr_accuracy = tr_accuracy / nb_tr_steps
    torch.cuda.empty_cache()
    gc.collect()

    elapsed = time.time() - time_start

    print(epoch_prefix)
    print(f"{epoch_prefix} Training loss    : {epoch_loss:.4f}")
    print(f"{epoch_prefix} Training accuracy: {tr_accuracy:.4f}")
    print(f"{epoch_prefix} Model saved to pytorch_model_e{epoch}.bin  [{elapsed / 60:.2f} mins]")
    print(epoch_prefix)
    del ids, mask, labels

# Predicting labels for validation data using Transformer model
def inference(roberta_model, dl):
    predictions = defaultdict(list)
    seen_words_idx = defaultdict(list)

    for batch in dl:
        # Load input ids and mask into the device
        ids = batch["input_ids"].to(CONF['CONFIG']['device'])
        mask = batch["attention_mask"].to(CONF['CONFIG']['device'])
        # Predict Logits
        outputs = roberta_model(ids, attention_mask=mask, return_dict=False)
        del ids, mask
        # Getting predicted label ids
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

# Getting predictions for validation data using Transformer model and documenting the predictions
def get_predictions(roberta_model, df, dl):
    # Getting predictions
    all_labels = inference(roberta_model, dl)
    final_preds = []
    # Chunks, consisting more than 4 words, having same labels are documented
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
            if cls != 'O' and cls != '' and end - j > 4:
                final_preds.append((idx, cls.replace('I-', ''),
                                    ' '.join(map(str, list(range(j, end))))))
            j = end

    df_pred = pd.DataFrame(final_preds)
    df_pred.columns = ['id', 'class', 'predictionstring']
    return df_pred

# Calculate percentage of overlap of ground truth and prediction
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

# Calculate F1 score and accuracy of the Transformer model
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

# Validation process for the Transformer model
def validate(roberta_model, df_all, df_val, dl_val, epoch, valid_idx, unique_ids):
    time_start = time.time()

    # Put model in eval model
    roberta_model.eval()

    # Valid targets: needed because df_val has a subset of the columns
    df_valid = df_all.loc[df_all['id'].isin(unique_ids[valid_idx])]

    # OOF predictions
    oof = get_predictions(roberta_model, df_val, dl_val)

    # Compute F1-score ans Accuracy
    f1s = []
    accs = []
    classes = oof['class'].unique()

    epoch_prefix = f"[Epoch {epoch + 1:2d} / {CONF['CONFIG']['epochs']:2d}]"
    print(f"{epoch_prefix} Validation F1 scores")

    for c in classes:
        pred_df = oof.loc[oof['class'] == c].copy()
        gt_df = df_valid.loc[df_valid['discourse_type'] == c].copy()
        f1, acc = score_feedback_comp(pred_df, gt_df)
        print(f"{epoch_prefix}   * {c:<10} \t F1: {f1:4f} \t Accuracy: {acc:4f}")
        f1s.append(f1)
        accs.append(acc)

    epoch_f1 = np.mean(f1s)
    elapsed = time.time() - time_start
    print(epoch_prefix)
    print(f'{epoch_prefix} Overall Validation F1: {epoch_f1:.4f} [{elapsed:.2f} secs]')
    print(f'{epoch_prefix} Overall Validation Accuracy: {np.mean(accs):.4f} [{elapsed:.2f} secs]')
    print(epoch_prefix)
    return epoch_f1

# Load state dictionary of Transformer model
def load_model():
    model = AutoModelForTokenClassification.from_pretrained(CONF['MODEL_NAME'])
    model.load_state_dict(torch.load(f'{CONF['MODEL_PATH']}/best_roberta_model.bin', map_location='cpu'))
    model.eval()
    print('Model loaded.')
    return model

# Predict labels for essay inputted from the app
def app_inference(batch):
    predictions = defaultdict(list)
    seen_words_idx = defaultdict(list)

    # Load state dictionary of the saved model
    roberta_model = load_model()

    # Load input ids and mask into the model
    ids = batch["input_ids"]
    mask = batch["attention_mask"]

    # Predict the output labels
    outputs = roberta_model(ids, attention_mask=mask, return_dict=False)
    batch_preds = torch.argmax(outputs[0], axis=-1).cpu().numpy()

    # Store predicted labels
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

# Using Transformer model, predict and return labels for essay inputted from the app
def app_get_predictions(tokenized_text):
    pred = app_inference(tokenized_text)[0]
    final_preds = []
    j = 0
    # Chunks, consisting more than 4 words, having same labels are documented
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

# Dataset class for the LSTM model
class MyDataset(Dataset):
    def __init__(self, file_path, word_vocab, label_vocab, ids, df):
        # Initializing word and label vocab mappers
        self.word_vocab = word_vocab
        self.label_vocab = label_vocab
        self.ids = ids
        # Loading essays from txt files
        train_names, train_texts = [], []
        for f in ids:
            train_names.append(f)
            train_texts.append(open(file_path + f + '.txt', 'r').read())
            df_texts = pd.DataFrame({'id': train_names, 'text': train_texts})
        # Splitting the essays into words
        df_texts['text_split'] = df_texts.text.str.split()
        all_entities = []
        # Assigning labels for each word
        for _, row in df_texts.iterrows():
            total = len(row['text_split'])
            entities = ["O"] * total
            for _, row2 in df[df['id'] == row['id']].iterrows():
                discourse = row2['discourse_type']
                list_ix = [int(x) for x in row2['predictionstring'].split(' ')]
                entities[list_ix[0]] = f"B-{discourse}"
                for k in list_ix[1:]: entities[k] = f"I-{discourse}"
            all_entities.append(entities)
        df_texts['entities'] = all_entities
        # Storing essays split into words and their corresponding labels
        self.texts = df_texts['text_split'].values
        self.labels = df_texts['entities'].values

    def __len__(self):
        # Returns number of essays in the dataset
        return len(self.labels)

    def __getitem__(self, item):
        # Returns words and corresponding labels of the requested essay
        text_id = []
        label_id = []
        text = self.texts[item]
        label = self.labels[item]
        for word in text:
            text_id.append(self.word_vocab.word_to_id(word))
        text_tensor = torch.tensor(text_id).long()
        for label_ele in label:
            label_id.append(self.label_vocab.label_to_id(label_ele))
        label_tensor = torch.tensor(label_id).long()

        return {'text': text_tensor, 'label': label_tensor}

# Mapper Class for words and their ids
class WordVocabulary(object):
    def __init__(self, word_to_id):
        # Initialize word to id and id to word mapper dictionaries
        self._word_to_id = word_to_id
        self._id_to_word = {v:k for k,v in self._word_to_id.items()}

    def unk(self):
        # Returns id of unknown token
        return self._word_to_id['<unk>']

    def pad(self):
        # Returns id of pad token
        return self._word_to_id['<pad>']

    def size(self):
        # Returns number of words in the vocabulary
        return len(self._id_to_word)

    def word_to_id(self, word):
        # Returns id of requested word
        if word in self._word_to_id.keys():
            return self._word_to_id[word]
        elif 'Ġ' + word in self._word_to_id.keys():
            return self.word_to_id('Ġ' + word)
        elif 'Ġ' + word.lower() in self._word_to_id.keys():
            return self.word_to_id('Ġ' + word.lower())
        else:
            return self.word_to_id('<unk>')

    def id_to_word(self, cur_id):
        # Returns word corresponding to the requested id
        if 'Ġ' in self._id_to_word[cur_id]:
            return self._id_to_word[cur_id].replace('Ġ', '')
        return self._id_to_word[cur_id]

    def items(self):
        # Returns the dictionary with all the words and their corresponding ids
        return self._word_to_id.items()

# Mapper class for the labels and their ids
class LabelVocabulary(object):
    def __init__(self, labels):
        # Initialize the label to id and id to label mapper dictionaries
        self._label_to_id = {k:v for v,k in enumerate(labels)}
        self._id_to_label = {v:k for k,v in self._label_to_id.items()}

    def size(self):
        # Returns the number of labels in the vocabulary
        return len(self._id_to_label)

    def label_to_id(self, label):
        # Returns the id of the requested label
        return self._label_to_id[label]

    def id_to_label(self, cur_id):
        # Returns the label of the corresponding id
        return self._id_to_label[cur_id]

# Custom collate function
def my_collate(batch_tensor):
    # List of lengths of each entry in the batch
    word_seq_lengths = torch.LongTensor(list(map(len, batch_tensor)))
    # List of indexes of entries in the descending order of their lengths
    _, word_perm_idx = word_seq_lengths.sort(0, descending=True)
    # Sort the batch entries in the descending order of their lengths
    batch_tensor.sort(key=lambda x: len(x), reverse=True)
    # List of lengths of entries in the batch
    tensor_length = [len(sq) for sq in batch_tensor]
    # Pad the sequences to the max length among the entries
    batch_tensor = pad_sequence(batch_tensor, batch_first=True, padding_value=0)
    return batch_tensor, tensor_length, word_perm_idx

# Custom collate function
def my_collate_fn(batch):
    # Call custom collate function for each key-value pair in the batch
    return {key: my_collate([d[key] for d in batch]) for key in batch[0]}

# Loading the pretrained word embeddings using Glove
def load_pretrain_emb(embedding_path):
    embedd_dim = 100
    embedd_dict = dict()
    # Load words and their embeddings from glove.6B.100d.txt file
    with open(embedding_path, 'r', encoding="utf8") as file:
        for line in file:
            line = line.strip()
            if len(line) == 0:
                continue
            tokens = line.split()
            if not embedd_dim + 1 == len(tokens):
                continue
            embedd = np.empty([1, embedd_dim])
            embedd[:] = tokens[1:]
            first_col = tokens[0]
            embedd_dict[first_col] = embedd
    return embedd_dict, embedd_dim

# Build embedding for all words in the vocabulary
def build_pretrain_embedding(embedding_path, word_vocab, embedd_dim=100):
    embedd_dict = dict()
    # Load pretrained embedding if available
    if embedding_path is not None:
        embedd_dict, embedd_dim = load_pretrain_emb(embedding_path)
    scale = np.sqrt(3.0 / embedd_dim)
    # Create empty embedding matrix
    pretrain_emb = np.empty([word_vocab.size(), embedd_dim])
    # Fill the embedding matrix with either existing word embeddings or new random embeddings
    for word, index in word_vocab.items():
        if word in embedd_dict:
            pretrain_emb[index, :] = embedd_dict[word]
        elif word.lower() in embedd_dict:
            pretrain_emb[index, :] = embedd_dict[word.lower()]
        else:
            pretrain_emb[index, :] = np.random.uniform(-scale, scale, [1, embedd_dim])
    return pretrain_emb

# Learning rae decay for the LSTM model
def lr_decay(optimizer, epoch, decay_rate, init_lr):
    lr = init_lr / (1 + decay_rate * epoch)
    print("Learning rate is set as:", lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer

# Get mask for the LSTM model
def get_mask(batch_tensor):
    mask = batch_tensor.eq(0)
    mask = mask.eq(0)
    return mask

# LSTM model
class NamedEntityRecog(nn.Module):
    def __init__(self, vocab_size, word_embed_dim, word_hidden_dim, tag_num, dropout, pretrain_embed=None):
        # Initialize different layers of the LSTM model
        super(NamedEntityRecog, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.input_dim = word_embed_dim
        self.embeds = nn.Embedding(vocab_size, word_embed_dim, padding_idx=0)
        if pretrain_embed is not None:
            self.embeds.weight.data.copy_(torch.from_numpy(pretrain_embed))
        self.lstm = nn.LSTM(self.input_dim, word_hidden_dim, batch_first=True, bidirectional=True)
        self.hidden2tag = nn.Linear(word_hidden_dim * 2, tag_num)

    def random_embedding(self, vocab_size, embedding_dim):
        # Returns random word embedding
        pretrain_emb = np.empty([vocab_size, embedding_dim])
        scale = np.sqrt(3.0 / embedding_dim)
        for index in range(1, vocab_size):
            pretrain_emb[index, :] = np.random.uniform(-scale, scale, [1, embedding_dim])
        return pretrain_emb

    def neg_log_likelihood_loss(self, word_inputs, word_seq_lengths, batch_label, mask):
        # Function for training the LSTM model
        # Get batch size and sequence lengths
        batch_size = word_inputs.size(0)
        seq_len = word_inputs.size(1)
        # Get word embeddings
        word_embeding = self.embeds(word_inputs.to(device))
        # Packing the padded sequences
        packed_words = pack_padded_sequence(word_embeding, word_seq_lengths, True)
        hidden = None
        # Passing the packed inputs to the LSTM model
        lstm_out, hidden = self.lstm(packed_words, hidden)
        # Unpacking the packed output
        lstm_out, _ = pad_packed_sequence(lstm_out)
        # Switching the first and the second dimensions of the output
        lstm_out = lstm_out.transpose(0, 1)
        # Dropping information from a few neurons
        feature_out = self.drop(lstm_out)
        # Get logits for all the label clases
        feature_out = self.hidden2tag(feature_out)
        # Calculate loss using Cross Entropy
        loss_function = nn.CrossEntropyLoss(ignore_index=0, reduction='sum')
        # Get the final loss and predicted labels for the inputted sequences of words
        feature_out = feature_out.contiguous().view(batch_size * seq_len, -1)
        total_loss = loss_function(feature_out, batch_label.contiguous().view(batch_size * seq_len).to(device))
        _, tag_seq = torch.max(feature_out, 1)
        tag_seq = tag_seq.view(batch_size, seq_len)
        tag_seq = mask.long().to(device) * tag_seq
        return total_loss, tag_seq

    def forward(self, word_inputs, word_seq_lengths, mask):
        # Function to validate the LSTM model
        # Get batch size and sequence lengths
        batch_size = word_inputs.size(0)
        seq_len = word_inputs.size(1)
        # Get word embeddings
        word_embeding = self.embeds(word_inputs.to(device))
        # Packing the padded sequences
        packed_words = pack_padded_sequence(word_embeding, word_seq_lengths, True)
        hidden = None
        # Passing the packed inputs to the LSTM model
        lstm_out, hidden = self.lstm(packed_words, hidden)
        # Unpacking the packed output
        lstm_out, _ = pad_packed_sequence(lstm_out)
        # Switching the first and the second dimensions of the output
        lstm_out = lstm_out.transpose(0, 1)
        # Dropping information from a few neurons
        feature_out = self.drop(lstm_out)
        # Get logits for all the label clases
        feature_out = self.hidden2tag(feature_out)
        # Get the predicted labels for the inputted sequences of words
        feature_out = feature_out.contiguous().view(batch_size * seq_len, -1)
        _, tag_seq = torch.max(feature_out, 1)
        tag_seq = tag_seq.view(batch_size, seq_len)
        tag_seq = mask.long().to(device) * tag_seq
        return tag_seq

    def predict(self, word_inputs):
        # Function to predict labels for the words in essay inputted from the app
        # Get sequence length
        seq_len = len(word_inputs)
        # Get word embeddings
        word_embeding = self.embeds(word_inputs)
        hidden = None
        # Pass the word embeddings into the lstm model
        lstm_out, hidden = self.lstm(word_embeding.view(1,seq_len,-1), hidden)
        # Switch the first and second dimensions of the output
        lstm_out = lstm_out.transpose(0, 1)
        # Drop information from some neurons
        feature_out = self.drop(lstm_out)
        # Get logits for all the label classes
        feature_out = self.hidden2tag(feature_out)
        # Get the final predictions
        feature_out = feature_out.contiguous().view(seq_len, -1)
        _, tag_seq = torch.max(feature_out, 1)
        tag_seq = tag_seq.view(1, seq_len)
        return tag_seq

# Training LSTM model
def train_model(dataloader, model, optimizer, use_gpu=True):
    # Put the model in training mode
    model.train()
    batch_num = 0
    f1 = []
    for batch in dataloader:
        batch_num += 1
        batch_f1 = 0
        # Set gradient to zero
        model.zero_grad()
        batch_text, seq_length, word_perm_idx = batch['text']
        batch_label, _, _ = batch['label']
        if use_gpu:
            # Load intput and target into the device
            batch_text = batch_text.to(device)
            batch_label = batch_label.to(device)
        mask = get_mask(batch_text)
        # Forward Pass
        loss, output = model.neg_log_likelihood_loss(batch_text, seq_length, batch_label, mask)
        # Backward pass
        loss.backward()
        clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        # Calculate training f1 score
        for i in range(len(batch_label)):
            gr_tr = torch.masked_select(batch_label[i], mask[i])
            pr_tr = torch.masked_select(output[i], mask[i])
            batch_f1 += f1_score(gr_tr.cpu(), pr_tr.cpu(), average='macro')
        f1.append(batch_f1/ len(mask))
    train_f1 = sum(f1) / batch_num
    print(f'Training F1 score: {train_f1}')

# Evaluating the LSTM model
def evaluate(dataloader, model, use_gpu=True):
    # Put model in evaluation mode
    model.eval()
    batch_num = 0
    f1 = []
    for batch in dataloader:
        batch_num += 1
        batch_f1 = 0
        batch_text, seq_length, word_perm_idx = batch['text']
        batch_label, _, _ = batch['label']
        if use_gpu:
            # Load the input into the device
            batch_text = batch_text.to(device)
            batch_label = batch_label.to(device)
        mask = get_mask(batch_text)
        with torch.no_grad():
            # Make predictions
            tag_seq = model(batch_text, seq_length, mask)
        # Calculate validation f1 score
        for i in range(len(batch_label)):
            gr_tr = torch.masked_select(batch_label[i], mask[i])
            pr_tr = torch.masked_select(tag_seq[i], mask[i])
            batch_f1 += f1_score(gr_tr.cpu(), pr_tr.cpu(), average='macro')
        f1.append(batch_f1 / len(mask))
    val_f1 = sum(f1) / batch_num
    print(f'Validation F1 score: {val_f1}')
    return val_f1
