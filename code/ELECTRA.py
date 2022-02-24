# CHANGES IN LINE 22

import os
import gc
import time
from tqdm import tqdm
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

import torch
from torch.utils.data import Dataset, DataLoader

from transformers import AutoConfig, AutoTokenizer, AutoModelForTokenClassification

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# This is used to download the model from the huggingface hub
# MODEL_NAME = 'bert-base-uncased'
MODEL_NAME = 'google/electra-small-discriminator'

# Path where to download the model
MODEL_PATH = 'model'


# Max length for the tokenization and the model
# For BERT-like models it's 512 in general
MAX_LENGTH = 512

# The overlapping tokens when chunking the texts
# Possibly a power of 2 would have been better
# Tried with 386 and didn't improve
DOC_STRIDE = 200

# Training configuration
# 5 epochs with different learning rates (inherited from Chris')
# Haven't tried variations yet
config = {'train_batch_size': 10,
          'valid_batch_size': 10,
          'epochs': 5,
          'learning_rates': [2.5e-5, 2.5e-5, 2.5e-6, 2.5e-6, 2.5e-7],
          'max_grad_norm': 10,
          'device': 'cuda'}

df_all = pd.read_csv('/home/ubuntu/capstone/train.csv')
print(df_all.shape)
print(df_all.head())

# https://www.kaggle.com/raghavendrakotala/fine-tunned-on-roberta-base-as-ner-problem-0-533
train_names, train_texts = [], []
for f in tqdm(list(os.listdir('/home/ubuntu/capstone/train'))):
    train_names.append(f.replace('.txt', ''))
    train_texts.append(open('/home/ubuntu/capstone//train/' + f, 'r').read())

    df_texts = pd.DataFrame({'id': train_names, 'text': train_texts})

df_texts['text_split'] = df_texts.text.str.split()
print(df_texts.head())

# df_texts = df_texts[:1000]

# https://www.kaggle.com/cdeotte/pytorch-bigbird-ner-cv-0-615
all_entities = []
for _, row in tqdm(df_texts.iterrows(), total=len(df_texts)):

    total = len(row['text_split'])
    entities = ["O"] * total

    for _, row2 in df_all[df_all['id'] == row['id']].iterrows():
        discourse = row2['discourse_type']
        list_ix = [int(x) for x in row2['predictionstring'].split(' ')]
        entities[list_ix[0]] = f"B-{discourse}"
        for k in list_ix[1:]: entities[k] = f"I-{discourse}"
    all_entities.append(entities)

df_texts['entities'] = all_entities
df_texts.to_csv('train_NER.csv', index=False)

print(df_texts.shape)
print(df_texts.head())

# Check that we have created one entity/label for each word correctly
print((df_texts['text_split'].str.len() == df_texts['entities'].str.len()).all())

# Create global dictionaries to use during training and inference

# https://www.kaggle.com/cdeotte/pytorch-bigbird-ner-cv-0-615
output_labels = ['O', 'B-Lead', 'I-Lead', 'B-Position', 'I-Position', 'B-Claim', 'I-Claim', 'B-Counterclaim', 'I-Counterclaim',
          'B-Rebuttal', 'I-Rebuttal', 'B-Evidence', 'I-Evidence', 'B-Concluding Statement', 'I-Concluding Statement']

LABELS_TO_IDS = {v:k for k,v in enumerate(output_labels)}
IDS_TO_LABELS = {k:v for k,v in enumerate(output_labels)}

print(LABELS_TO_IDS)

# CHOOSE VALIDATION INDEXES
IDS = df_all.id.unique()
print(f'There are {len(IDS)} train texts. We will split 90% 10% for validation.')

# TRAIN VALID SPLIT 90% 10%
np.random.seed(42)
train_idx = np.random.choice(np.arange(len(IDS)),int(0.9*len(IDS)),replace=False)
valid_idx = np.setdiff1d(np.arange(len(IDS)),train_idx)
np.random.seed(None)

# CREATE TRAIN SUBSET AND VALID SUBSET
df_train = df_texts.loc[df_texts['id'].isin(IDS[train_idx])].reset_index(drop=True)
df_val = df_texts.loc[df_texts['id'].isin(IDS[valid_idx])].reset_index(drop=True)

print(f"FULL Dataset : {df_texts.shape}")
print(f"TRAIN Dataset: {df_train.shape}")
print(f"TEST Dataset : {df_val.shape}")


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


download_model()

# Instantiate the tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)


# This function is a simple map between text_split and entities
# We have verified that we have a 1:1 mapping above
# See above: (df_texts['text_split'].str.len() == df_texts['entities'].str.len()).all() == True
def get_labels(word_ids, word_labels):
    label_ids = []
    for word_idx in word_ids:
        if word_idx is None:
            label_ids.append(-100)
        else:
            label_ids.append(LABELS_TO_IDS[word_labels[word_idx]])
    return label_ids


# Tokenize texts, possibly generating more than one tokenized sample for each text
def tokenize(df, to_tensor=True, with_labels=True):
    # This is what's different from a longformer
    # Read the parameters with attention
    encoded = tokenizer(df['text_split'].tolist(),
                        is_split_into_words=True,
                        return_overflowing_tokens=True,
                        stride=DOC_STRIDE,
                        max_length=MAX_LENGTH,
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


# Tokenize both training and validation dataframes
tokenized_train = tokenize(df_train)
tokenized_val = tokenize(df_val)

# Original number of rows
print(len(df_train))

# Number of samples generated when chunking
print(len(tokenized_train['input_ids']))

# Back-reference.
# The first 2 ones mean that the second row was split into 2 samples
# And the 3 twos mean that the third row was split into 3 samples
print(tokenized_train['overflow_to_sample_mapping'][:10])

# Further exploration of the cases for those who are interested:

## Original text:
#print(df_train.iloc[1]['text'])
## The four 512-token chunks generated by the tokenization procedure:
#print(tokenizer.decode(tokenized_train['input_ids'][1]))
#print("========")
#print(tokenizer.decode(tokenized_train['input_ids'][2]))

class FeedbackPrizeDataset(Dataset):
    def __init__(self, tokenized_ds):
        self.data = tokenized_ds

    def __getitem__(self, index):
        item = {k: self.data[k][index] for k in self.data.keys()}
        return item

    def __len__(self):
        return len(self.data['input_ids'])


# Create Datasets and DataLoaders for training and validation dat

ds_train = FeedbackPrizeDataset(tokenized_train)
dl_train = DataLoader(ds_train, batch_size=config['train_batch_size'],
                      shuffle=True, num_workers=1, pin_memory=False)

ds_val = FeedbackPrizeDataset(tokenized_val)
dl_val = DataLoader(ds_val, batch_size=config['valid_batch_size'],
                    shuffle=False, num_workers=1, pin_memory=False)


# https://www.kaggle.com/raghavendrakotala/fine-tunned-on-roberta-base-as-ner-problem-0-533
def train(model, optimizer, dl_train, epoch):
    time_start = time.time()

    # Set learning rate to the one in config for this epoch
    for g in optimizer.param_groups:
        g['lr'] = config['learning_rates'][epoch]
    lr = optimizer.param_groups[0]['lr']

    epoch_prefix = f"[Epoch {epoch + 1:2d} / {config['epochs']:2d}]"
    print(f"{epoch_prefix} Starting epoch {epoch + 1:2d} with LR = {lr}")

    # Put model in training mode
    model.train()

    # Accumulator variables
    tr_loss, tr_accuracy = 0, 0
    nb_tr_examples, nb_tr_steps = 0, 0
    for idx, batch in enumerate(dl_train):

        ids = batch['input_ids'].to(config['device'], dtype=torch.long)
        mask = batch['attention_mask'].to(config['device'], dtype=torch.long)
        labels = batch['labels'].to(config['device'], dtype=torch.long)

        loss, tr_logits = model(input_ids=ids, attention_mask=mask, labels=labels,
                                return_dict=False)
        tr_loss += loss.item()

        nb_tr_steps += 1
        nb_tr_examples += labels.size(0)

        if idx % 200 == 0:
            loss_step = tr_loss / nb_tr_steps
            print(f"{epoch_prefix}     Steps: {idx:4d} --> Loss: {loss_step:.4f}")

        # compute training accuracy
        flattened_targets = labels.view(-1)  # shape (batch_size * seq_len,)
        active_logits = tr_logits.view(-1, model.config.num_labels)  # shape (batch_size * seq_len, num_labels)
        flattened_predictions = torch.argmax(active_logits, axis=1)  # shape (batch_size * seq_len,)

        # only compute accuracy at active labels
        active_accuracy = labels.view(-1) != -100  # shape (batch_size, seq_len)

        labels = torch.masked_select(flattened_targets, active_accuracy)
        predictions = torch.masked_select(flattened_predictions, active_accuracy)

        tmp_tr_accuracy = accuracy_score(labels.cpu().numpy(), predictions.cpu().numpy())
        tr_accuracy += tmp_tr_accuracy

        # gradient clipping
        torch.nn.utils.clip_grad_norm_(
            parameters=model.parameters(), max_norm=config['max_grad_norm']
        )

        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    epoch_loss = tr_loss / nb_tr_steps
    tr_accuracy = tr_accuracy / nb_tr_steps

    torch.save(model.state_dict(), f'pytorch_model_e{epoch}.bin')
    torch.cuda.empty_cache()
    gc.collect()

    elapsed = time.time() - time_start

    print(epoch_prefix)
    print(f"{epoch_prefix} Training loss    : {epoch_loss:.4f}")
    print(f"{epoch_prefix} Training accuracy: {tr_accuracy:.4f}")
    print(f"{epoch_prefix} Model saved to pytorch_model_e{epoch}.bin  [{elapsed / 60:.2f} mins]")
    print(epoch_prefix)
    del ids, mask, labels


# from Rob Mulla @robikscube
# https://www.kaggle.com/robikscube/student-writing-competition-twitch
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

def inference(dl):
    # These 2 dictionaries will hold text-level data
    # Helping in the merging process by accumulating data
    # Through all the chunks
    predictions = defaultdict(list)
    seen_words_idx = defaultdict(list)

    for batch in dl:
        ids = batch["input_ids"].to(config['device'])
        mask = batch["attention_mask"].to(config['device'])
        outputs = model(ids, attention_mask=mask, return_dict=False)
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


# https://www.kaggle.com/zzy990106/pytorch-ner-infer
# code has been modified from original
# I moved the iteration over the batches to inference because
# samples from the same text might have be split into different batches
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


def validate(model, df_all, df_val, dl_val, epoch):
    time_start = time.time()

    # Put model in eval model
    model.eval()

    # Valid targets: needed because df_val has a subset of the columns
    df_valid = df_all.loc[df_all['id'].isin(IDS[valid_idx])]

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

config_model = AutoConfig.from_pretrained(MODEL_PATH+'/config.json')
model = AutoModelForTokenClassification.from_pretrained(
                   MODEL_PATH+'/pytorch_model.bin',config=config_model)
model.to(config['device'])

# Instantiate optimizer
optimizer = torch.optim.Adam(params=model.parameters(), lr=config['learning_rates'][0])

# Loop
for epoch in range(config['epochs']):
    print()
    train(model, optimizer, dl_train, epoch)
    validate(model, df_all, df_val, dl_val, epoch)

print("Final model saved as 'pytorch_model.bin'")
torch.save(model.state_dict(), 'pytorch_model.bin')

