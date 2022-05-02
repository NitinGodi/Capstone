# Import packages
import os
import torch
from sklearn.metrics import accuracy_score, f1_score
import pickle
import torch.optim as optim
import Feedback_util as util
from Feedback_model_selection import get_model
import json
from Feedback_util import lr_decay, train_model, evaluate


# Initialize and load configuration variables
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
f = open('Feedback_config.json')
CONF = json.load(f)
ID2CLASS = {v: k for k, v in CONF['CLASS2ID'].items()}
LABELS_TO_IDS = {v:k for k,v in enumerate(CONF['OUTPUT_LABELS'])}
IDS_TO_LABELS = {k:v for k,v in enumerate(CONF['OUTPUT_LABELS'])}
device = torch.device("cuda")

# Training and testing Naive-Bayes model
if CONF['MODEL'] == 'naive-bayes':
    # Preprocessing and tokenizing data
    df_train, df_val = util.preprocessing(CONF['MODEL'])

    # Initializing pipeline for Naive-Bayes model
    nb_pipeline = get_model(CONF['MODEL'])

    # Training the model on sentences in training set
    nb_model = nb_pipeline.fit(df_train["text"], df_train["label"])

    # Saving the model
    pickle.dump(nb_model, open(f'{CONF["MODEL_PATH"]}/naive_bayes_model.pkl','wb'))

    # Predicting classes for sentences in validation set
    df_val['predictions'] = nb_model.predict(df_val['text'])
    df_val['pred_class'] = df_val['predictions'].map(ID2CLASS)

    # Calculating Validation Accuracy and F1 score
    print(f'Validation accuracy= {accuracy_score(df_val["predictions"], df_val["label"])}')
    print(f'Validation F1 score= {f1_score(df_val["predictions"], df_val["label"], average="macro")}')

# Training and testing LSTM model
elif CONF['MODEL'] == 'lstm':
    # Preprocessing and tokenizing data
    pretrain_word_embedding, train_dataloader, val_dataloader, word_vocab, label_vocab = util.preprocessing(CONF['MODEL'])

    # Initializing LSTM model
    lstm_model = get_model(CONF['MODEL'], word_vocab, label_vocab, pretrain_word_embedding)

    # Training and validating the LSTM model and storing the model with the best Macro F1 score
    lstm_model = lstm_model.to(device)

    # Initializing the optimizer
    optimizer = optim.SGD(lstm_model.parameters(), lr=CONF['lr'], momentum=0.9)

    best_f1 = -1
    early_stop = 0

    for epoch in range(CONF['lstm_epochs']):
        print(f"\nEpoch {epoch+1}/{CONF['lstm_epochs']}:")
        optimizer = lr_decay(optimizer, epoch, 0.05, CONF['lr'])

        # Training the model
        train_model(train_dataloader, lstm_model, optimizer)

        # Validating the model
        new_f1 = evaluate(val_dataloader, lstm_model)

        # Saving the model with the best F1 score
        if new_f1 > best_f1:
            best_f1 = new_f1
            print('New best f1:', best_f1)
            early_stop = 0
            torch.save(lstm_model.state_dict(), f'{CONF["MODEL_PATH"]}/best_lstm_model.bin')
        else:
            early_stop += 1

        # Early stopping with patience 10
        if early_stop > CONF['patience']:
            print('early stop')
            break

# Training and testing Roberta model
elif CONF['MODEL'] == 'roberta':
    # Preprocessing and tokenizing data
    preprocessed_data = util.preprocessing(CONF['MODEL'])

    # Initializing Roberta model
    roberta_model, optimizer = get_model(CONF['MODEL'])

    # Training and validating the model
    roberta_model.to(CONF['CONFIG']['device'])

    best_f1 = -1
    for epoch in range(CONF['CONFIG']['epochs']):
        util.train(roberta_model, optimizer, preprocessed_data['dl_train'], epoch)
        f1 = util.validate(roberta_model, preprocessed_data['df'], preprocessed_data['df_val'], preprocessed_data['dl_val'], epoch, preprocessed_data['valid_idx'], preprocessed_data['unique_ids'])
        if f1 > best_f1:
            best_f1 = f1
            # Saving the model
            torch.save(roberta_model.state_dict(), f'{CONF["MODEL_PATH"]}/best_roberta_model.bin')
