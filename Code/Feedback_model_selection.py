# Import packages
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from transformers import AutoConfig, AutoModelForTokenClassification
import json
import torch
from Feedback_util import NamedEntityRecog

# Load configuration variable
f = open('Feedback_config.json')
CONF = json.load(f)

# Create and return desired model
def get_model(model, word_vocab = None, label_vocab = None, pretrain_word_embedding = None):
    """Create and return desired model"""
    # Naive-Bayes model
    if model == 'naive-bayes':
        # Pipeline with CountVectorizer, TfidfTransformer (for feature extraction) and Multinomial Naive-Bayes model.
        nb_pipeline = Pipeline([('vect', CountVectorizer()),
                                ('tfidf', TfidfTransformer()),
                                ('clf', MultinomialNB())])
        return nb_pipeline

    # LSTM model
    elif model == 'lstm':
        lstm_model = NamedEntityRecog(word_vocab.size(), CONF['word_embed_dim'], CONF['word_hidden_dim'], label_vocab.size(),
                                 CONF['dropout'], pretrain_embed=pretrain_word_embedding)
        return lstm_model

    # Roberta model
    elif model == 'roberta':
        # Downloading Roberta model and initializing Adam optimizer
        config_model = AutoConfig.from_pretrained(CONF['MODEL_NAME'])
        config_model.num_labels = 15
        roberta_model = AutoModelForTokenClassification.from_pretrained(CONF['MODEL_NAME'],
                                                               config=config_model)
        optimizer = torch.optim.Adam(params=roberta_model.parameters(), lr=CONF['CONFIG']['learning_rates'][0])
        return roberta_model, optimizer
