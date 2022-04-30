# Capstone : Feedback Tool - Identifying Argumentative Essay Elements


## Downloading Data and essential files
### Step 1: Download your Kaggle API
Log in to Kaggle and access your account. 

Scroll down to the API section. 

Click on ‘Create New API Token’ and download the kaggle.json file which contains your API token.

### Step 2: Run project-shell.sh file

Open your terminal.

Type the following command:

bash Feedback_shell.sh <kaggle.json_location> <project_folder_location>

Replace:

<kaggle.json_location> with the path of the kaggle.json file download in the previous step.

<project_folder_location> with the path of this project downloaded in your system.

This will download the data, Glove file for word embeddings, and pretrained models of Naive-Bayes, LSTM, and ROBERTA


## Training Model
Pretrained models are already available. If you still need to train the models then follows these steps.

### Step 1: Configuration
The three available models for taining are Naive-Bayes, LSTM and ROBERTA.

In Feedback_config.json file, change the value of "MODEL" to one of the following:
  - "naive-bayes" : Naive - Bayes model
  - "lstm" : LSTM model
  - "roberta" : ROBERTA model

### Step 2: Run Feedback_train.py file

##### Note: GPU with memory of at least 16 GB is required for training the models. Change the batch sizes in the 'Feedback-config.json' file while using smaller GPUs. 

## Application
### Step 1: Run Feedback_app.py file

### Step 2: Click on the ip address in the console or copy the ip address and paste it in url bar of the web browser and press enter.

