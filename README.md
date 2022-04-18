# Capstone


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

### Step 3: Download glove.6B.100d.txt from internet and save it in Code folder.

### Note: GPU with memory of at least 8 GB is required to run the python files. 


## Training Model
### Step 1: Configuration
The three available models for taining are Naive-Bayes, LSTM and ROBERTA.

In Feedback_config.json file, change the value of "MODEL" to oe of the following:
  - "naive-bayes" : Naive - Bayes model
  - "lstm" : LSTM model
  - "roberta" : ROBERTA model

### Step 2: Run Feedback_train.py file


## Application
### Step 1: Run Feedback_app.py file

### Step 2: Click on the ip address in the console or copy the ip address and paste it in url bar of the web browser and press enter.

