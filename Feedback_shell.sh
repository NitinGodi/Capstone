curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python3 get-pip.py
pip install -q kaggle
mkdir -p ~/.kaggle
cp $1/kaggle.json ~/.kaggle/
cat ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
cd $2
mkdir Data
kaggle competitions download -c feedback-prize-2021 -p $2/Data
cd Data
unzip feedback-prize-2021.zip
rm feedback-prize-2021.zip sample_submission.csv
rm -R test
cd $2/Code 
mkdir pretrained_models
cd pretrained_models
curl -L "https://drive.google.com/file/d/1TXNCH2sVkEk9aWTUb4mi0PnDdNAOVLLH/view?usp=sharing" > glove.6B.100d.txt
curl -L "https://drive.google.com/file/d/176qq84KDHT2PsFoRwSk4GjMO6E0TeBPK/view?usp=sharing" > naive_bayes_model.pkl
curl -L "https://drive.google.com/file/d/198DhrmxEFTGnXuuqTuu_s25he90DdznE/view?usp=sharing" > best_lstm_model.bin
curl -L "https://drive.google.com/file/d/1kz4-IzkESbnwmeI7cjmPO0c-SXA6Aoad/view?usp=sharing" > best_roberta_model.bin
