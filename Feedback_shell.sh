#!/bin/sh
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