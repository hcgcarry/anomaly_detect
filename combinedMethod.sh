python main.py  action=train dataset=SWAT
python main.py action=test dataset=SWAT
python main.py action=train dataPreprocessing=correlation dataset=SWAT
python main.py action=test dataPreprocessing=correlation dataset=SWAT