OUTPATH=data/text8  # path where processed files will be stored

python preprocess_character.py $OUTPATH/vocab $OUTPATH/train.txt
python preprocess_character.py $OUTPATH/vocab $OUTPATH/valid.txt
python preprocess_character.py $OUTPATH/vocab $OUTPATH/test.txt