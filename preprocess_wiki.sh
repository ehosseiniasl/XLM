OUTPATH=data/wiki/processed/XLM_en/30k  # path where processed files will be stored

# This will create three files: $OUTPATH/{train,valid,test}.en.pth
# After that we're all set
python preprocess.py $OUTPATH/vocab $OUTPATH/train.en &
python preprocess.py $OUTPATH/vocab $OUTPATH/valid.en &
python preprocess.py $OUTPATH/vocab $OUTPATH/test.en