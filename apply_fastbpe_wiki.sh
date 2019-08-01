OUTPATH=data/wiki/processed/XLM_en/30k  # path where processed files will be stored
FASTBPE=tools/fastBPE/fast  # path to the fastBPE tool

# create output path
#mkdir -p $OUTPATH

# learn bpe codes on the training set (or only use a subset of it)
#$FASTBPE learnbpe 30000 data/wiki/txt/train.en > $OUTPATH/codes

#$FASTBPE applybpe $OUTPATH/train.en data/wiki/txt/train.en $OUTPATH/codes &
#$FASTBPE applybpe $OUTPATH/valid.en data/wiki/txt/valid.en $OUTPATH/codes &
#$FASTBPE applybpe $OUTPATH/test.en data/wiki/txt/test.en $OUTPATH/codes

# and get the post-BPE vocabulary:
#cat $OUTPATH/train.en | $FASTBPE getvocab - > $OUTPATH/vocab

# This will create three files: $OUTPATH/{train,valid,test}.en.pth
# After that we're all set
python preprocess.py $OUTPATH/vocab $OUTPATH/train.en 
#python preprocess.py $OUTPATH/vocab $OUTPATH/valid.en
#python preprocess.py $OUTPATH/vocab $OUTPATH/test.en