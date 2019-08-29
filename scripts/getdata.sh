echo "=== Acquiring datasets ==="
echo "---"

#mkdir -p resources
#cd resources
cd data

#echo "- Downloading enwik8 (Character)"
#mkdir -p enwik8
#cd enwik8
#wget --continue http://mattmahoney.net/dc/enwik8.zip
#python3 ../../prep_enwik8.py
#mkdir test
#mkdir valid
#mkdir train
#mv test* test
#mv train* train
#mv val* valid
#cd ..

#echo "- Downloading Penn Treebank (PTB)"
#mkdir -p ptb && cd ptb
#wget --quiet --continue http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
#tar -xzf simple-examples.tgz
#mkdir test
#mkdir valid
#mkdir train
#mv simple-examples/data/ptb.train.txt train/
#mv simple-examples/data/ptb.test.txt test/
#mv simple-examples/data/ptb.valid.txt valid/
#cd ..

#echo "- Downloading WikiText-2 (WT2)"
#wget --quiet --continue https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip
#unzip -q wikitext-2-v1.zip
#cd wikitext-2
#mkdir test
#mkdir valid
#mkdir train
#mv wiki.train.tokens train/
#mv wiki.valid.tokens valid/
#mv wiki.test.tokens test/
#cd ..

echo "- Downloading WikiText-103 (WT2)"
wget --continue https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip
unzip -q wikitext-103-v1.zip
cd wikitext-103
mkdir test
mkdir valid
mkdir train
mv wiki.valid.tokens valid/
mv wiki.test.tokens test/
python scripts/split_articles.py
cd ..

echo "---"
echo "Happy language modeling :)"