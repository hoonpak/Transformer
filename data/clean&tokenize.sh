mkdir training test

./tokenizer.perl -l de -threads 8 < /home/user19/bag/4.RNN/dataset/training/commoncrawl.de-en.de > training/training_de.txt
./tokenizer.perl -l de -threads 8 < /home/user19/bag/4.RNN/dataset/training/europarl-v7.de-en.de >> training/training_de.txt
./tokenizer.perl -l de -threads 8 < /home/user19/bag/4.RNN/dataset/training/news-commentary-v9.de-en.de >> training/training_de.txt

./tokenizer.perl -l en -threads 8 < /home/user19/bag/4.RNN/dataset/training/commoncrawl.de-en.en > training/training_en.txt
./tokenizer.perl -l en -threads 8 < /home/user19/bag/4.RNN/dataset/training/europarl-v7.de-en.en >> training/training_en.txt
./tokenizer.perl -l en -threads 8 < /home/user19/bag/4.RNN/dataset/training/news-commentary-v9.de-en.en >> training/training_en.txt

./tokenizer.perl -l de -threads 5 < /home/user19/bag/4.RNN/dataset/test/newstest2014_de.txt > test/test_de.txt
./tokenizer.perl -l en -threads 5 < /home/user19/bag/4.RNN/dataset/test/newstest2014_en.txt > test/test_en.txt
