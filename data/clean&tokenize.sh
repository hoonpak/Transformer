mkdir training test

./tokenizer.perl -l de -threads 8 < training/commoncrawl.de-en.de > training/training_de.txt
./tokenizer.perl -l de -threads 8 < training/europarl-v7.de-en.de >> training/training_de.txt
./tokenizer.perl -l de -threads 8 < training/news-commentary-v9.de-en.de >> training/training_de.txt

./tokenizer.perl -l en -threads 8 < training/commoncrawl.de-en.en > training/training_en.txt
./tokenizer.perl -l en -threads 8 < training/europarl-v7.de-en.en >> training/training_en.txt
./tokenizer.perl -l en -threads 8 < training/news-commentary-v9.de-en.en >> training/training_en.txt

./tokenizer.perl -l de -threads 5 < test/newstest2014_de.txt > test/test_de.txt
./tokenizer.perl -l en -threads 5 < test/newstest2014_en.txt > test/test_en.txt

./tokenizer.perl -l en -threads 8 < training/commoncrawl.fr-en.en > training/training_enfr_en.txt
./tokenizer.perl -l en -threads 8 < training/europarl-v7.fr-en.en >> training/training_enfr_en.txt
./tokenizer.perl -l en -threads 8 < training/giga-fren.release2.fixed.en >> training/training_enfr_en.txt
./tokenizer.perl -l en -threads 8 < training/news-commentary-v9.fr-en.en >> training/training_enfr_en.txt
./tokenizer.perl -l en -threads 8 < training/undoc.2000.fr-en.en >> training/training_enfr_en.txt

./tokenizer.perl -l en -threads 5 < test/enfr_en_test.txt > test/test_enfr_en.txt

./tokenizer.perl -l fr -threads 8 < training/commoncrawl.fr-en.fr > training/training_enfr_fr.txt
./tokenizer.perl -l fr -threads 8 < training/europarl-v7.fr-en.fr >> training/training_enfr_fr.txt
./tokenizer.perl -l fr -threads 8 < training/giga-fren.release2.fixed.fr >> training/training_enfr_fr.txt
./tokenizer.perl -l fr -threads 8 < training/news-commentary-v9.fr-en.fr >> training/training_enfr_fr.txt
./tokenizer.perl -l fr -threads 8 < training/undoc.2000.fr-en.fr >> training/training_enfr_fr.txt

./tokenizer.perl -l fr -threads 5 < test/enfr_fr_test.txt > test/test_enfr_fr.txt
