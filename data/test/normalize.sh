cat newstest2013-src.de.sgm | perl normalize.perl en > test_de_2013.txt
cat newstest2013-src.en.sgm | perl normalize.perl en > test_en_2013.txt

sed 's/<[^>]*>//g; /^$/d' test_de_2013.txt > test_de_2013_.txt
sed 's/<[^>]*>//g; /^$/d' test_en_2013.txt > test_en_2013_.txt

../tokenizer.perl -l de -threads 5 < test_de_2013_.txt > test_de_2013.txt
../tokenizer.perl -l en -threads 5 < test_en_2013_.txt > test_en_2013.txt