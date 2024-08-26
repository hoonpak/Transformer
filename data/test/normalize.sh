cat test/newstest2014-fren-src.en.sgm | perl normalize.perl en > test_enfr_en.txt
cat test/newstest2014-fren-ref.fr.sgm | perl normalize.perl en > test_enfr_fr.txt

sed 's/<[^>]*>//g; /^$/d' test_enfr_en.txt > enfr_en_test.txt
sed 's/<[^>]*>//g; /^$/d' test_enfr_fr.txt > enfr_fr_test.txt