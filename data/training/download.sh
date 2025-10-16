wget https://www.statmt.org/wmt13/training-parallel-europarl-v7.tgz
wget https://www.statmt.org/wmt13/training-parallel-commoncrawl.tgz
# wget https://www.statmt.org/wmt13/training-parallel-un.tgz
wget https://www.statmt.org/wmt14/training-parallel-nc-v9.tgz
# wget https://www.statmt.org/wmt10/training-giga-fren.tar

tar -xvzf training-parallel-europarl-v7.tgz
tar -xvzf training-parallel-commoncrawl.tgz
# tar -xvzf training-parallel-un.tgz
tar -xvzf training-parallel-nc-v9.tgz
# tar -xvf training-giga-fren.tar
# gzip -d giga-fren.release2.fixed.en.gz
# gzip -d giga-fren.release2.fixed.fr.gz