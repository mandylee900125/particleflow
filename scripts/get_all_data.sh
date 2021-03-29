
#!/bin/bash
set -e

rm -Rf test_tmp_delphes
mkdir test_tmp_delphes
cd test_tmp_delphes

mkdir -p experiments
mkdir -p data/pythia8_ttbar
mkdir -p data/pythia8_ttbar/raw
mkdir -p data/pythia8_ttbar/processed

cd data/pythia8_ttbar/raw/

for j in {0..10}
do
  for i in {0..49}
  do
    wget --no-check-certificate -nc https://zenodo.org/record/4452283/files/tev14_pythia8_ttbar_"$j"_"$i".pkl.bz2
  done
done

bzip2 -d *
