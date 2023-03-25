#!/usr/bin/bash
> valid.txt
> invalid.txt

rm -rf csv
mkdir csv

for i in $(find XML* | tqdm | grep .xml | grep -v ipynb)
do
    python ../museexport/musexmlex.py $i > /dev/null 2>&1 && echo $i >> valid.txt || echo $i >> invalid.txt
    
done
