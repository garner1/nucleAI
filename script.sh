#!/usr/bin/env bash

dirSample=$1 # the full path to the sample dir

echo ${dirSample}
cd ${dirSample}
echo "Uncompress"
[[ -d ${dirSample}/*polygon ]] || tar -xf *.gz -C ${dirSample}  

echo "Generate the masks"
parallel "[[ -f {}.morphometrics.pkl ]] || /usr/local/share/anaconda3/bin/ipython /home/garner1/Work/pipelines/nucleAI/py/test.polygon2mask.patch.morphometry.py {}" ::: ${dirSample}/*_polygon/TCGA-*.svs/*-features.csv

echo "Generate the descriptors"
#rm -f ${dirSample}/*.features.pkl
[[ -f ${dirSample}/*.freq*.covdNN*.features.pkl ]] || /usr/local/share/anaconda3/bin/ipython /home/garner1/Work/pipelines/nucleAI/py/test.mask2descriptor.py 10 50 ${dirSample}



