#!/usr/bin/env bash

dirSample=$1 # the full path to the sample dir
echo ${dirSample}

cd ${dirSample}
echo "Uncompress"
tar -xf *.gz -C $PWD

echo "Generate the masks"
parallel "/usr/local/share/anaconda3/bin/ipython /home/garner1/Work/pipelines/nucleAI/py/phase1_step0.polygon2mask.patch.morphometry.py {}" ::: ${dirSample}/*_polygon/TCGA-*.svs/*-features.csv

echo "Clean up"
parallel "rm {}" ::: ${dirSample}/*_polygon/TCGA-*.svs/*-features.csv

echo "Generate the descriptors"
/usr/local/share/anaconda3/bin/ipython /home/garner1/Work/pipelines/nucleAI/py/phase1_step1.mask2descriptor_wo_intensity.py 10 50 ${dirSample} # !!! adapt to tissue !!!!






