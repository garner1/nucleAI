#!/usr/bin/env bash

dirSample=$1 # the full path to the sample dir

echo ${dirSample}
cd ${dirSample}

if [ ! -d ${dirSample}/*polygon ]
then
    echo "Uncompress"
    tar -xf *.gz -C ${dirSample}
fi

echo "Generate the masks"
parallel "[[ -f {}.morphometrics.pkl ]] || ipython /home/garner1/Work/pipelines/nucleAI/py/test.polygon2mask.patch.morphometry.py {}" ::: ${dirSample}/*_polygon/TCGA-*.svs/*-features.csv

if [ ! -f ${dirSample}/*.freq*.covdNN*.features.pkl ]
then
    echo "Generate the descriptors"
    ipython py/test.mask2descriptor.py 10 50 ${dirSample}
fi

echo "Clean up"
parallel "rm {}" ::: ${dirSample}/*_polygon/TCGA-*.svs/*-features.csv





