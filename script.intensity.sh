#!/usr/bin/env bash

svsSample=$1 # the full path to the sample svs file
cancer_type=`echo $svsSample | cut -d '/' -f5|cut -d '_' -f1` 
samplename=`echo $svsSample | cut -d '/' -f7 | cut -d'.' -f1` 
polygons='/media/garner1/hdd2/TCGA_polygons/'$cancer_type'/'$samplename'.*.tar.gz/'$samplename'.*.tar.gz'
dirSample='/media/garner1/hdd2/TCGA_polygons/'$cancer_type'/'$samplename'.*.tar.gz'

echo ${cancer_type} $samplename #$polygons $dirSample

if test -d $dirSample; then
    echo "Uncompress"
    cd $dirSample
    tar -xf *.gz -C $PWD

    echo "Generate the intensity features"
    cp $svsSample ~/local.svs
    /usr/local/share/anaconda3/bin/ipython /home/garner1/Work/pipelines/nucleAI/py/test.covd_with_intensity_parallelOverPatches.py $dirSample ~/local.svs #$svsSample 

    mkdir -p /home/garner1/Work/pipelines/nucleAI/data/${samplename}
    mv ${dirSample}/*_polygon/${samplename}.*/*.morphometrics+intensity.pkl /home/garner1/Work/pipelines/nucleAI/data/${samplename}

    rm ~/local.svs

    echo "Clean up"
    parallel "rm {}" ::: ${dirSample}/*_polygon/TCGA-*.svs/*-features.csv
fi




