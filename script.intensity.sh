#!/usr/bin/env bash

svsSample=$1 # the full path to the sample svs file
cancer_type=`echo $svsSample | cut -d '/' -f5` 
samplename=`echo $svsSample | cut -d '/' -f7` 
polygons='/media/garner1/hdd2/TCGA_polygons/'$cancer_type'/'$samplename'.tar.gz/'$samplename'.tar.gz'
dirSample='/media/garner1/hdd2/TCGA_polygons/'$cancer_type'/'$samplename'.tar.gz'
echo ${cancer_type} $samplename
cd $dirSample


echo "Uncompress"
tar -xf *.gz -C $dirSample

echo "Generate the intensity features"
cp $svsSample ~/local.svs
/usr/local/share/anaconda3/bin/ipython /home/garner1/Work/pipelines/nucleAI/py/test.covd_with_intensity_parallelOverNuclei.py $dirSample ~/local.svs
rm ~/local.svs

echo "Clean up"
parallel "rm {}" ::: ${dirSample}/*_polygon/TCGA-*.svs/*-features.csv





