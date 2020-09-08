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
/usr/local/share/anaconda3/bin/ipython /home/garner1/Work/pipelines/nucleAI/py/test.covd_with_intensity_parallelOverNuclei.py $dirSample $svsSample

# parallel "[[ -f {}.morphometrics.pkl ]] || /usr/local/share/anaconda3/bin/ipython /home/garner1/Work/pipelines/nucleAI/py/test.polygon2mask.patch.morphometry.py {}" ::: ${dirSample}/*_polygon/TCGA-*.svs/*-features.csv

# if [ ! -f ${dirSample}/*.freq*.covdNN*.features.pkl ]
# then
#     echo "Generate the descriptors"
#     /usr/local/share/anaconda3/bin/ipython /home/garner1/Work/pipelines/nucleAI/py/test.mask2descriptor.py 10 50 ${dirSample}
# fi

echo "Clean up"
parallel "rm {}" ::: ${dirSample}/*_polygon/TCGA-*.svs/*-features.csv





