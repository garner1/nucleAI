#!/usr/bin/env bash

for svs in `ls /media/garner1/hdd2/svs_LUAD/*/*.svs`; do bash script.intensity.sh $svs; done
#######################################
# To project in 2d the covds optained so far
#/usr/local/share/anaconda3/bin/ipython3 test.covd_2dProjection.py ../data/covds/LUAD

#############################################
# for s in  `ls -d /media/garner1/hdd2/TCGA_polygons/*/TCGA-*.svs.tar.gz | shuf`
# do
#     time bash /home/garner1/Work/pipelines/nucleAI/script_test.sh ${s}
# done




