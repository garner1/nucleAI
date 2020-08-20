#!/usr/bin/env bash

for s in  `ls -d /media/garner1/hdd2/TCGA_polygons/*/TCGA-*.svs.tar.gz | shuf`
do
    time bash /home/garner1/Work/pipelines/nucleAI/script_test.sh ${s}
done




