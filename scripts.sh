#!/usr/bin/env bash

step=1 # if doing first or second step in pipeline
for svs in `ls /media/garner1/hdd2/svs_LUSC/*/*.svs`; do bash script.intensity.sh $svs $step; done

step=2 # if doing first or second step in pipeline
for svs in `ls /media/garner1/hdd2/svs_LUSC/*/*.svs`; do bash script.intensity.sh $svs $step; done

#######################################
# To project in 2d the covds optained so far
#/usr/local/share/anaconda3/bin/ipython3 test.covd_2dProjection.py ../data/covds/LUAD





