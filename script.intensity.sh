#!/usr/bin/env bash

svsSample=$1 # the full path to the sample svs file
step=$2    # the first or second step in the pipeline: values are 1 or 2

cancer_type=`echo $svsSample | cut -d '/' -f5|cut -d '_' -f2` 
samplename=`echo $svsSample | cut -d '/' -f7 | cut -d'.' -f1` 
polygons='/media/garner1/hdd2/TCGA_polygons/'$cancer_type'/'$samplename'.*.tar.gz/'$samplename'.*.tar.gz'
dirSample='/media/garner1/hdd2/TCGA_polygons/'$cancer_type'/'$samplename'.*.tar.gz'

echo ${cancer_type} ${samplename}

if [ $step == 1 ]; then
    if test -d $dirSample; then
	echo "Uncompress"
	cd $dirSample
	tar -xf *.gz -C $PWD

	echo "Generate the intensity features"
	cp $svsSample ~/local.svs
	rm -f ${dirSample}/*_polygon/${samplename}.*/*.pkl # clean the directory
	/usr/local/share/anaconda3/bin/ipython /home/garner1/Work/pipelines/nucleAI/py/test.covd_with_intensity_parallelOverPatches.py $dirSample ~/local.svs

	mkdir -p /home/garner1/Work/pipelines/nucleAI/data/features/${cancer_type}/${samplename}
	mv ${dirSample}/*_polygon/${samplename}.*/*.pkl /home/garner1/Work/pipelines/nucleAI/data/features/${cancer_type}/${samplename}

	echo "Clean up"
	rm ~/local.svs
	parallel "rm {}" ::: ${dirSample}/*_polygon/TCGA-*.svs/*-features.csv
    fi
fi

# After the first part has finished run this
if [ $step == 2 ]; then
    for s in `ls -d data/features/${cancer_type}/TCGA-*`
    do
	sample_id=`echo "${s}" | cut -d '/' -f4`
	# If covds data is not present run the script
	# [ -f data/covds/BRCA/${sample_id}/*.pkl ] || ~/anaconda3/bin/ipython py/test.mask2descriptor.py 10 50 ${s}
	[ -f data/covds/${cancer_type}/${sample_id}/*.pkl ] || /usr/local/share/anaconda3/bin/ipython py/test.mask2descriptor.py 10 50 ${s}
    done
fi

# #/usr/local/share/anaconda3/bin/ipython py/phase1_step1.mask2descriptor_wo_intensity.py 10 50 data/features_wo_intensity/${cancer_type}/${samplename}
# # for sample in `ls data`; do /usr/local/share/anaconda3/bin/ipython py/phase1_step1.mask2descriptor_with_intensity.py 10 50 data/${sample}; done



