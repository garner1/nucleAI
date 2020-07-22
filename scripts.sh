#!/usr/bin/env bash

for s in  `ls -d /media/garner1/hdd2/TCGA_polygons/*/TCGA-*.svs.tar.gz | grep -v LUAD`
do
    time bash /home/garner1/Work/pipelines/nucleAI/script.sh ${s}
done

# for dir in `ls -d /media/garner1/hdd2/TCGA_polygons/* | grep -v LUAD`
# do
#     echo ${dir}
#     cd ${dir}
#     for s in  TCGA-*.svs.tar.gz
#     do
# 	echo "Uncompress"
# 	time tar -xf ${s}/${s} -C ${s}  
# 	echo "Generate the masks"
# 	time parallel "/usr/local/share/anaconda3/bin/ipython /home/garner1/Work/pipelines/nucleAI/py/test.polygon2mask.patch.morphometry.py {}" ::: ${s}/*_polygon/TCGA-*.svs/*-features.csv
# 	echo "Generate the descriptors"
# 	time /usr/local/share/anaconda3/bin/ipython /home/garner1/Work/pipelines/nucleAI/py/test.mask2descriptor.py 10 50 ${s}
#     done
# done



