#!/bin/sh

mkdir -p charpragcap/resources/visual_genome_data
cd charpragcap/resources/visual_genome_data
wget https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip
wget https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip
unzip images.zip
unzip images2.zip
#rm images.zip
#rm images2.zip

mv VG_100K_2/* VG_100K/
rmdir VG_100K_2

mkdir ../visual_genome_JSON
cd ../visual_genome_JSON
wget http://visualgenome.org/static/data/dataset/image_data.json.zip
unzip image_data.json.zip
rm image_data.json.zip

wget http://visualgenome.org/static/data/dataset/region_descriptions.json.zip
unzip region_descriptions.json.zip
rm region_descriptions.json.zip
cd ../..
