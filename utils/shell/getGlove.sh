#!/bin/sh

mkdir -p resources/wordEmbeddings
cd resources/wordEmbeddings
wget http://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip
rm glove.6B.zip

cd ../..
