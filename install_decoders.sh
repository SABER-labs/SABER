#!/bin/sh
rm -rf decoders
scp -r vigi99@10.96.100.112:/Users/vigi99/AirtelDEV/saber/swig_decoder_setup ./
set -xe
git clone https://github.com/PaddlePaddle/DeepSpeech 
cd DeepSpeech
git checkout a76fc69
cd ..
mv DeepSpeech/decoders/swig ./decoders
rm -rf DeepSpeech
cd decoders
cp ../swig_decoder_setup/ctc_decoders.py ctc_decoders.py
cp ../swig_decoder_setup/swig_setup.py setup.py
cp ../swig_decoder_setup/scorer.cpp scorer.cpp
cp ../swig_decoder_setup/scorer.h scorer.h
cp ../swig_decoder_setup/swig_decoder_setup.sh setup.sh
cp ../swig_decoder_setup/decoder_utils.cpp decoder_utils.cpp
cp ../swig_decoder_setup/decoder_utils.h decoder_utils.h
cp ../swig_decoder_setup/ctc_beam_search_decoder.cpp ctc_beam_search_decoder.cpp
sed -i "s/name='swig_decoders'/name='ctc_decoders'/g" setup.py
sed -i "s/size_t blank_id = vocabulary\.size()/size_t blank_id = 0/g" ctc_greedy_decoder.cpp
sed -i "s/py_modules=\['swig_decoders'\]/py_modules=\['ctc_decoders', 'swig_decoders'\]/g" setup.py
chmod +x setup.sh
./setup.sh
echo 'Installing kenlm'
cd kenlm
mkdir build
cd build
cmake3 -DPYTHON_EXECUTABLE=/usr/bin/python3.6 ..
make -j 40
cd ..
cd ..