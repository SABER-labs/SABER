#!/usr/bin/env bash

if [ ! -d kenlm ]; then
    git clone https://github.com/luotao1/kenlm.git
    echo -e "\n"
fi

if [ ! -d openfst-1.6.3 ]; then
    echo "Download and extract openfst ..."
    wget http://www.openfst.org/twiki/pub/FST/FstDownload/openfst-1.6.3.tar.gz
    tar -xzvf openfst-1.6.3.tar.gz
    echo -e "\n"
fi

if [ ! -d ThreadPool ]; then
    git clone https://github.com/progschj/ThreadPool.git
    echo -e "\n"
fi

if [ ! -d sentencepiece ]; then
    git clone https://github.com/google/sentencepiece.git
    # cd sentencepiece
    # mkdir build
    # cd build
    # cmake ..
    # make -j $(nproc)
    # sudo make install
    # sudo ldconfig -v
    echo -e "\n"
fi

echo "Install decoders ..."
python3.6 setup.py install --user --num_processes 4 --prefix=