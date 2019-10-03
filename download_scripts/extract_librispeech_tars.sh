for OUTPUT in $(ls /tts_data/asrdata/librispeech_tars)
do
    filename=`echo $OUTPUT | sed -e 's/.tar.gz//g'`
    cd /tts_data/asrdata/librispeech/
    cp /tts_data/asrdata/librispeech_tars/$OUTPUT /tts_data/asrdata/librispeech/
    tar xvf $OUTPUT
    rm -rf $OUTPUT
done