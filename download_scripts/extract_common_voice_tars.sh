for OUTPUT in $(ls /tts_data/asrdata/common_voice_v3_tars)
do
    filename=`echo $OUTPUT | sed -e 's/.tar.gz//g'`
    mkdir /tts_data/asrdata/common_voice_v3/$filename
    cd /tts_data/asrdata/common_voice_v3/$filename/
    cp /tts_data/asrdata/common_voice_v3_tars/$OUTPUT /tts_data/asrdata/common_voice_v3/$filename/
    tar xvf $OUTPUT
    rm -rf $OUTPUT
done