for type in validated train test dev other invalidated
do
    num=$(find /tts_data/asrdata/common_voice_v3 -type f -name "$type.tsv" | xargs wc -l | egrep total | egrep -o "[0-9]+")
    echo "$type - $num"
done