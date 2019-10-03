age_total=0
total_val=0
for file in $(find /tts_data/asrdata/common_voice_v3 -type f -name "validated.tsv")
do
    total=$(cat $file | wc -l)
    age_column=$(cat $file | cut -f6 | sed '/^[[:space:]]*$/d' | wc -l)
    percentage=$(bc -l <<<"scale=2; $age_column/$total * 100")
    age_total=$(expr $age_total + $age_column)
    total_val=$(expr $total_val + $total)
    echo "Number of columns of age, total and percentage for $file are $age_column, $total, $percentage"
done
echo "Total number of age validated samples are $age_total, $total_val"
