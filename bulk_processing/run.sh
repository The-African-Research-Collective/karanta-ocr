

for i in {3..12}
do
    python3 main.py --input /Users/odunayoogundepo/newspaper-parser/data/jw/southern_sotho/requests_dir/batch_llm_requsts_file_allenai_olmOCR_7B_0725_FP8_${i}.jsonl  \
        --output /Users/odunayoogundepo/newspaper-parser/data/jw/southern_sotho/response_dir/batch_llm_requsts_file_allenai_olmOCR_7B_0725_FP8_${i} \
        --ports 8006 8005 8007 \
        --model-name allenai/olmOCR-7B-0725-FP8
    sleep 1
done