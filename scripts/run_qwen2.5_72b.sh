python privacy_emb_collector.py \
    --model_name Qwen/Qwen2.5-72B-Instruct \
    --layer_number 64 66 68 70 72 \

python self_monitor.py \
    --model_name Qwen/Qwen2.5-72B-Instruct \
    --layer_number 64 66 68 70 72 \
    --output_dir ./results/qwen2.5_72b_self_monitor_results \
    --hierarchical \

python self_repair.py \
    --model_name Qwen/Qwen2.5-72B-Instruct \
    --self_monitor_layer 64 \
    --hierarchical \

# evaluate on mt-bench
python self_repair_mt.py \
    --model_name Qwen/Qwen2.5-72B-Instruct \
    --self_monitor_layer 64 \
    --hierarchical \
    --evaluate_mt_bench \
    --mt_bench_model Qwen/Qwen2.5-72B-Instruct \
    --output_dir ./results/qwen2.5_72b_mt_bench_results \