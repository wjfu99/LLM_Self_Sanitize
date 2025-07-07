model_name="meta-llama/Llama-3.1-8B-Instruct" # 32 layers

python privacy_emb_collector.py \
    --model_name $model_name \
    --layer_number 26 27 28 29 30 \

python self_monitor.py \
    --model_name $model_name \
    --layer_number 26 27 28 29 30 \
    --hierarchical \

python self_repair.py \
    --model_name $model_name \
    --self_monitor_layer 26 \
    --hierarchical \

python -m baselines.baselines \
    --model_name $model_name \


# evaluate on mt-bench
# python self_repair.py \
#     --model_name $model_name \
#     --self_monitor_layer 52 \
#     --hierarchical \
#     --evaluate_mt_bench \
#     --mt_bench_model Qwen/Qwen2.5-72B-Instruct \
#     --output_dir ./results/qwen2.5_72b_mt_bench_results \