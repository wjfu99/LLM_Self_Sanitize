model_name="meta-llama/Llama-3.1-8B-Instruct" # 64 layers

python privacy_emb_collector.py \
    --model_name $model_name \
    --layer_number 50 52 54 56 58 \

python self_monitor.py \
    --model_name $model_name \
    --layer_number 50 52 54 56 58 \
    --hierarchical \

python self_repair.py \
    --model_name $model_name \
    --self_monitor_layer 52 \
    --hierarchical \

python -m baselines.baselines \
    --model_name $model_name \
    --self_monitor_layer 52 \
    --hierarchical \


# evaluate on mt-bench
# python self_repair.py \
#     --model_name $model_name \
#     --self_monitor_layer 52 \
#     --hierarchical \
#     --evaluate_mt_bench \
#     --mt_bench_model Qwen/Qwen2.5-72B-Instruct \
#     --output_dir ./results/qwen2.5_72b_mt_bench_results \