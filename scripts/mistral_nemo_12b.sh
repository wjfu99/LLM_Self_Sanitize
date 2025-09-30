model_name="mistralai/Mistral-Nemo-Instruct-2407" # 40 layers

python privacy_emb_collector.py \
    --model_name $model_name \
    --layer_number 32 33 34 35 36 \

python self_monitor.py \
    --model_name $model_name \
    --layer_number 32 33 34 35 36 \
    --hierarchical \

python self_repair.py \
    --model_name $model_name \
    --self_monitor_layer 32 \
    --hierarchical \

python -m baselines.baselines \
    --model_name $model_name \

python -m baselines.safety_ft \
    --model_name $model_name \
# evaluate on mt-bench
# python self_repair.py \
#     --model_name $model_name \
#     --self_monitor_layer 52 \
#     --hierarchical \
#     --evaluate_mt_bench \
#     --mt_bench_model Qwen/Qwen2.5-72B-Instruct \
#     --output_dir ./results/qwen2.5_72b_mt_bench_results \