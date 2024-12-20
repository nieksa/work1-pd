tasks=("NCvsSWEDD")
models=("Design6" "Design7")

for task in "${tasks[@]}"; do
    for model in "${models[@]}"; do
        python main.py --task "$task" --model_name "$model" --epochs 50 --lr 0.0001 --train_bs 16 --val_bs 16 --num_workers 16
    done
done


