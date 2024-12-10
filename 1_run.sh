tasks=("PDvsSWEDD" "NCvsSWEDD")
models=("ViT" "ResNet18" "ResNet50" "C3D" "I3D" "SlowFast" "VGG" "DenseNet121" "DenseNet264")

# 循环执行任务和模型组合
for task in "${tasks[@]}"; do
    for model in "${models[@]}"; do
        python main.py --task "$task" --model_name "$model"
    done
done