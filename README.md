# Bi-SPGCN
Bi-level Superconductivity Prediction Graph Convolutional Network

1. 本仓库基于[DeepGATGNn](https://github.com/superlouis/GATGNN)构建；

2. 环境版本要求位于[requirement.txt](./requirements.txt)

3. 运行命令可按照如下使用：
    ```bash
    # Order
    python main.py --data_path="temp_data/order_data" --run_mode="Repeat" --model="CGCNN_demo" --batch_size="63"  --save_model="FALSE" --epochs="500" --aug="True"  --aug_times=5 --aug_stage=0.0  --reprocess="True"  --format="cif" --gc_count="10" --pt="True"

    # Disorder
    python main.py --data_path="temp_data/disorder_data" --run_mode="Repeat" --model="DEEP_GATGNN_demo" --batch_size="31"  --save_model="FALSE" --epochs="500" --aug="True"  --aug_times=2 --aug_stage=0.0  --reprocess="True"  --format="cif" --gc_count="15"
	```