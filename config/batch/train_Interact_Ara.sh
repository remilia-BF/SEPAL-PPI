#!/bin/bash
# 自动生成的批量训练脚本 (New 1280 dim pipeline)
# 数据集: Interact_Ara
# 模型: esm2_15b

set -e  # 遇到错误时停止

echo '===================='
echo '开始训练数据集: Interact_Ara'
echo '===================='

# 步骤1: 获取原始嵌入
python emb_tools/creatlmdbbyesme.py \
  -m esm2_15b \
  -f dataset/Interact_Ara/protein.fasta \
  -o emb/Interact_Ara \
  --suffix Interact_Ara \
  --commit-interval 100 --precision bf16 --no-bucket

# 步骤2: 训练基础模型 (Average Pooling)
python sepal-ppi.py --hydra --config Interact_Ara/sepal-ppi-avg-noinput --epochs 25 \
  --output-dir results/Interact_Ara/esm2_15b/sepal-ppi-avg-noinput

# 步骤3: 微调获取线性层 (Fusion Finetune)
python sepal-ppi.py --hydra --config Interact_Ara/fusion_finetune_avg_to_1280 --mode fusion_finetune \
  --output-dir results/Interact_Ara/esm2_15b/sepal-ppi-fusion_finetune_avg_to_1280

# 步骤4: 生成残基级别嵌入 (Suffix: .avg.1280)
python emb_tools/creatlmdbbyesme.py \
  -m esm2_15b \
  -f dataset/Interact_Ara/protein.fasta \
  -o emb/Interact_Ara \
  --suffix Interact_Ara.avg.1280 \
  --commit-interval 100 --precision bf16 \
  --input-layer-ckpt results/Interact_Ara/esm2_15b/sepal-ppi-fusion_finetune_avg_to_1280/input_layer.pth \
  --source-lmdb emb/Interact_Ara/esm2_15b.Interact_Ara.lmdb

# 步骤5: 训练残基级别数据 (Projector Feature Contant)
python sepal-ppi.py --hydra --config Interact_Ara/sepal-ppi-projector-feature-contant --epochs 10 \
  --output-dir results/Interact_Ara/esm2_15b/sepal-ppi-projector-feature-contant

# 步骤6: CIS 推理训练
python sepal-ppi.py --hydra --config Interact_Ara/sepal-ppi-cis --epochs 50 \
  --output-dir results/Interact_Ara/esm2_15b/sepal-ppi-cis

# 步骤7: 集成训练
python sepal-ppi.py --mode ensemble_train --ensemble-config config/sepal-ppi/Interact_Ara/esm_ensemble.yaml \
  --predict-prot-feature-folder mutifeature/Interact_Ara --output-dir results/Interact_Ara/esm2_15b/cis_residue_ensemble
