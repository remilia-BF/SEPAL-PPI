#!/bin/bash
# 自动生成的批量训练脚本 (New 1280 dim pipeline)
# 数据集: Strings_plant50_int4
# 模型: esm2_15b

set -e  # 遇到错误时停止

echo '===================='
echo '开始训练数据集: Strings_plant50_int4'
echo '===================='

# 步骤1: 获取原始嵌入
python emb_tools/creatlmdbbyesme.py \
  -m esm2_15b \
  -f dataset/Strings_plant50/protein.fasta \
  -o emb/Strings_plant50_int4 \
  --suffix Strings_plant50_int4 \
  --commit-interval 100 --precision int4 --no-bucket

# 步骤2: 训练基础模型 (Average Pooling)
python sepal-ppi.py --hydra --config Strings_plant50_int4/sepal-ppi-avg-noinput --epochs 25 \
  --output-dir results/Strings_plant50_int4/esm2_15b/sepal-ppi-avg-noinput

# 步骤3: 微调获取线性层 (Fusion Finetune)
python sepal-ppi.py --hydra --config Strings_plant50_int4/fusion_finetune_avg_to_1280 --mode fusion_finetune \
  --output-dir results/Strings_plant50_int4/esm2_15b/sepal-ppi-fusion_finetune_avg_to_1280

# 步骤4: 生成残基级别嵌入 (Suffix: .avg.1280)
python emb_tools/creatlmdbbyesme.py \
  -m esm2_15b \
  -f dataset/Strings_plant50/protein.fasta \
  -o emb/Strings_plant50_int4 \
  --suffix Strings_plant50_int4.avg.1280 \
  --commit-interval 100 --precision int4 \
  --input-layer-ckpt results/Strings_plant50_int4/esm2_15b/sepal-ppi-fusion_finetune_avg_to_1280/input_layer.pth \
  --source-lmdb emb/Strings_plant50_int4/esm2_15b.Strings_plant50_int4.lmdb

# 步骤5: 训练残基级别数据 (Projector Feature Contant)
python sepal-ppi.py --hydra --config Strings_plant50_int4/sepal-ppi-projector-feature-contant --epochs 10 \
  --output-dir results/Strings_plant50_int4/esm2_15b/sepal-ppi-projector-feature-contant

rm emb/Strings_plant50_bf16/esm2_15b.Strings_plant50_bf16.lmdb/noCLSeos.lmdb
# 步骤6: CIS 推理训练
python sepal-ppi.py --hydra --config Strings_plant50_int4/sepal-ppi-cis --epochs 50 \
  --output-dir results/Strings_plant50_int4/esm2_15b/sepal-ppi-cis

# 步骤7: 集成训练
python sepal-ppi.py --mode ensemble_train --ensemble-config config/sepal-ppi/Strings_plant50_int4/esm_ensemble.yaml \
  --predict-prot-feature-folder mutifeature/Strings_plant50_int4 --output-dir results/Strings_plant50_int4/esm2_15b/cis_residue_ensemble



python emb_tools/creatlmdbbyesme.py \
  -m esm2_15b \
  -f predict/Gmax/Gly/subset.fasta \
  -o emb/Gly/int4 \
  --suffix gly_int4.avg.1280 \
  --commit-interval 100 --precision int4 \
  --input-layer-ckpt results/Strings_plant50_int4/esm2_15b/sepal-ppi-fusion_finetune_avg_to_1280/input_layer.pth 


python sepal-ppi.py --mode ensemble_predict --config config/predict/st50_cis_feature_int4/predict_Gly_hh70.yaml  \
  --output-dir results/Strings_plant50_int4/gly/hh70 \
  --predict-prot-feature-folder predict/Gmax/Gly/mutifeature

python sepal-ppi.py --mode ensemble_predict --config config/predict/st50_cis_feature_int4/predict_Gly_hh50.yaml  \
  --output-dir results/Strings_plant50_int4/gly/hh50 \
  --predict-prot-feature-folder predict/Gmax/Gly/mutifeature

python sepal-ppi.py --mode ensemble_predict --config config/predict/st50_cis_feature_int4/predict_Gly_hl.yaml  \
  --output-dir results/Strings_plant50_int4/gly/hl \
  --predict-prot-feature-folder predict/Gmax/Gly/mutifeature

python sepal-ppi.py --mode ensemble_predict --config config/predict/st50_cis_feature_int4/predict_Gly_ll.yaml  \
  --output-dir results/Strings_plant50_int4/gly/ll \
  --predict-prot-feature-folder predict/Gmax/Gly/mutifeature


