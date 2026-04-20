#!/usr/bin/env python3
"""
批量生成训练配置文件和训练命令的工具 (New Version)

该脚本根据新的训练步骤 (1280 dim) 自动为多个数据集生成 YAML 配置文件和对应的训练命令。
支持的数据集: Strings_plant50, Interact_Ara

使用方法:
    python src/tools/batch_train_datase_new.py --datasets Interact_Ara --run
"""

import os
import argparse
from pathlib import Path
from typing import List, Dict, Optional
import yaml
import sys

# 模型嵌入维度映射
MODEL_DIM_MAP = {
    "esm2_15b": 5120,
    "esm2_3b": 2560,
    "esm2_650m": 1280,
    "esm2_150m": 640,
    "esm1b_650m": 1280,
    "esmc_300m": 960,
    "esmc_600m": 1152,
}

# 数据集配置信息
DATASET_CONFIGS = {
    "Strings_plant50": {
        "protein_fasta_path": "dataset/Strings_plant50/protein.fasta",
        "all_feature_folder": "mutifeature/Strings_plant50",
        "train_file": "dataset/Strings_plant50/c1Train.txt",
        "validation_file": "dataset/Strings_plant50/c2Test.txt",
        "test_file": "dataset/Strings_plant50/c3Test.txt",
        "fasta_file": "dataset/Strings_plant50/protein.fasta",
        "cache_size": 12000,
        "cis_type": False,
    },
    "Interact_Ara": {
        "protein_fasta_path": "dataset/Interact_Ara/protein.fasta",
        "all_feature_folder": "mutifeature/Interact_Ara",
        "train_file": "dataset/Interact_Ara/c1Train.txt",
        "validation_file": "dataset/Interact_Ara/c2Validation.txt",
        "test_file": "dataset/Interact_Ara/c3Test.txt",
        "fasta_file": "dataset/Interact_Ara/protein.fasta",
        "cache_size": 16500,
        "cis_type": False,
    },
    "Human": {
        "protein_fasta_path": "dataset/Human/2193/protein.fasta",
        "all_feature_folder": "mutifeature/Human",
        "train_file": "dataset/Human/2193/c1Train.txt",
        "validation_file": "dataset/Human/2193/c2Validation.txt",
        "test_file": "dataset/Human/2193/c3Test.txt",
        "fasta_file": "dataset/Human/2193/protein.fasta",
        "cache_size": 12000,
        "cis_type": False,
    },
    "Strings_plant90": {
        "protein_fasta_path": "dataset/Strings_plant90/protein.fasta",
        "all_feature_folder": "mutifeature/Strings_plant90",
        "train_file": "dataset/Strings_plant90/c1Train.txt",
        "validation_file": "dataset/Strings_plant90/c2Test.txt",
        "test_file": "dataset/Strings_plant90/c3Test.txt",
        "fasta_file": "dataset/Strings_plant90/protein.fasta",
        "cache_size": 12000,
        "cis_type": False,
    },
}

# 配置模板定义
CONFIG_TEMPLATES = {
    "sepal-ppi-avg-noinput": {
        "experiment_name": "sepal-ppi-avg(change1280)",
        "defaults": [
            "_self_",
            {"training": "default_nolr"},
            {"model": "avg_pooling"},
            {"data": None},  # {dataset}/esm2_15b.avg
            {"logging": "default"},
            {"output": "default"},
            {"inference": "default"},
        ],
        "change_embedding_dim": 1280,
        "_data_config_type": "avg",
    },
    "sepal-ppi-projector-feature-contant": {
        "experiment_name": "sepal-ppi-feature-contant",
        "defaults": [
            "_self_",
            {"training": "default_nolr"},
            {"model": "projector-feature-contant"},
            {"data": None},  # {dataset}/esm2_15binput1280
            {"logging": "default"},
            {"output": "default"},
            {"inference": "default"},
        ],
        "change_embedding_dim": 1280,
        "protein_fasta_path": None,
        "all_feature_folder": None,
        "_data_config_type": "input1280",
    },
    "sepal-ppi-cis": {
        "experiment_name": "sepal-ppi-cis",
        "defaults": [
            "_self_",
            {"training": "default_nolr"},
            {"model": "cis"},
            {"data": None},  # {dataset}/esm2_15b.cis
            {"logging": "default"},
            {"output": "default"},
            {"inference": "default"},
        ],
        "_data_config_type": "cis",
    },
}

# 微调配置模板: fusion_finetune_avg_to_1280.yaml
FINETUNING_TEMPLATE = {
    "experiment_name": "fusion-finetune",
    "description": "fusion_finetune_avg_to_1280",
    "defaults": [
        "_self_",
        {"training": "low"},
        {"logging": "default"},
        {"output": "default"},
        {"inference": "default"},
        {"data": None}, # {dataset}/esm2_15b.avg
        {"model": "avg_pooling-change"},
    ],
    "finetune": {
        "base_checkpoint": None, # results/{dataset}/esm2_15b/sepal-ppi-avg-noinput/best_model.pth
        "use_pretrained_embedding_dim": False,
        "freeze_except": ["input_layer", "classifier"],
    },
    "change_embedding_dim": 1280,
    "protein_fasta_path": None,
    "all_feature_folder": None,
    "_data_config_type": "avg",
}

# 集成配置模板: esm_ensemble.yaml
ENSEMBLE_TEMPLATE = {
    "configs": [],
    "mode": "logits",
    "debug": False,
    "train_model": "weighted",
}

class BatchTrainGenerator:
    def __init__(self, base_dir: str = ".", model: str = "esm2_15b"):
        self.base_dir = Path(base_dir)
        self.config_dir = self.base_dir / "config"
        self.sepal_ppi_dir = self.config_dir / "sepal-ppi"
        self.data_config_dir = self.config_dir / "data"
        self.model = model
        self.model_dim = MODEL_DIM_MAP.get(model, 5120)

    def _compute_max_length(self, fasta_rel_path: str, fallback: int) -> int:
        fasta_path = self.base_dir / fasta_rel_path
        if not fasta_path.is_file():
            return fallback

        max_len = 0
        cur_len = 0
        try:
            with open(fasta_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    if line.startswith(">"):
                        if cur_len > max_len:
                            max_len = cur_len
                        cur_len = 0
                    else:
                        cur_len += len(line)
            if cur_len > max_len:
                max_len = cur_len
        except Exception:
            return fallback

        return max_len or fallback

    def check_dataset_files(self, dataset: str) -> bool:
        """检查数据集必须文件是否存在"""
        if dataset not in DATASET_CONFIGS:
            print(f"Error: Dataset {dataset} missing in config.")
            return False
            
        dataset_info = DATASET_CONFIGS[dataset]
        required_files = [
            dataset_info["protein_fasta_path"],
            dataset_info["train_file"],
            dataset_info["validation_file"],
            dataset_info["test_file"],
            dataset_info["all_feature_folder"]
        ]
        
        missing = []
        for f in required_files:
            if not os.path.exists(os.path.join(self.base_dir, f)):
                missing.append(f)
        
        if missing:
            print(f"Error: Missing files for dataset {dataset}:")
            for m in missing:
                print(f"  - {m}")
            return False
        return True

    def generate_data_config(self, dataset: str, config_type: str) -> Path:
        """
        生成 data yaml 配置文件
        config_type: "avg", "cis", "input1280"
        """
        if dataset not in DATASET_CONFIGS:
            raise ValueError(f"未知数据集: {dataset}")

        dataset_info = DATASET_CONFIGS[dataset]
        
        if config_type == "avg":
            embedding_file = f"emb/{dataset}/{self.model}.{dataset}.lmdb/avg.lmdb"
            embedding_dim = self.model_dim
            config_name = f"{self.model}.avg.yaml"
            target_precision = "fp32"
        elif config_type == "cis":
            embedding_file = f"emb/{dataset}/{self.model}.{dataset}.lmdb/onlyCLS.lmdb"
            embedding_dim = self.model_dim
            config_name = f"{self.model}.cis.yaml"
            target_precision = "fp32"
        else: # input1280
            embedding_file = f"emb/{dataset}/{self.model}.{dataset}.avg.1280.lmdb/noCLSeos.lmdb/bucketed_embeddings.lmdb"
            embedding_dim = 1280
            config_name = f"{self.model}input1280.yaml"
            target_precision = "bf16"

        max_length = self._compute_max_length(dataset_info["fasta_file"], 1024)

        config = {
            "use_sequence_data": True,
            "embedding_file": embedding_file,
            "embedding_dim": embedding_dim,
            "target_precision": target_precision,
            "train_file": dataset_info["train_file"],
            "validation_file": dataset_info["validation_file"],
            "test_file": dataset_info["test_file"],
            "fasta_file": dataset_info["fasta_file"],
            "cache_size": dataset_info.get("cache_size", 12000),
            "max_length": max_length,
        }
        
        if config_type in ("avg", "cis"):
            config["cis_type"] = True
        elif "cis_type" in dataset_info:
            config["cis_type"] = dataset_info["cis_type"]

        output_dir = self.data_config_dir / dataset
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / config_name
        self._write_yaml_with_comments(output_path, config, "# 默认数据配置\n\n")
        print(f"生成数据配置: {output_path}")
        return output_path

    def generate_sepal_ppi_config(self, dataset: str, config_type: str, template: Dict) -> Path:
        """生成 sepal-ppi 配置文件"""
        if dataset not in DATASET_CONFIGS:
            raise ValueError(f"未知数据集: {dataset}")

        dataset_info = DATASET_CONFIGS[dataset]
        config = self._deep_copy_config(template)
        
        data_config_type = config.pop("_data_config_type", "input1280")
        if data_config_type == "avg":
            data_config = f"{dataset}/{self.model}.avg"
        elif data_config_type == "cis":
            data_config = f"{dataset}/{self.model}.cis"
        else:
            data_config = f"{dataset}/{self.model}input1280"

        new_defaults = []
        for item in config["defaults"]:
            if isinstance(item, dict) and "data" in item:
                new_defaults.append({"data": data_config})
            else:
                new_defaults.append(item)
        config["defaults"] = new_defaults

        if "protein_fasta_path" in config:
            config["protein_fasta_path"] = dataset_info["protein_fasta_path"]
        if "all_feature_folder" in config:
            config["all_feature_folder"] = dataset_info["all_feature_folder"]

        output_dir = self.sepal_ppi_dir / dataset
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / f"{config_type}.yaml"
        self._write_yaml(output_path, config)
        print(f"生成配置: {output_path}")
        return output_path
    
    def generate_finetuning_config(self, dataset: str) -> Path:
        """生成 fusion_finetune_avg_to_1280 配置文件"""
        if dataset not in DATASET_CONFIGS:
            raise ValueError(f"未知数据集: {dataset}")
        
        dataset_info = DATASET_CONFIGS[dataset]
        config = self._deep_copy_config(FINETUNING_TEMPLATE)
        config.pop("_data_config_type", None)

        data_config = f"{dataset}/{self.model}.avg"
        
        new_defaults = []
        for item in config["defaults"]:
            if isinstance(item, dict) and "data" in item:
                new_defaults.append({"data": data_config})
            else:
                new_defaults.append(item)
        config["defaults"] = new_defaults
        
        config["protein_fasta_path"] = dataset_info["protein_fasta_path"]
        config["all_feature_folder"] = dataset_info["all_feature_folder"]
        config["finetune"]["base_checkpoint"] = f"results/{dataset}/{self.model}/sepal-ppi-avg-noinput/best_model.pth"

        output_dir = self.sepal_ppi_dir / dataset
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / "fusion_finetune_avg_to_1280.yaml"
        self._write_yaml(output_path, config)
        print(f"生成配置: {output_path}")
        return output_path

    def generate_ensemble_config(self, dataset: str) -> Path:
        """生成 esm_ensemble.yaml"""
        config = self._deep_copy_config(ENSEMBLE_TEMPLATE)
        config["configs"] = [
            f"results/{dataset}/{self.model}/sepal-ppi-projector-feature-contant/best_config.yaml",
            f"results/{dataset}/{self.model}/sepal-ppi-cis/best_config.yaml",
        ]

        output_dir = self.sepal_ppi_dir / dataset
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / "esm_ensemble.yaml"
        self._write_yaml(output_path, config)
        print(f"生成配置: {output_path}")
        return output_path

    def _deep_copy_config(self, config: Dict) -> Dict:
        import copy
        return copy.deepcopy(config)

    def _write_yaml(self, path: Path, config: Dict):
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    def _write_yaml_with_comments(self, path: Path, config: Dict, header: str = ""):
        with open(path, "w", encoding="utf-8") as f:
            if header:
                f.write(header)
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    def generate_all_configs(self, dataset: str) -> Dict[str, Path]:
        configs = {}
        print(f"\n生成数据配置文件...")
        configs["data_avg"] = self.generate_data_config(dataset, "avg")
        configs["data_cis"] = self.generate_data_config(dataset, "cis")
        configs["data_input1280"] = self.generate_data_config(dataset, "input1280")

        print(f"\n生成 sepal-ppi 配置文件...")
        for config_type, template in CONFIG_TEMPLATES.items():
            configs[config_type] = self.generate_sepal_ppi_config(
                dataset, config_type, self._deep_copy_config(template)
            )

        print(f"\n生成微调配置文件...")
        configs["finetuning"] = self.generate_finetuning_config(dataset)

        print(f"\n生成集成配置文件...")
        configs["ensemble"] = self.generate_ensemble_config(dataset)
        return configs

    def generate_training_commands(self, dataset: str, epochs_config: Optional[Dict] = None) -> List[str]:
        if epochs_config is None:
            epochs_config = {
                "avg": 25,
                "projector": 25,
                "cis": 50,
                "finetune": 25 # Assuming some default if needed
            }

        commands = []
        dataset_info = DATASET_CONFIGS[dataset]
        fasta_path = dataset_info["protein_fasta_path"]

        # #获取原始嵌入
        commands.append(f"""# 步骤1: 获取原始嵌入
python emb_tools/creatlmdb.py \\
  -m {self.model} \\
  -f {fasta_path} \\
  -o emb/{dataset} \\
  --suffix {dataset} \\
  --commit-interval 100""")

        # #利用预平均池化的数据，训练一个基础模型
        commands.append(f"""# 步骤2: 训练基础模型 (Average Pooling)
python sepal-ppi.py --hydra --config {dataset}/sepal-ppi-avg-noinput --epochs {epochs_config['avg']} \\
  --output-dir results/{dataset}/{self.model}/sepal-ppi-avg-noinput""")

        # #利用基础模型获取线性层压缩esm 嵌入的emb_dim (微调)
        commands.append(f"""# 步骤3: 微调获取线性层 (Fusion Finetune)
python sepal-ppi.py --hydra --config {dataset}/fusion_finetune_avg_to_1280 --mode fusion_finetune \\
  --output-dir results/{dataset}/{self.model}/sepal-ppi-fusion_finetune_avg_to_1280""")

        # #利用这个线性层生成残基级别嵌入
        commands.append(f"""# 步骤4: 生成残基级别嵌入 (Suffix: .avg.1280)
python emb_tools/creatlmdb.py \\
  -m {self.model} \\
  -f {fasta_path} \\
  -o emb/{dataset} \\
  --suffix {dataset}.avg.1280 \\
  --commit-interval 100 \\
  --input-layer-ckpt results/{dataset}/{self.model}/sepal-ppi-fusion_finetune_avg_to_1280/input_layer.pth \\
  --source-lmdb emb/{dataset}/{self.model}.{dataset}.lmdb""")

        # #利用这个线性层生成的嵌入训练残基级别数据和注意力池化层
        commands.append(f"""# 步骤5: 训练残基级别数据 (Projector Feature Contant)
python sepal-ppi.py --hydra --config {dataset}/sepal-ppi-projector-feature-contant --epochs {epochs_config['projector']} \\
  --output-dir results/{dataset}/{self.model}/sepal-ppi-projector-feature-contant""")

        # #CIS 训练
        commands.append(f"""# 步骤6: CIS 推理训练
python sepal-ppi.py --hydra --config {dataset}/sepal-ppi-cis --epochs {epochs_config['cis']} \\
  --output-dir results/{dataset}/{self.model}/sepal-ppi-cis""")

        # #集成训练
        commands.append(f"""# 步骤7: 集成训练
python sepal-ppi.py --mode ensemble_train --ensemble-config config/sepal-ppi/{dataset}/esm_ensemble.yaml \\
  --predict-prot-feature-folder mutifeature/{dataset} --output-dir results/{dataset}/{self.model}/cis_residue_ensemble""")

        return commands

    def generate_shell_script(self, datasets: List[str], output_path: str = "train_all_new.sh", epochs_config: Optional[Dict] = None) -> str:
        script_lines = [
            "#!/bin/bash",
            "# 自动生成的批量训练脚本 (New 1280 dim pipeline)",
            f"# 数据集: {', '.join(datasets)}",
            f"# 模型: {self.model}",
            "",
            "set -e  # 遇到错误时停止",
            "",
        ]

        for dataset in datasets:
            script_lines.append(f"echo '===================='")
            script_lines.append(f"echo '开始训练数据集: {dataset}'")
            script_lines.append(f"echo '===================='")
            script_lines.append("")

            commands = self.generate_training_commands(dataset, epochs_config)
            for cmd in commands:
                script_lines.append(cmd)
                script_lines.append("")

        script_content = "\n".join(script_lines)

        output_file = self.base_dir / output_path
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(script_content)
        os.chmod(output_file, 0o755)

        print(f"\n生成训练脚本: {output_file}")
        return script_content

def main():
    parser = argparse.ArgumentParser(
        description="批量生成训练配置文件和训练命令 (New Version)",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument("--datasets", nargs="+", default=["Interact_Ara"], help="要处理的数据集列表")
    parser.add_argument("--model", default="esm2_15b", help="使用的模型名称 (默认: esm2_15b)")
    parser.add_argument("--generate-only", action="store_true", help="只生成配置文件")
    parser.add_argument("--run", action="store_true", help="生成后运行")
    parser.add_argument("--output-script", default="train_all_new.sh", help="输出脚本名称")
    parser.add_argument("--base-dir", default=".", help="项目根目录")
    
    # Epoch parameters customization
    parser.add_argument("--epochs-avg", type=int, default=25)
    parser.add_argument("--epochs-projector", type=int, default=10) # Default 10 per user example
    parser.add_argument("--epochs-cis", type=int, default=50)

    args = parser.parse_args()
    
    generator = BatchTrainGenerator(args.base_dir, args.model)
    
    epochs_config = {
        "avg": args.epochs_avg,
        "projector": args.epochs_projector,
        "cis": args.epochs_cis
    }

    # 1. Dataset Check
    for dataset in args.datasets:
        if not generator.check_dataset_files(dataset):
            sys.exit(1)

    # 2. Generate Configs
    for dataset in args.datasets:
        print(f"\n=== 处理数据集: {dataset} ===")
        generator.generate_all_configs(dataset)

    if args.generate_only:
        return

    # 3. Generate Shell Script
    # Determine the output script path
    
    # Check if user provided an output_script different from the default
    user_provided_script_name = False
    for arg in sys.argv:
        if "--output-script" in arg:
            user_provided_script_name = True
            break
            
    final_output_script = args.output_script
    if not user_provided_script_name and args.output_script == "train_all_new.sh":
        # 如果用户没有指定 output-script，则自动根据数据集生成
        # 保存路径: config/batch/train_{dataset}.sh
        # 如果是多个数据集，将连接它们的名称
        dataset_str = "_".join(args.datasets)
        output_dir = Path("config/batch")
        output_dir.mkdir(parents=True, exist_ok=True)
        final_output_script = str(output_dir / f"train_{dataset_str}.sh")

    generator.generate_shell_script(args.datasets, final_output_script, epochs_config)

    # 4. Generate Commands Output for user visibility
    print("\n" + "=" * 50)
    print("生成训练命令预览:")
    print("=" * 50)
    for dataset in args.datasets:
        print(f"\n数据集 {dataset}:")
        for cmd in generator.generate_training_commands(dataset, epochs_config):
            print(cmd)
            print()

    if args.run:
        print(f"\n=== 开始运行训练脚本: {final_output_script} ===")
        import subprocess
        subprocess.run(["bash", final_output_script], check=True)

if __name__ == "__main__":
    main()
