#!/usr/bin/env python3
"""
批量生成训练配置文件和训练命令的工具

该脚本根据训练步骤自动为多个数据集生成YAML配置文件和对应的训练命令。
支持的数据集: Strings_plant50, Interact_Ara

完整训练流程：
0. 步骤0: 生成初始嵌入 (creatlmdb.py 不带 input-layer-ckpt)
1. 步骤1: 平均池化训练 (sepal-ppi-avg，使用 avg.lmdb)
2. 步骤2: 使用训练好的input_layer生成新嵌入 (creatlmdb.py 带 input-layer-ckpt)
3. 步骤3: 直接注意力训练 (sepal-ppi-feature-contant，使用 esm2_15binput640)
4. 步骤4: 新输入平均训练 (sepal-ppi-avg-input，使用 esm2_15binput640)
5. 步骤5: 微调
6. 步骤6: CIS推理 (sepal-ppi-cis，使用 onlyCLS.lmdb)
7. 步骤7: 集成训练
8. 步骤8: 集成推理

使用方法:
    # 列出所有可用数据集
    python src/tools/batch_train_datase.py --list-datasets

    # 为指定数据集生成配置和训练脚本
    python src/tools/batch_train_datase.py --datasets Interact_Ara Strings_plant50 

    # 只生成配置文件，不生成shell脚本
    python src/tools/batch_train_datase.py --datasets Strings_plant50 --generate-only

    # 自定义训练轮数
    python src/tools/batch_train_datase.py --datasets Interact_Ara --epochs-avg 15 --epochs-feature-contant 15

    # 生成后直接运行
    python src/tools/batch_train_datase.py --datasets Interact_Ara --run
"""

import os
import argparse
from pathlib import Path
from typing import List, Dict, Optional
import yaml


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
        # 数据文件路径
        "train_file": "dataset/Strings_plant50/c1Train.txt",
        "validation_file": "dataset/Strings_plant50/c2Test.txt",
        "test_file": "dataset/Strings_plant50/c3Test.txt",
        "fasta_file": "dataset/Strings_plant50/protein.fasta",
        # 其他配置
        "cache_size": 60000,
        "max_length": 2560,
        "cis_type": False,
    },
    "Interact_Ara": {
        "protein_fasta_path": "dataset/Interact_Ara/protein.fasta",
        "all_feature_folder": "mutifeature/Interact_Ara",
        # 数据文件路径
        "train_file": "dataset/Interact_Ara/c1Train.txt",
        "validation_file": "dataset/Interact_Ara/c2Validation.txt",
        "test_file": "dataset/Interact_Ara/c3Test.txt",
        "fasta_file": "dataset/Interact_Ara/protein.fasta",
        # 其他配置
        "cache_size": 300000,
        "max_length": 1280,
        "cis_type": False,
    },
    "Human": {
        "protein_fasta_path": "dataset/Human/2193/protein.fasta",
        "all_feature_folder": "mutifeature/Human",
        # 数据文件路径
        "train_file": "dataset/Human/2193/c1Train.txt",
        "validation_file": "dataset/Human/2193/c2Validation.txt",
        "test_file": "dataset/Human/2193/c3Test.txt",
        "fasta_file": "dataset/Human/2193/protein.fasta",
        # 其他配置
        "cache_size": 15000,
        "max_length": 2193,
        "cis_type": True,
    },
    "Strings_plant90": {
        "protein_fasta_path": "dataset/Strings_plant90/protein.fasta",
        "all_feature_folder": "mutifeature/Strings_plant90",
        # 数据文件路径
        "train_file": "dataset/Strings_plant90/c1Train.txt",
        "validation_file": "dataset/Strings_plant90/c2Test.txt",
        "test_file": "dataset/Strings_plant90/c3Test.txt",
        "fasta_file": "dataset/Strings_plant90/protein.fasta",
        # 其他配置
        "cache_size": 60000,
        "max_length": 2560,
        "cis_type": False,
    },
}


# 配置模板定义 - sepal-ppi-avg 使用 avg 数据配置
CONFIG_TEMPLATES = {
    "sepal-ppi-avg": {
        "experiment_name": "sepal-ppi-avg(change640)",
        "defaults": [
            "_self_",
            {"training": "default_nolr"},
            {"model": "avg_pooling"},
            {"data": None},  # 将在生成时填充为 {dataset}/esm2_15b.avg
            {"logging": "default"},
            {"output": "default"},
            {"inference": "default"},
        ],
        "change_embedding_dim": 640,
        # 使用 avg 数据配置
        "_data_config_type": "avg",
    },
    "sepal-ppi-feature-contant": {
        "experiment_name": "sepal-ppi-feature-contant",
        "defaults": [
            "_self_",
            {"training": "default_nolr"},
            {"model": "feature-contant"},
            {"data": None},  # 将在生成时填充为 {dataset}/esm2_15binput640
            {"logging": "default"},
            {"output": "default"},
            {"inference": "default"},
        ],
        "change_embedding_dim": 640,
        "protein_fasta_path": None,
        "all_feature_folder": None,
        "_data_config_type": "input640",
    },
    "sepal-ppi-avg-input": {
        "experiment_name": "sepal-ppi-avg-input",
        "defaults": [
            "_self_",
            {"training": "default_nolr"},
            {"model": "avg_pooling"},
            {"data": None},  # 将在生成时填充为 {dataset}/esm2_15binput640
            {"logging": "default"},
            {"output": "default"},
            {"inference": "default"},
        ],
        "change_embedding_dim": 640,
        "_data_config_type": "input640",
    },
    "sepal-ppi-cis": {
        "experiment_name": "sepal-ppi-cis",
        "defaults": [
            "_self_",
            {"training": "cis_lr"},
            {"model": "cis"},
            {"data": None},  # 将在生成时填充为 {dataset}/esm2_15b.cis
            {"logging": "default"},
            {"output": "default"},
            {"inference": "default"},
        ],
        "_data_config_type": "cis",
    },
}

# 微调配置模板
FINETUNING_TEMPLATE = {
    "experiment_name": "fusion-finetune",
    "description": "Load the optimal weights from AVG and only fine tune the fusion module",
    "defaults": [
        "_self_",
        {"training": "low"},
        {"logging": "default"},
        {"output": "default"},
        {"inference": "default"},
        {"data": None},
        {"model": "feature-contant"},
    ],
    "finetune": {
        "base_checkpoint": None,
        "freeze_except": ["input_layer", "preprocessing", "pooling", "classifier"],
    },
    "change_embedding_dim": 640,
    "protein_fasta_path": None,
    "all_feature_folder": None,
}

# 集成配置模板
ENSEMBLE_TEMPLATE = {
    "configs": [],
    "mode": "logits",
    "debug": False,
    "train_model": "weighted",
}


class BatchTrainGenerator:
    """批量训练配置生成器"""

    def __init__(self, base_dir: str = ".", model: str = "esm2_15b"):
        self.base_dir = Path(base_dir)
        self.config_dir = self.base_dir / "config"
        self.sepal_ppi_dir = self.config_dir / "sepal-ppi"
        self.finetuning_dir = self.config_dir / "finetuning"
        self.ensemble_dir = self.config_dir / "ensemble"
        self.data_config_dir = self.config_dir / "data"
        self.model = model
        self.model_dim = MODEL_DIM_MAP.get(model, 5120)

    def _compute_max_length(self, fasta_rel_path: str, fallback: int) -> int:
        """根据 fasta 文件动态计算最长序列长度。

        如果文件不存在或解析失败，则回退到 fallback。
        """
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

    def generate_data_config(
        self,
        dataset: str,
        config_type: str,  # "avg" or "input640"
    ) -> Path:
        """
        生成 data yaml 配置文件
        
        config_type:
            - "avg": 使用原始 avg.lmdb，用于 sepal-ppi-avg
            - "input640": 使用 input_layer 降维后的 bucketed_embeddings.lmdb，用于后续训练
        """
        if dataset not in DATASET_CONFIGS:
            raise ValueError(f"未知数据集: {dataset}")

        dataset_info = DATASET_CONFIGS[dataset]
        
        if config_type == "avg":
            # avg 配置：使用原始嵌入的 avg.lmdb
            # 文件路径: emb/{dataset}/esm2_15b.{dataset}.lmdb/avg.lmdb
            embedding_file = f"emb/{dataset}/{self.model}.{dataset}.lmdb/avg.lmdb"
            embedding_dim = self.model_dim
            config_name = f"{self.model}.avg.yaml"
            target_precision = "fp32"
        elif config_type == "cis":
            # cis 配置：使用原始嵌入的 onlyCLS.lmdb
            # 文件路径: emb/{dataset}/esm2_15b.{dataset}.lmdb/onlyCLS.lmdb
            embedding_file = f"emb/{dataset}/{self.model}.{dataset}.lmdb/onlyCLS.lmdb"
            embedding_dim = self.model_dim
            config_name = f"{self.model}.cis.yaml"
            target_precision = "fp32"
        else:  # input640
            # input640 配置：使用 input_layer 降维后的 bucketed_embeddings.lmdb
            # 文件路径: emb/{dataset}/esm2_15b.{dataset}.avg.640.lmdb/noCLSeos.lmdb/bucketed_embeddings.lmdb
            embedding_file = f"emb/{dataset}/{self.model}.{dataset}.avg.640.lmdb/noCLSeos.lmdb/bucketed_embeddings.lmdb"
            embedding_dim = 640
            config_name = f"{self.model}input640.yaml"
            target_precision = "bf16"

        # 动态计算最长序列长度
        max_length = self._compute_max_length(dataset_info["fasta_file"], dataset_info["max_length"])

        config = {
            "use_sequence_data": True,
            "embedding_file": embedding_file,
            "embedding_dim": embedding_dim,
            "target_precision": target_precision,
            "train_file": dataset_info["train_file"],
            "validation_file": dataset_info["validation_file"],
            "test_file": dataset_info["test_file"],
            "fasta_file": dataset_info["fasta_file"],
            # 需求：所有生成的 esm2_15b*.yaml 统一使用 cache_size=18000
            "cache_size": 18000,
            "max_length": max_length,
        }
        
        # 添加 cis_type：avg/cis 都设为 True，其余沿用数据集默认
        if config_type in ("avg", "cis"):
            config["cis_type"] = True
        elif "cis_type" in dataset_info:
            config["cis_type"] = dataset_info["cis_type"]

        # 创建目录
        output_dir = self.data_config_dir / dataset
        output_dir.mkdir(parents=True, exist_ok=True)

        # 写入配置文件
        output_path = output_dir / config_name
        self._write_yaml_with_comments(output_path, config, "# 默认数据配置\n\n")
        print(f"生成数据配置: {output_path}")
        return output_path

    def generate_sepal_ppi_config(
        self,
        dataset: str,
        config_type: str,
        template: Dict,
    ) -> Path:
        """生成sepal-ppi配置文件"""
        if dataset not in DATASET_CONFIGS:
            raise ValueError(f"未知数据集: {dataset}")

        dataset_info = DATASET_CONFIGS[dataset]
        config = self._deep_copy_config(template)
        
        # 确定使用哪种数据配置
        data_config_type = config.pop("_data_config_type", "input640")
        if data_config_type == "avg":
            data_config = f"{dataset}/{self.model}.avg"
        elif data_config_type == "cis":
            data_config = f"{dataset}/{self.model}.cis"
        else:
            data_config = f"{dataset}/{self.model}input640"

        # 处理defaults列表，更新data配置
        new_defaults = []
        for item in config["defaults"]:
            if isinstance(item, dict) and "data" in item:
                new_defaults.append({"data": data_config})
            else:
                new_defaults.append(item)
        config["defaults"] = new_defaults

        # 如果配置需要protein_fasta_path和all_feature_folder
        if "protein_fasta_path" in config:
            config["protein_fasta_path"] = dataset_info["protein_fasta_path"]
        if "all_feature_folder" in config:
            config["all_feature_folder"] = dataset_info["all_feature_folder"]

        # 创建目录
        output_dir = self.sepal_ppi_dir / dataset
        output_dir.mkdir(parents=True, exist_ok=True)

        # 写入配置文件
        output_path = output_dir / f"{config_type}.yaml"
        self._write_yaml(output_path, config)
        print(f"生成配置: {output_path}")
        return output_path

    def generate_finetuning_config(self, dataset: str) -> Path:
        """生成微调配置文件"""
        if dataset not in DATASET_CONFIGS:
            raise ValueError(f"未知数据集: {dataset}")

        dataset_info = DATASET_CONFIGS[dataset]
        config = self._deep_copy_config(FINETUNING_TEMPLATE)

        # 使用 input640 数据配置
        data_config = f"{dataset}/{self.model}input640"

        # 处理defaults列表
        new_defaults = []
        for item in config["defaults"]:
            if isinstance(item, dict) and "data" in item:
                new_defaults.append({"data": data_config})
            else:
                new_defaults.append(item)
        config["defaults"] = new_defaults

        # 填充数据集相关字段
        config["protein_fasta_path"] = dataset_info["protein_fasta_path"]
        config["all_feature_folder"] = dataset_info["all_feature_folder"]
        config["finetune"]["base_checkpoint"] = (
            f"results/{dataset}/{self.model}/sepal-ppi-avg/best_model.pth"
        )

        # 创建目录
        output_dir = self.finetuning_dir / dataset
        output_dir.mkdir(parents=True, exist_ok=True)

        # 写入配置文件
        output_path = output_dir / "fusion_finetune_avg2att.yaml"
        self._write_yaml(output_path, config)
        print(f"生成配置: {output_path}")
        return output_path

    def generate_ensemble_config(self, dataset: str) -> Path:
        """生成集成配置文件"""
        config = self._deep_copy_config(ENSEMBLE_TEMPLATE)
        config["configs"] = [
            f"results/{dataset}/{self.model}/sepal-ppi-feature-contant/best_config.yaml",
            f"results/{dataset}/{self.model}/sepal-ppi-cis/best_config.yaml",
        ]

        # 创建目录
        output_dir = self.ensemble_dir / dataset
        output_dir.mkdir(parents=True, exist_ok=True)

        # 写入配置文件
        output_path = output_dir / "esm_ensemble.yaml"
        self._write_yaml(output_path, config)
        print(f"生成配置: {output_path}")
        return output_path

    def _deep_copy_config(self, config: Dict) -> Dict:
        """深拷贝配置字典"""
        import copy
        return copy.deepcopy(config)

    def _write_yaml(self, path: Path, config: Dict):
        """写入YAML文件"""
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(
                config,
                f,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False,
            )

    def _write_yaml_with_comments(self, path: Path, config: Dict, header: str = ""):
        """写入YAML文件（带注释头）"""
        with open(path, "w", encoding="utf-8") as f:
            if header:
                f.write(header)
            yaml.dump(
                config,
                f,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False,
            )

    def generate_all_configs(self, dataset: str) -> Dict[str, Path]:
        """为指定数据集生成所有配置文件"""
        configs = {}

        # 首先生成 data 配置文件
        print(f"\n生成数据配置文件...")
        configs["data_avg"] = self.generate_data_config(dataset, "avg")
        configs["data_cis"] = self.generate_data_config(dataset, "cis")
        configs["data_input640"] = self.generate_data_config(dataset, "input640")

        # 生成sepal-ppi配置
        print(f"\n生成 sepal-ppi 配置文件...")
        for config_type, template in CONFIG_TEMPLATES.items():
            configs[config_type] = self.generate_sepal_ppi_config(
                dataset, config_type, self._deep_copy_config(template)
            )

        # 生成微调配置
        print(f"\n生成微调配置文件...")
        configs["finetuning"] = self.generate_finetuning_config(dataset)

        # 生成集成配置
        print(f"\n生成集成配置文件...")
        configs["ensemble"] = self.generate_ensemble_config(dataset)

        return configs

    def generate_training_commands(self, dataset: str, epochs_config: Optional[Dict] = None) -> List[str]:
        """生成训练命令列表"""
        if epochs_config is None:
            epochs_config = {
                "avg": 25,
                "feature_contant": 25,
                "avg_input": 25,
                "finetune": 25,
                "cis": 50,
            }

        commands = []
        dataset_info = DATASET_CONFIGS[dataset]
        fasta_path = dataset_info["protein_fasta_path"]

        # 步骤0: 生成初始嵌入（不带 input-layer-ckpt）
        commands.append(f"""# 步骤0: 生成初始嵌入
python emb_tools/creatlmdb.py \\
  -m {self.model} \\
  -f {fasta_path} \\
  -o emb/{dataset} \\
  --suffix {dataset} \\
  --commit-interval 50""")

        # 步骤1: 平均池化训练（使用 avg 嵌入）
        commands.append(f"""# 步骤1: 平均池化训练 (使用 avg.lmdb)
    python sepal-ppi.py --hydra --config {dataset}/sepal-ppi-avg --epochs {epochs_config['avg']} --learning-rate 0.00001 \\
      --output-dir results/{dataset}/{self.model}/sepal-ppi-avg""")

        # 步骤2: 生成新的嵌入（使用input_layer）
        commands.append(f"""# 步骤2: 使用训练好的input_layer生成新嵌入
python emb_tools/creatlmdb.py \\
  -m {self.model} \\
  -f {fasta_path} \\
  -o emb/{dataset} \\
  --suffix {dataset}.avg.640 \\
  --commit-interval 100 \\
  --input-layer-ckpt results/{dataset}/{self.model}/sepal-ppi-avg/input_layer.pth""")

        # 步骤3: 直接注意力训练（使用 input640 嵌入）
        commands.append(f"""# 步骤3: 直接注意力训练 (使用 input_layer 降维后的嵌入)
python sepal-ppi.py --hydra --config {dataset}/sepal-ppi-feature-contant --epochs {epochs_config['feature_contant']} \\
  --output-dir results/{dataset}/{self.model}/sepal-ppi-feature-contant""")

        # 步骤4: 新输入平均训练（使用 input640 嵌入）
        commands.append(f"""# 步骤4: 新输入平均训练 (使用 input_layer 降维后的嵌入)
python sepal-ppi.py --hydra --config {dataset}/sepal-ppi-avg-input --epochs {epochs_config['avg_input']} \\
  --output-dir results/{dataset}/{self.model}/sepal-ppi-avg-input""")

        # 步骤5: 微调
        commands.append(f"""# 步骤5: 微调
python sepal-ppi.py --hydra --config finetuning/{dataset}/fusion_finetune_avg2att --mode fusion_finetune --epochs {epochs_config['finetune']} \\
  --output-dir results/{dataset}/{self.model}/sepal-ppi-finetune-att""")

        # 步骤6: CIS推理 (使用 onlyCLS.lmdb)
        commands.append(f"""# 步骤6: CIS推理 (使用 onlyCLS.lmdb)
python sepal-ppi.py --hydra --config {dataset}/sepal-ppi-cis --epochs {epochs_config['cis']} \\
  --output-dir results/{dataset}/{self.model}/sepal-ppi-cis""")

        # 步骤7: 集成训练
        commands.append(f"""# 步骤7: 集成训练
python sepal-ppi.py --mode ensemble_train --ensemble-config config/ensemble/{dataset}/esm_ensemble.yaml \\
  --predict-prot-feature-folder mutifeature/{dataset} --output-dir results/{dataset}/{self.model}/sepal_ppi_ensemble_att_cis""")

        # 步骤8: 集成推理
        commands.append(f"""# 步骤8: 集成推理
python sepal-ppi.py --mode ensemble_predict --config config/predict/{dataset}_{self.model}.yaml \\
  --output-dir results/{dataset}/{self.model}/ensemble_predict --predict-prot-feature-folder mutifeature/{dataset}""")

        return commands

    def generate_shell_script(self, datasets: List[str], output_path: str = "train_all.sh", epochs_config: Optional[Dict] = None) -> str:
        """生成完整的训练shell脚本"""
        script_lines = [
            "#!/bin/bash",
            "# 自动生成的批量训练脚本",
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
        description="批量生成训练配置文件和训练命令",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 为指定数据集生成配置和命令
  python src/tools/batch_train_datase.py --datasets Strings_plant50 Interact_Ara

  # 只生成配置文件，不生成脚本
  python src/tools/batch_train_datase.py --datasets Strings_plant50 --generate-only

  # 生成脚本并直接运行
  python src/tools/batch_train_datase.py --datasets Interact_Ara --run

  # 列出所有可用数据集
  python src/tools/batch_train_datase.py --list-datasets
        """,
    )

    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["Strings_plant50", "Interact_Ara"],
        help="要处理的数据集列表",
    )
    parser.add_argument(
        "--model",
        default="esm2_15b",
        help="使用的模型名称 (默认: esm2_15b)",
    )
    parser.add_argument(
        "--generate-only",
        action="store_true",
        help="只生成配置文件，不生成shell脚本",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="生成后立即运行训练脚本",
    )
    parser.add_argument(
        "--output-script",
        default="train_all.sh",
        help="输出的训练脚本名称",
    )
    parser.add_argument(
        "--list-datasets",
        action="store_true",
        help="列出所有可用的数据集",
    )
    parser.add_argument(
        "--base-dir",
        default=".",
        help="项目根目录路径",
    )
    parser.add_argument(
        "--epochs-avg",
        type=int,
        default=25,
        help="平均池化训练轮数",
    )
    parser.add_argument(
        "--epochs-feature-contant",
        type=int,
        default=25,
        help="特征注意力训练轮数",
    )
    parser.add_argument(
        "--epochs-avg-input",
        type=int,
        default=25,
        help="新输入平均训练轮数",
    )
    parser.add_argument(
        "--epochs-finetune",
        type=int,
        default=25,
        help="微调训练轮数",
    )
    parser.add_argument(
        "--epochs-cis",
        type=int,
        default=50,
        help="CIS推理训练轮数",
    )

    args = parser.parse_args()

    if args.list_datasets:
        print("可用数据集:")
        for dataset in DATASET_CONFIGS.keys():
            info = DATASET_CONFIGS[dataset]
            print(f"  - {dataset}")
            print(f"      protein_fasta_path: {info['protein_fasta_path']}")
            print(f"      all_feature_folder: {info['all_feature_folder']}")
            print(f"      train_file: {info['train_file']}")
            print(f"      validation_file: {info['validation_file']}")
            print(f"      test_file: {info['test_file']}")
        return

    generator = BatchTrainGenerator(args.base_dir, args.model)

    epochs_config = {
        "avg": args.epochs_avg,
        "feature_contant": args.epochs_feature_contant,
        "avg_input": args.epochs_avg_input,
        "finetune": args.epochs_finetune,
        "cis": args.epochs_cis,
    }

    # 生成配置文件
    print("=" * 50)
    print("开始生成配置文件...")
    print("=" * 50)

    for dataset in args.datasets:
        if dataset not in DATASET_CONFIGS:
            print(f"警告: 跳过未知数据集 '{dataset}'")
            continue

        print(f"\n{'='*50}")
        print(f"处理数据集: {dataset}")
        print(f"{'='*50}")
        generator.generate_all_configs(dataset)

    if args.generate_only:
        print("\n配置文件生成完成!")
        return

    # 生成训练命令
    print("\n" + "=" * 50)
    print("生成训练命令...")
    print("=" * 50)

    for dataset in args.datasets:
        if dataset not in DATASET_CONFIGS:
            continue
        print(f"\n数据集 {dataset} 的训练命令:")
        print("-" * 30)
        commands = generator.generate_training_commands(dataset, epochs_config)
        for cmd in commands:
            print(cmd)
            print()

    # 生成shell脚本
    script_content = generator.generate_shell_script(args.datasets, args.output_script, epochs_config)

    if args.run:
        print("\n" + "=" * 50)
        print("开始运行训练脚本...")
        print("=" * 50)
        import subprocess
        subprocess.run(["bash", args.output_script], check=True)


if __name__ == "__main__":
    main()
