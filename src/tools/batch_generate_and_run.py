#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量生成并运行 SEPAL-PPI 训练配置

- 从 src/tools/dataset.csv 读取 (dataset, embedding) 组合
- 针对每个组合，基于 config/sepal-ppi/{dataset}/ 下的七类训练模板：
  [featurelar, featuremed, nofeature, input-feature, input-nofeature, avg, cis]
  生成“已解析”的普通 YAML 配置，保存到 src/tools/generated/{dataset}/{embedding}/{type}/config.yaml
- 自动选择合适的数据 data 组（cis 使用 {embedding}.cis，其余使用 {embedding}.yaml）并替换到最终配置
- 对 featuremed / featurelar / input-feature 自动切换 preprocessing 为 {featurelarge_*_S{dataset}}
- 可选择 --execute 直接依次启动训练；默认仅生成并输出待运行命令

注意：本脚本不修改原有 config/ 下的文件；它利用项目内 HydraConfigManager 解析 Hydra 配置，
然后在内存里替换 data 与 preprocessing，再导出为普通 YAML，供 sepal-ppi.py 的非 Hydra 模式直接使用。
"""

import argparse
import csv
import sys
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import yaml
import re

# 动态加入仓库根路径，便于导入内部模块
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.utils.hydra_config_manager import HydraConfigManager  # noqa: E402

CONFIG_DIR = ROOT / "config"
SEPAL_PPI_DIR = CONFIG_DIR / "sepal-ppi"
DATA_DIR = CONFIG_DIR / "data"
MODEL_DIR = CONFIG_DIR / "model"
PREPROC_DIR = CONFIG_DIR / "model_unit" / "preprocessing"
TOOLS_DIR = ROOT / "src" / "tools"
GENERATED_DIR = TOOLS_DIR / "generated"

# 七类训练配置名（与 config/sepal-ppi/{dataset}/ 下文件前缀一致）
TRAIN_TYPES = [
    "featurelar",
    "featuremed",
    "nofeature",
    "input-feature",
    "input-nofeature",
    "avg",
    "cis",
]

# 训练轮数建议：参考项目示例
EPOCHS_HINT = {
    "avg": 20,
    "cis": 80,
}
DEFAULT_EPOCHS = 20

# 需要按数据集切换 preprocessing 的训练类型及对应前缀
PREPROC_PREFIX = {
    "featuremed": "featurelarge_med",
    "featurelar": "featurelarge_large",
    "input-feature": "featurelarge_med",
}


def read_dataset_csv(csv_path: Path) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or len(row) < 2:
                continue
            ds = row[0].strip()
            emb = row[1].strip()
            if ds and emb:
                pairs.append((ds, emb))
    return pairs


def find_data_key(dataset: str, embedding: str, train_type: str) -> Optional[Tuple[str, Path]]:
    """根据 dataset / embedding / 训练类型 选取合适的数据 data 键和对应 YAML 路径。
    返回 (data_key, yaml_path) 或 None(若不存在合适文件)。
    规则：
      - cis: 优先 {embedding}.cis.yaml
            - input-* : 使用 {embedding}.yaml（取消 input640 / input1280 优先级）
            - 其他: {embedding}.yaml
    """
    ds_dir = DATA_DIR / dataset
    # 安全：目录不存在直接返回 None
    if not ds_dir.exists():
        return None

    def exist(name: str) -> Optional[Path]:
        p = ds_dir / f"{name}.yaml"
        return p if p.exists() else None

    # cis
    if train_type == "cis":
        key = f"{dataset}/{embedding}.cis"
        path = exist(f"{embedding}.cis")
        return (key, path) if path else None

    # input 系列
    if train_type in ("input-feature", "input-nofeature"):
        # 直接使用 {embedding}.yaml，不再尝试 input640 / input1280
        path = exist(embedding)
        if path:
            key = f"{dataset}/{embedding}"
            return key, path
        return None

    # 常规
    path = exist(embedding)
    if path:
        key = f"{dataset}/{embedding}"
        return key, path
    return None


def load_yaml(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def patch_resolved_config(resolved: Dict, dataset: str, train_type: str, data_yaml: Dict) -> Dict:
    """将已解析配置中的 data 与（必要时）preprocessing 替换为目标版本。"""
    # 替换 data（整个字典用 data_yaml 覆盖）
    resolved["data"] = data_yaml

    # 将模型 embedding_dim 与数据源保持一致（若提供）
    if "embedding_dim" in data_yaml:
        resolved.setdefault("model", {})["embedding_dim"] = data_yaml["embedding_dim"]

    # 仅对需要切换 preprocessing 的类型做替换
    if train_type in PREPROC_PREFIX:
        prefix = PREPROC_PREFIX[train_type]
        preproc_name = f"{prefix}_{dataset}"
        preproc_yaml_path = PREPROC_DIR / f"{preproc_name}.yaml"
        if not preproc_yaml_path.exists():
            raise FileNotFoundError(f"缺少预处理配置: {preproc_yaml_path}")
        preproc_yaml = load_yaml(preproc_yaml_path)

        # 写入到 model.model_unit.preprocessing
        model_block = resolved.setdefault("model", {})
        model_unit = model_block.setdefault("model_unit", {})
        model_unit["preprocessing"] = preproc_yaml

        # 同步更新到 model.model_config.preprocessing（训练创建模型依赖该字段）
        model_cfg = model_block.setdefault("model_config", {})
        model_cfg["preprocessing"] = preproc_yaml

    # 规范化各模块的维度，确保和 data.embedding_dim / model.embedding_dim 一致
    _normalize_model_dims(resolved)

    # 尝试解析我们刚刚注入的 preprocessing 中的 ${...} 插值，避免非 Hydra 路径报错
    _resolve_preprocessing_placeholders_inplace(resolved)

    return resolved


def _normalize_model_dims(resolved: Dict):
    """将 model.model_config.* 中的 embedding_dim/input_dim 归一化：
    - 所有模块的 embedding_dim = model.embedding_dim
    - input_data.input_dim = data.embedding_dim
    注意：不修改 preprocessing.feature_files.* 的 embedding_dim（这些是特征自身的维度）。
    """
    try:
        model = resolved.setdefault("model", {})
        data = resolved.get("data", {})
        mc = model.setdefault("model_config", {})

        model_dim = int(model.get("embedding_dim", data.get("embedding_dim", 1280)))
        data_dim = int(data.get("embedding_dim", model_dim))

        # input_data
        inp = mc.setdefault("input_data", {})
        inp["embedding_dim"] = model_dim
        inp["input_dim"] = data_dim

        # 这些模块的 embedding_dim 与模型维度一致
        for key in ("prepooling", "pooling", "post_pooling"):
            blk = mc.get(key)
            if isinstance(blk, dict):
                blk["embedding_dim"] = model_dim

        # 学习器输入维度由工厂内部计算（基于交互输出），这里无需强制，但可提供一个合理默认
        clf = mc.get("learning_architecture")
        if isinstance(clf, dict):
            # 仅当未提供时设置，避免覆盖显式配置
            clf.setdefault("input_dim", model_dim)
    except Exception:
        # 容错：不因归一化失败中断流程
        pass


_PLACEHOLDER_MODEL_DIM = re.compile(r"^\$\{\s*model\.embedding_dim\s*\}$")
_PLACEHOLDER_RELATIVE = re.compile(r"^\$\{\s*\.\.(?:\.(?:[a-zA-Z0-9_]+))+\s*\}$")


def _resolve_preprocessing_placeholders_inplace(resolved: Dict):
    """解析我们注入的 preprocessing 中最常见的占位符：
    - ${model.embedding_dim}
    - ${..feature_files.<name>.feature_dim}
    - ${..projection.hidden_dim}
    - ${..feature_files.<name>.embedding_dim}
    仅解析 preprocessing 块内部，其他位置保持不变。
    """
    try:
        model = resolved.get("model", {})
        mc = model.get("model_config", {})
        preproc = mc.get("preprocessing")
        if not isinstance(preproc, dict):
            return

        model_dim = int(model.get("embedding_dim", resolved.get("data", {}).get("embedding_dim", 1280)))

        # 简单访问器
        def get_in(d: Dict, path: List[str], default=None):
            cur = d
            for p in path:
                if not isinstance(cur, dict) or p not in cur:
                    return default
                cur = cur[p]
            return cur

        def set_in(d: Dict, path: List[str], value):
            cur = d
            for p in path[:-1]:
                if p not in cur or not isinstance(cur[p], dict):
                    cur[p] = {}
                cur = cur[p]
            cur[path[-1]] = value

        # 递归解析字符串占位符
        def resolve_value(node, ctx):
            if isinstance(node, str):
                s = node.strip()
                # ${model.embedding_dim}
                if _PLACEHOLDER_MODEL_DIM.match(s):
                    return model_dim
                # 相对占位符 ${..feature_files.xxx.feature_dim} / ${..projection.hidden_dim}
                if s.startswith("${..") and s.endswith("}"):
                    parts = s[3:-1].split('.')  # 去掉 ${ 和 }，且保留以 .. 开头后的路径
                    # parts 形如: ['feature_files','s1ssttoken','feature_dim']
                    #             或  ['projection','hidden_dim']
                    value = get_in(preproc, parts)
                    return value if value is not None else node
                return node
            elif isinstance(node, list):
                return [resolve_value(x, ctx) for x in node]
            elif isinstance(node, dict):
                return {k: resolve_value(v, ctx) for k, v in node.items()}
            else:
                return node

        resolved_preproc = resolve_value(preproc, preproc)
        # 回写
        mc["preprocessing"] = resolved_preproc
        model.setdefault("model_unit", {})["preprocessing"] = resolved_preproc

    except Exception:
        # 非关键错误：失败则保持原样，运行时工厂仍会覆盖关键维度
        pass


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description="批量生成并可选运行训练配置")
    parser.add_argument("--execute", action="store_true", help="生成后立即依次启动训练")
    parser.add_argument("--only", nargs="*", default=None,
                        help="仅运行指定训练类型（可多选），如: featuremed avg cis")
    parser.add_argument("--dry-run", action="store_true", help="仅打印将要执行的命令")
    args = parser.parse_args()

    chosen_types = set(args.only) if args.only else set(TRAIN_TYPES)

    csv_path = TOOLS_DIR / "dataset.csv"
    if not csv_path.exists():
        print(f"未找到 CSV: {csv_path}")
        sys.exit(1)

    pairs = read_dataset_csv(csv_path)
    if not pairs:
        print("CSV 为空或未解析到有效条目")
        sys.exit(1)

    # 准备 Hydra 管理器（用于解析基础模板）
    hydra_mgr = HydraConfigManager(str(CONFIG_DIR))

    # 生成命令与配置
    ensure_dir(GENERATED_DIR)
    run_commands: List[List[str]] = []

    # 先按数据集分组，再遍历嵌入
    from collections import defaultdict
    group: Dict[str, List[str]] = defaultdict(list)
    for ds, emb in pairs:
        group[ds].append(emb)

    for ds in sorted(group.keys()):
        embeddings = group[ds]
        for emb in embeddings:
            for t in TRAIN_TYPES:
                if t not in chosen_types:
                    continue

                # 选择 data 键与 yaml
                dk = find_data_key(ds, emb, t)
                if not dk:
                    print(f"跳过: 数据文件不存在 -> dataset={ds}, embedding={emb}, type={t}")
                    continue
                data_key, data_yaml_path = dk
                data_yaml = load_yaml(data_yaml_path)

                # 解析基础 Hydra 训练模板
                base_cfg_name = f"sepal-ppi/{ds}/sepal-ppi-{t}"
                try:
                    cfg = hydra_mgr.load_config(base_cfg_name)
                    resolved = hydra_mgr.resolve_config(cfg)
                except Exception as e:
                    print(f"解析基础配置失败: {base_cfg_name} -> {e}")
                    continue

                # 应用数据与（必要时）preprocessing 替换
                try:
                    final_cfg = patch_resolved_config(resolved, ds, t, data_yaml)
                except Exception as e:
                    print(f"替换配置失败: dataset={ds}, type={t} -> {e}")
                    continue

                # 保存最终普通 YAML 到 tools/generated
                out_dir = GENERATED_DIR / ds / emb / t
                ensure_dir(out_dir)
                out_cfg_path = out_dir / "config.yaml"
                with out_cfg_path.open("w", encoding="utf-8") as f:
                    yaml.safe_dump(final_cfg, f, allow_unicode=True, sort_keys=False)

                # 生成输出目录名称（与示例一致）
                run_name = f"sepal_ppi_training_seq-{t}"
                output_dir = ROOT / "results" / "final" / ds / emb / run_name
                ensure_dir(output_dir)

                # 训练轮数
                epochs = EPOCHS_HINT.get(t, DEFAULT_EPOCHS)

                # 组装命令（非 Hydra 模式，直接加载我们导出的 YAML）
                cmd = [
                    sys.executable,
                    str(ROOT / "sepal-ppi.py"),
                    "--config", str(out_cfg_path),
                    "--epochs", str(epochs),
                    "--output-dir", str(output_dir),
                ]
                run_commands.append(cmd)
                print(f"已生成: {out_cfg_path} -> 将运行 {epochs} epochs")

    # 生成 run_all.sh 便于手动执行
    sh_path = GENERATED_DIR / "run_all.sh"
    with sh_path.open("w", encoding="utf-8") as f:
        f.write("#!/usr/bin/env zsh\nset -euo pipefail\n\n")
        for cmd in run_commands:
            # 用空格安全拼接（对简单路径足够）
            f.write(" ".join(cmd) + "\n")
    sh_path.chmod(0o755)

    print(f"\n共生成 {len(run_commands)} 个训练任务。脚本: {sh_path}")

    if args.dry_run and not args.execute:
        print("dry-run 模式：仅生成不执行。")
        return

    if args.execute:
        # 顺序执行，前一个完成再进行下一个
        for i, cmd in enumerate(run_commands, 1):
            print(f"\n[{i}/{len(run_commands)}] 运行: {' '.join(cmd)}")
            proc = subprocess.run(cmd)
            if proc.returncode != 0:
                print(f"命令失败，已中止: {' '.join(cmd)}")
                sys.exit(proc.returncode)
        print("\n所有任务执行完成。")


if __name__ == "__main__":
    main()
