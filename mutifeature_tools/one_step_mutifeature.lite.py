#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
一键式多特征工具脚本 (优化版)
只生成指定特征：HMM相关(6项), s1ssttoken, sasa, secondary_structure
如果 s1ssttoken 生成失败，自动跳过不中断。

使用方法:
python one_step_mutifeature.py -p /path/to/pdb/files -f /path/to/fasta/file -o /path/to/output
或省略 -f，自动从 PDB 生成 FASTA:
python one_step_mutifeature.py -p /path/to/pdb/files -o /path/to/output
"""

import os
import sys
import argparse
import subprocess
import time
import json
import re
from pathlib import Path
from typing import Optional, Dict, List, Any, Set
import logging

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    yaml = None
    YAML_AVAILABLE = False

try:
    from Bio.PDB import PDBParser, PPBuilder
    BIOPYTHON_AVAILABLE = True
except Exception:
    PDBParser = None
    PPBuilder = None
    BIOPYTHON_AVAILABLE = False

def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """设置日志记录"""
    logger = logging.getLogger("one_step_mutifeature")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level.upper()))
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(formatter)
    
    if not logger.handlers:
        logger.addHandler(console_handler)
    
    return logger

def check_dependencies() -> Dict[str, bool]:
    """检查依赖脚本是否存在 (已移除 pdb2pqrjson.py)"""
    script_dir = Path(__file__).parent
    dependencies = {
        'cereat_prosst.py': script_dir / 'cereat_prosst.py',
        'generate_hmm_features.py': script_dir / 'generate_hmm_features.py', 
        'extract_features_fast.py': script_dir / 'extract_features_fast.py'
    }
    
    status = {}
    for name, path in dependencies.items():
        status[name] = path.exists()
        if not status[name]:
            print(f"警告: 依赖脚本不存在: {path}")
    
    return status

def load_fasta_lengths(fasta_path: Path, logger: logging.Logger) -> Dict[str, int]:
    """解析FASTA文件，返回蛋白质长度映射"""
    lengths: Dict[str, int] = {}
    current_id: Optional[str] = None
    seq_parts: List[str] = []
    try:
        with open(fasta_path, 'r', encoding='utf-8') as fasta_file:
            for raw_line in fasta_file:
                line = raw_line.strip()
                if not line:
                    continue
                if line.startswith('>'):
                    if current_id is not None:
                        sequence = ''.join(seq_parts)
                        lengths[current_id] = len(sequence)
                    current_id = line[1:].split()[0]
                    seq_parts = []
                else:
                    seq_parts.append(line)
        if current_id is not None:
            sequence = ''.join(seq_parts)
            lengths[current_id] = len(sequence)
    except Exception as exc:
        logger.error(f"读取FASTA失败: {exc}")
        return {}
    
    logger.info(f"FASTA序列解析完成: {len(lengths)} 个条目")
    return lengths

def determine_feature_length(value: Any) -> Optional[int]:
    """推断特征字段的长度"""
    if isinstance(value, list) or isinstance(value, tuple):
        return len(value)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped == '': return 0
        if '],' in stripped:
            return len(stripped.split('],'))
        if '[' in stripped or ']' in stripped:
            try:
                parsed = json.loads(stripped) if (stripped.startswith('[') and stripped.endswith(']')) else json.loads(f"[{stripped}]")
                if isinstance(parsed, list): return len(parsed)
            except: pass
        if ',' in stripped:
            return len([t for t in stripped.split(',') if t.strip()])
        return 1
    return None

def write_failed_proteins(output_dir: Path, failures: List[Dict[str, str]], logger: logging.Logger):
    """将验证失败详情写入文件"""
    failed_path = output_dir / 'failed_proteins.txt'
    try:
        with open(failed_path, 'w', encoding='utf-8') as failed_file:
            if not failures:
                failed_file.write("All validated feature files matched FASTA lengths.\n")
            else:
                for item in failures:
                    failed_file.write(f"file: {item.get('file')}\nid: {item.get('protein_id')}\nreason: {item.get('reason')}\n----------------\n")
    except Exception:
        pass

def validate_feature_lengths(fasta_file: Path, output_dir: Path, verification_results: Dict[str, bool], logger: logging.Logger):
    """检查生成的JSON特征文件与FASTA长度是否一致"""
    protein_lengths = load_fasta_lengths(fasta_file, logger)
    if not protein_lengths: return

    failures = []
    for filename, exists in verification_results.items():
        if not exists: continue
        
        file_path = output_dir / filename
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except:
            continue
            
        if not isinstance(data, list): continue
        
        for entry in data:
            if not isinstance(entry, dict): continue
            pid = entry.get('protein_id')
            if not pid or pid not in protein_lengths: continue
            
            expected = protein_lengths[pid]
            for key, value in entry.items():
                if key == 'protein_id': continue
                actual = determine_feature_length(value)
                if actual is not None and actual != expected:
                    failures.append({'file': filename, 'protein_id': pid, 'reason': f'length mismatch {key}: exp={expected}, act={actual}'})

    if failures:
        logger.warning(f"发现 {len(failures)} 个长度不匹配的条目，详情请见 failed_proteins.txt")
        write_failed_proteins(output_dir, failures, logger)

def run_script(script_path: Path, args: List[str], logger: logging.Logger) -> bool:
    """运行Python脚本"""
    try:
        cmd = [sys.executable, str(script_path)] + args
        logger.info(f"执行: {script_path.name}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            return True
        else:
            logger.error(f"脚本报错 {script_path.name}: {result.stderr[:200]}...") # 只显示前200字符错误
            return False
    except Exception as e:
        logger.error(f"调用异常 {script_path.name}: {e}")
        return False

# 每个步骤对应的全部输出文件（模块级常量，便于其他函数引用）
STEP_FILES: Dict[str, List[str]] = {
    's1ssttoken': ['s1ssttoken.json'],
    'HMM特征': [
        'hmm_domain_flag.json',
        'hmm_confidence_score.json',
        'hmm_emission_prob.json',
        'hmm_core_site_flag.json',
        'hmm_boundary_site_flag.json',
        'hmm_binding_domain_flag.json'
    ],
    '快速特征': [
        'sasa_features.json',
        'secondary_structure_features.json'
    ]
}


def load_enabled_features(config_path: Optional[Path], logger: logging.Logger) -> Optional[Set[str]]:
    """从 feature_concat.yaml 解析 enabled: true 的特征文件名集合。
    返回 None 表示无法读取配置，应生成全部文件。"""
    if config_path is None or not config_path.exists():
        if config_path is not None:
            logger.warning(f"配置文件不存在: {config_path}，将生成所有特征")
        return None

    if not YAML_AVAILABLE:
        logger.warning("未安装 PyYAML，无法解析配置，将生成所有特征。可运行: pip install pyyaml")
        return None

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            raw = f.read()
        # 将 ${...} 占位符替换为空字符串，避免 yaml 解析报错
        raw = re.sub(r'\$\{[^}]+\}', '""', raw)
        cfg = yaml.safe_load(raw)

        feature_files_cfg = cfg.get('feature_files', {})
        if not isinstance(feature_files_cfg, dict):
            logger.warning("配置中 feature_files 格式异常，将生成所有特征")
            return None

        enabled: Set[str] = set()
        for feat_name, feat_cfg in feature_files_cfg.items():
            if not isinstance(feat_cfg, dict):
                continue
            if feat_cfg.get('enabled', True):  # 默认视为 enabled
                fp = feat_cfg.get('file_path', '')
                if fp:
                    enabled.add(fp)

        logger.info(f"从配置读取到 {len(enabled)} 个 enabled 特征文件: {sorted(enabled)}")
        return enabled
    except Exception as exc:
        logger.warning(f"解析配置文件失败: {exc}，将生成所有特征")
        return None


def get_needed_files_for_step(step_name: str, enabled_features: Optional[Set[str]]) -> List[str]:
    """获取步骤中实际需要生成的文件（根据 enabled 配置过滤）"""
    all_files = STEP_FILES.get(step_name, [])
    if enabled_features is None:
        return all_files  # 未提供配置时生成全部
    return [f for f in all_files if f in enabled_features]


def check_step_outputs(output_dir: Path, step_name: str, logger: logging.Logger,
                       enabled_features: Optional[Set[str]] = None,
                       pdb_dir: Optional[Path] = None) -> bool:
    """检查步骤中需要的输出文件是否已存在
    
    对于 s1ssttoken 步骤，如果提供了 pdb_dir，会智能比较 JSON 中的蛋白与 PDB 目录的差异，
    仅在所有蛋白都已处理时才跳过。
    """
    targets = get_needed_files_for_step(step_name, enabled_features)
    if not targets: return False
    
    missing = [f for f in targets if not (output_dir / f).exists() or (output_dir / f).stat().st_size == 0]

    if not missing:
        # 文件都存在，但对 s1ssttoken 需做增量检查
        if step_name == 's1ssttoken' and pdb_dir is not None:
            json_file = output_dir / targets[0]  # s1ssttoken.json
            try:
                import json as _json
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = _json.load(f)
                json_ids = {entry.get('protein_id') for entry in data if isinstance(entry, dict)}
                pdb_ids = {p.stem for p in pdb_dir.iterdir() if p.suffix.lower() == '.pdb'}
                new_ids = pdb_ids - json_ids
                if new_ids:
                    logger.info(f"步骤 '{step_name}': 发现 {len(new_ids)} 个新蛋白需要追加处理")
                    return False  # 不删除旧文件，交给 cereat_prosst --append 处理
                else:
                    logger.info(f"步骤 '{step_name}' 产物已存在且包含所有 {len(pdb_ids)} 个蛋白，跳过。")
                    return True
            except Exception as exc:
                logger.warning(f"检查 {step_name} 增量状态时出错: {exc}，将重新执行")
                return False
        
        logger.info(f"步骤 '{step_name}' 产物已存在，跳过生成。")
        return True

    # 如果有缺失，对非 s1ssttoken 步骤清理旧文件以防混淆
    if step_name != 's1ssttoken':
        for f in targets:
            p = output_dir / f
            if p.exists():
                try: p.unlink()
                except: pass
            
    return False

def run_s1ssttoken_extraction(pdb_dir: Path, output_dir: Path, logger: logging.Logger) -> bool:
    """运行 s1ssttoken 生成 (原 PROSST)"""
    logger.info(">>> 步骤: 生成 s1ssttoken")
    
    # 按照您的要求，输出文件名定为 s1ssttoken.json
    output_file = output_dir / "s1ssttoken.json"
    
    args = [
        '--input', str(pdb_dir),
        '--output', str(output_file),
        '--vocab_size', '2048',
        '--mode', 'multiprocess',
        '--append'
    ]
    
    script_path = Path(__file__).parent / 'cereat_prosst.py'
    
    # 特殊处理：如果失败，只返回 False，不抛出异常，以便主流程捕捉并跳过
    if not run_script(script_path, args, logger):
        logger.warning("! s1ssttoken 生成失败。将跳过此文件，继续后续步骤。")
        return False
        
    return True

def run_hmm_feature_extraction(fasta_file: Path, output_dir: Path, logger: logging.Logger) -> bool:
    """运行 HMM 特征提取"""
    logger.info(">>> 步骤: 生成 HMM 特征")
    
    repo_root = Path(__file__).resolve().parents[1]
    hmm_data_dir = repo_root / 'cache' / 'hmm' / 'domains'

    args = [
        '--fasta', str(fasta_file),
        '--output', str(output_dir),
        '--hmm_data_dir', str(hmm_data_dir),
        '--hmm_db_path', './data/weights/Pfam-A.hmm'
    ]
    
    script_path = Path(__file__).parent / 'generate_hmm_features.py'
    return run_script(script_path, args, logger)

def run_fast_feature_extraction(pdb_dir: Path, output_dir: Path, logger: logging.Logger) -> bool:
    """运行快速特征提取 (SASA, Secondary Structure)"""
    logger.info(">>> 步骤: 生成 SASA 和 二级结构特征")
    
    args = [
        '--input_dir', str(pdb_dir),
        '--output_dir', str(output_dir),
        '--probe_radius', '1.4',
        '--num_workers', '16'
    ]
    
    script_path = Path(__file__).parent / 'extract_features_fast.py'
    return run_script(script_path, args, logger)

def generate_fasta_from_pdb(pdb_dir: Path, output_dir: Path, logger: logging.Logger) -> Optional[Path]:
    """使用 Biopython 从 PDB 文件生成 FASTA（每个 PDB 文件一条序列）"""
    if not BIOPYTHON_AVAILABLE:
        logger.error("未安装 Biopython，无法自动从 PDB 生成 FASTA。请安装 biopython 或手动提供 -f")
        return None

    fasta_path = output_dir / 'pdb_generated.fasta'
    pdb_files = sorted(
        [
            p for p in pdb_dir.iterdir()
            if p.is_file() and p.suffix.lower() in {'.pdb', '.ent'}
        ]
    )

    if not pdb_files:
        logger.error(f"在 {pdb_dir} 未找到 .pdb/.ent 文件，无法自动生成 FASTA")
        return None

    parser = PDBParser(QUIET=True)
    pp_builder = PPBuilder()
    written = 0

    try:
        with open(fasta_path, 'w', encoding='utf-8') as fasta_out:
            for pdb_file in pdb_files:
                protein_id = pdb_file.stem
                try:
                    structure = parser.get_structure(protein_id, str(pdb_file))
                    models = list(structure.get_models())
                    if not models:
                        logger.warning(f"跳过 {pdb_file.name}: 无可用模型")
                        continue

                    first_model = models[0]
                    seq_parts: List[str] = []
                    for chain in first_model:
                        peptides = pp_builder.build_peptides(chain)
                        if peptides:
                            chain_seq = ''.join(str(peptide.get_sequence()) for peptide in peptides)
                            if chain_seq:
                                seq_parts.append(chain_seq)

                    sequence = ''.join(seq_parts)
                    if not sequence:
                        logger.warning(f"跳过 {pdb_file.name}: 未解析到有效氨基酸序列")
                        continue

                    fasta_out.write(f">{protein_id}\n{sequence}\n")
                    written += 1
                except Exception as exc:
                    logger.warning(f"跳过 {pdb_file.name}: 解析失败 ({exc})")

        if written == 0:
            logger.error("自动生成 FASTA 失败：没有可写入的序列")
            return None

        logger.info(f"已自动生成 FASTA: {fasta_path} (共 {written} 条序列)")
        return fasta_path
    except Exception as exc:
        logger.error(f"写入自动 FASTA 失败: {exc}")
        return None

def verify_output_files(output_dir: Path, logger: logging.Logger,
                        enabled_features: Optional[Set[str]] = None) -> Dict[str, bool]:
    """验证输出文件，若提供 enabled_features 则只检查其中的文件"""
    # 汇总所有步骤的全部候选文件
    all_step_files = [f for files in STEP_FILES.values() for f in files]
    if enabled_features is not None:
        expected_files = [f for f in all_step_files if f in enabled_features]
    else:
        expected_files = all_step_files
    expected_files = sorted(set(expected_files))
    
    results = {}
    logger.info("-" * 30)
    logger.info("结果验证")
    
    for filename in expected_files:
        path = output_dir / filename
        exists = path.exists() and path.stat().st_size > 0
        results[filename] = exists
        mark = "✓" if exists else "✗"
        logger.info(f"{mark} {filename}")
    
    return results

def main():
    # 默认配置文件路径（相对于本脚本所在仓库根目录）
    _default_cfg = Path(__file__).resolve().parents[1] / 'config' / 'model_unit' / 'preprocessing' / 'feature_concat.yaml'

    parser = argparse.ArgumentParser(description="指定特征生成工具")
    parser.add_argument('-p', '--pdb_dir', type=str, required=True, help='PDB目录')
    parser.add_argument('-f', '--fasta_file', type=str, default=None, help='FASTA文件（可选，不提供时自动从PDB生成）')
    parser.add_argument('-o', '--output_dir', type=str, default=None, help='输出目录')
    parser.add_argument('--config', type=str, default=str(_default_cfg),
                        help='feature_concat.yaml 路径，用于读取 enabled 特征列表（默认自动检测）')
    parser.add_argument('--log_level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    
    args = parser.parse_args()
    logger = setup_logging(args.log_level)

    # 检查依赖
    if not all(check_dependencies().values()):
        logger.error("依赖缺失，退出")
        sys.exit(1)

    pdb_dir = Path(args.pdb_dir)
    fasta_file = Path(args.fasta_file) if args.fasta_file else None
    output_dir = Path(args.output_dir) if args.output_dir else pdb_dir.parent / "mutifeature"

    # 读取配置，确定需要生成哪些特征文件
    config_path = Path(args.config) if args.config else None
    enabled_features = load_enabled_features(config_path, logger)
    
    if not pdb_dir.exists():
        logger.error("PDB目录不存在")
        sys.exit(1)

    if fasta_file is not None and not fasta_file.exists():
        logger.error("FASTA文件不存在")
        sys.exit(1)

    (output_dir / 'model_cache').mkdir(parents=True, exist_ok=True)

    # 只有在 HMM 步骤有启用的特征时才需要 FASTA
    hmm_needed = bool(get_needed_files_for_step("HMM特征", enabled_features))
    if hmm_needed and fasta_file is None:
        logger.info("HMM 步骤需要 FASTA，开始从 PDB 自动生成")
        fasta_file = generate_fasta_from_pdb(pdb_dir, output_dir, logger)
        if fasta_file is None:
            logger.error("无法生成可用的 FASTA，退出")
            sys.exit(1)
    elif not hmm_needed and fasta_file is None:
        logger.info("HMM 步骤未启用，无需 FASTA 文件，跳过生成。")
    
    start_time = time.time()

    # 定义流程
    # 格式: (步骤名称, 执行函数)
    steps = [
        ("s1ssttoken", lambda: run_s1ssttoken_extraction(pdb_dir, output_dir, logger)),
        ("HMM特征", lambda: run_hmm_feature_extraction(fasta_file, output_dir, logger)),
        ("快速特征", lambda: run_fast_feature_extraction(pdb_dir, output_dir, logger))
    ]

    for step_name, step_func in steps:
        # 0. 根据 enabled 配置判断该步骤是否需要执行
        needed = get_needed_files_for_step(step_name, enabled_features)
        if not needed:
            logger.info(f"步骤 '{step_name}' 的所有输出文件均未启用，跳过。")
            continue

        # 1. 检查是否已存在（s1ssttoken 步骤传入 pdb_dir 做增量检查）
        step_pdb_dir = pdb_dir if step_name == 's1ssttoken' else None
        if check_step_outputs(output_dir, step_name, logger, enabled_features, pdb_dir=step_pdb_dir):
            continue

        # 2. 执行步骤
        success = step_func()

        # 3. 针对 s1ssttoken 的特殊逻辑：失败不中断，仅记录
        if not success:
            if step_name == "s1ssttoken":
                logger.warning(f"跳过 {step_name} 生成 (允许失败)，继续执行...")
            else:
                logger.error(f"步骤 {step_name} 失败，可能会影响最终结果。")

    # 验证与报告
    verification = verify_output_files(output_dir, logger, enabled_features)
    validate_feature_lengths(fasta_file, output_dir, verification, logger)
    
    logger.info(f"完成! 总耗时: {time.time() - start_time:.2f}s")
    logger.info(f"输出目录: {output_dir}")

if __name__ == "__main__":
    main()