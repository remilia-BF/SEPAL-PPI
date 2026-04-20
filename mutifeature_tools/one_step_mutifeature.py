#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
一键式多特征工具脚本
自动调用所有4个脚本来生成mutifeature JSON文件

使用方法:
python one_step_mutifeature.py -p /path/to/pdb/files -f /path/to/fasta/file -o /path/to/output

参数:
-p, --pdb_dir: PDB文件目录 (必需)
-f, --fasta_file: FASTA文件路径 (必需)  
-o, --output_dir: 输出目录 (可选，默认在PDB目录的父目录下)

示例:
python one_step_mutifeature.py -p predict/rice/pdb -f predict/rice/rice.test.fasta -o predict/rice/mutifeature
python one_step_mutifeature.py -p predict/rice/pdb -f predict/rice/rice.test.fasta

SEPAL 0.5.2
"""

import os
import sys
import argparse
import subprocess
import time
import json
from pathlib import Path
from typing import Optional, Dict, List, Any
import logging

def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """设置日志记录"""
    logger = logging.getLogger("one_step_mutifeature")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level.upper()))
    
    # 创建格式器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(formatter)
    
    # 添加处理器到logger
    if not logger.handlers:
        logger.addHandler(console_handler)
    
    return logger

def check_dependencies() -> Dict[str, bool]:
    """检查依赖脚本是否存在"""
    script_dir = Path(__file__).parent
    dependencies = {
        'cereat_prosst.py': script_dir / 'cereat_prosst.py',
        'generate_hmm_features.py': script_dir / 'generate_hmm_features.py', 
        'pdb2pqrjson.py': script_dir / 'pdb2pqrjson.py',
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
                        if current_id in lengths:
                            logger.warning(f"FASTA存在重复ID: {current_id}，将使用最后一次出现的序列")
                        lengths[current_id] = len(sequence)
                    current_id = line[1:].split()[0]
                    seq_parts = []
                else:
                    seq_parts.append(line)
        if current_id is not None:
            sequence = ''.join(seq_parts)
            if current_id in lengths:
                logger.warning(f"FASTA存在重复ID: {current_id}，将使用最后一次出现的序列")
            lengths[current_id] = len(sequence)
    except FileNotFoundError:
        logger.error(f"FASTA文件不存在: {fasta_path}")
        return {}
    except Exception as exc:
        logger.error(f"读取FASTA失败: {exc}")
        return {}
    
    logger.info(f"FASTA序列解析完成: {len(lengths)} 个条目")
    return lengths

def determine_feature_length(value: Any) -> Optional[int]:
    """推断特征字段的长度,用于和FASTA序列长度对比"""
    if isinstance(value, list):
        return len(value)
    if isinstance(value, tuple):
        return len(value)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped == '':
            return 0
        
        # 处理嵌套列表格式,如 "[0,0,1],[0,1,0],..." (二级结构的独热编码)
        if '],' in stripped:
            # 按 "]," 分割来计数子列表数量
            # 例如 "[0,0,1],[0,1,0]" 分割为 ["[0,0,1", "[0,1,0]"]
            parts = stripped.split('],')
            return len(parts)
        
        # 处理普通的JSON数组格式
        if '[' in stripped or ']' in stripped:
            candidates = [stripped]
            if not (stripped.startswith('[') and stripped.endswith(']')):
                candidates.append(f"[{stripped}]")
            for candidate in candidates:
                try:
                    parsed = json.loads(candidate)
                    if isinstance(parsed, list):
                        return len(parsed)
                except json.JSONDecodeError:
                    continue
        
        # 处理逗号分隔的数值列表
        if ',' in stripped:
            tokens = [token.strip() for token in stripped.split(',') if token.strip()]
            return len(tokens)
        return 1
    return None

def write_failed_proteins(output_dir: Path, failures: List[Dict[str, str]], logger: logging.Logger):
    """将失败详情写入failed_proteins.txt"""
    failed_path = output_dir / 'failed_proteins.txt'
    try:
        with open(failed_path, 'w', encoding='utf-8') as failed_file:
            if not failures:
                failed_file.write("All feature files matched the FASTA sequence lengths.\n")
            else:
                for item in failures:
                    lines = [
                        f"file: {item.get('file', '-')}",
                        f"protein_id: {item.get('protein_id', '-')}",
                        f"field: {item.get('field', '-')}",
                        f"expected_length: {item.get('expected', '-')}",
                        f"actual_length: {item.get('actual', '-')}",
                        f"reason: {item.get('reason', '-')}"
                    ]
                    failed_file.write('\n'.join(lines) + '\n')
                    failed_file.write('-' * 40 + '\n')
        logger.info(f"失败详情写入: {failed_path}")
    except Exception as exc:
        logger.error(f"写入failed_proteins.txt失败: {exc}")

def validate_feature_lengths(fasta_file: Path, output_dir: Path,
                             verification_results: Dict[str, bool],
                             logger: logging.Logger):
    """检查所有JSON特征文件与FASTA长度是否一致"""
    protein_lengths = load_fasta_lengths(fasta_file, logger)
    if not protein_lengths:
        logger.warning("FASTA序列为空或解析失败，跳过长度验证")
        write_failed_proteins(output_dir, [], logger)
        return
    
    failures: List[Dict[str, str]] = []
    for filename, exists in verification_results.items():
        if not exists:
            logger.warning(f"跳过长度验证，文件不存在: {filename}")
            continue
        file_path = output_dir / filename
        if not file_path.exists():
            logger.warning(f"跳过长度验证，找不到文件: {file_path}")
            continue
        try:
            with open(file_path, 'r', encoding='utf-8') as feature_file:
                data = json.load(feature_file)
        except Exception as exc:
            logger.error(f"读取{filename}失败: {exc}")
            continue
        if not isinstance(data, list):
            logger.warning(f"文件格式异常(应为list): {filename}")
            continue
        total_entries = len(data)
        mismatched_entries = 0
        for entry in data:
            if not isinstance(entry, dict):
                mismatched_entries += 1
                failures.append({
                    'file': filename,
                    'protein_id': 'UNKNOWN',
                    'field': '-',
                    'expected': 'N/A',
                    'actual': 'N/A',
                    'reason': 'invalid_entry'
                })
                continue
            protein_id = entry.get('protein_id')
            if not protein_id:
                mismatched_entries += 1
                failures.append({
                    'file': filename,
                    'protein_id': 'UNKNOWN',
                    'field': '-',
                    'expected': 'N/A',
                    'actual': 'N/A',
                    'reason': 'missing_protein_id'
                })
                continue
            expected_length = protein_lengths.get(protein_id)
            if expected_length is None:
                mismatched_entries += 1
                failures.append({
                    'file': filename,
                    'protein_id': protein_id,
                    'field': '-',
                    'expected': 'N/A',
                    'actual': 'N/A',
                    'reason': 'protein_not_in_fasta'
                })
                continue
            entry_mismatch = False
            for key, value in entry.items():
                if key == 'protein_id' or isinstance(value, dict):
                    continue
                length = determine_feature_length(value)
                if length is None:
                    continue
                if length != expected_length:
                    entry_mismatch = True
                    failures.append({
                        'file': filename,
                        'protein_id': protein_id,
                        'field': key,
                        'expected': str(expected_length),
                        'actual': str(length),
                        'reason': 'length_mismatch'
                    })
            if entry_mismatch:
                mismatched_entries += 1
        logger.info(f"{filename}: {mismatched_entries}/{total_entries} entries length mismatch")
    
    write_failed_proteins(output_dir, failures, logger)

def run_script(script_path: Path, args: List[str], logger: logging.Logger) -> bool:
    """运行Python脚本"""
    try:
        cmd = [sys.executable, str(script_path)] + args
        logger.info(f"执行命令: {' '.join(cmd)}")
        
        # 执行脚本
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=None)  # 无超时限制
        
        if result.returncode == 0:
            logger.info(f"脚本执行成功: {script_path.name}")
            if result.stdout:
                logger.debug(f"输出: {result.stdout}")
            return True
        else:
            logger.error(f"脚本执行失败: {script_path.name}")
            logger.error(f"错误输出: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error(f"脚本执行超时: {script_path.name}")
        return False
    except Exception as e:
        logger.error(f"执行脚本时出错: {script_path.name} - {e}")
        return False

def check_step_outputs(output_dir: Path, step_name: str, logger: logging.Logger, pdb_dir: Optional[Path] = None) -> tuple[bool, List[str], List[str]]:
    """检查某个步骤的输出文件是否都已存在
    
    返回:
        (all_exist, existing_files, missing_files)
    
    对于 PROSST 步骤，如果提供了 pdb_dir，会智能比较 JSON 中的蛋白与 PDB 目录的差异。
    """
    step_files = {
        'PROSST特征提取': ['prosst_features.json'],
        'HMM特征提取': [
            'hmm_domain_flag.json',
            'hmm_pfam_id.json',
            'hmm_confidence_score.json',
            'hmm_emission_prob.json',
            'hmm_core_site_flag.json',
            'hmm_boundary_site_flag.json',
            'hmm_binding_domain_flag.json'
        ],
        'PQR特征提取': ['pqr_feature.json'],
        '快速特征提取': [
            'sasa_features.json',
            'secondary_structure_features.json',
            'hydrophobicity_features.json'
        ]
    }
    
    expected_files = step_files.get(step_name, [])
    if not expected_files:
        return False, [], []
    
    existing = []
    missing = []
    
    for filename in expected_files:
        file_path = output_dir / filename
        if file_path.exists() and file_path.stat().st_size > 0:
            existing.append(filename)
        else:
            missing.append(filename)
    
    all_exist = len(missing) == 0
    
    if all_exist:
        # 对 PROSST 步骤做增量检查
        if step_name == 'PROSST特征提取' and pdb_dir is not None:
            json_file = output_dir / expected_files[0]  # prosst_features.json
            try:
                import json as _json
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = _json.load(f)
                json_ids = {entry.get('protein_id') for entry in data if isinstance(entry, dict)}
                pdb_ids = {p.stem for p in pdb_dir.iterdir() if p.suffix.lower() == '.pdb'}
                new_ids = pdb_ids - json_ids
                if new_ids:
                    logger.info(f"步骤 '{step_name}': 发现 {len(new_ids)} 个新蛋白需要追加处理")
                    return False, existing, list(new_ids)
                else:
                    logger.info(f"步骤 '{step_name}' 已包含所有 {len(pdb_ids)} 个蛋白，跳过")
            except Exception as exc:
                logger.warning(f"检查 {step_name} 增量状态时出错: {exc}，将重新执行")
                return False, existing, missing
        
        logger.info(f"步骤 '{step_name}' 的所有输出文件已存在，将跳过")
        for f in existing:
            logger.debug(f"  ✓ {f}")
    elif existing:
        # 对 PROSST 步骤不删除旧文件，交给 --append 处理
        if step_name != 'PROSST特征提取':
            logger.warning(f"步骤 '{step_name}' 的输出文件部分存在，将重新生成所有文件")
            logger.warning(f"  已存在: {', '.join(existing)}")
            logger.warning(f"  缺失: {', '.join(missing)}")
            for f in existing:
                try:
                    (output_dir / f).unlink()
                    logger.debug(f"  删除旧文件: {f}")
                except Exception as e:
                    logger.warning(f"  删除文件失败 {f}: {e}")
        else:
            logger.info(f"步骤 '{step_name}' 输出文件不完整，将以追加模式重新执行")
    
    return all_exist, existing, missing

def create_output_structure(output_dir: Path, logger: logging.Logger) -> bool:
    """创建输出目录结构"""
    try:
        # 创建主输出目录
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建子目录
        subdirs = ['PQR', 'model_cache']
        for subdir in subdirs:
            (output_dir / subdir).mkdir(exist_ok=True)
        
        logger.info(f"创建输出目录结构: {output_dir}")
        return True
        
    except Exception as e:
        logger.error(f"创建输出目录结构失败: {e}")
        return False

def run_prosst_feature_extraction(pdb_dir: Path, output_dir: Path, logger: logging.Logger) -> bool:
    """运行PROSST特征提取"""
    logger.info("=" * 50)
    logger.info("步骤 1: 运行PROSST特征提取")
    logger.info("=" * 50)
    
    # 设置输出文件路径
    prosst_output = output_dir / "prosst_features.json"
    
    # 构建参数
    args = [
        '--input', str(pdb_dir),
        '--output', str(prosst_output),
        '--vocab_size', '2048',
        '--mode', 'multiprocess',
        '--append'
    ]
    
    # 运行脚本
    script_path = Path(__file__).parent / 'cereat_prosst.py'
    return run_script(script_path, args, logger)

def run_hmm_feature_extraction(fasta_file: Path, output_dir: Path, logger: logging.Logger) -> bool:
    """运行HMM特征提取"""
    logger.info("=" * 50)
    logger.info("步骤 2: 运行HMM特征提取")
    logger.info("=" * 50)
    
    # 使用项目根目录下的全局缓存目录: ./cache/hmm/domains
    # 这样与项目中其他脚本的默认值保持一致，避免写入到输出目录下的子路径
    repo_root = Path(__file__).resolve().parents[1]
    hmm_data_dir = repo_root / 'cache' / 'hmm' / 'domains'

    # 构建参数
    args = [
        '--fasta', str(fasta_file),
        '--output', str(output_dir),
        '--hmm_data_dir', str(hmm_data_dir),
        '--hmm_db_path', './data/weights/Pfam-A.hmm'
    ]
    
    # 运行脚本
    script_path = Path(__file__).parent / 'generate_hmm_features.py'
    return run_script(script_path, args, logger)

def run_pqr_feature_extraction(pdb_dir: Path, output_dir: Path, logger: logging.Logger) -> bool:
    """运行PQR特征提取"""
    logger.info("=" * 50)
    logger.info("步骤 3: 运行PQR特征提取")
    logger.info("=" * 50)
    
    # 构建参数
    args = [
        '--input_dir', str(pdb_dir),
        '--output_dir', str(output_dir / 'PQR'),
        '--forcefield', 'PARSE',
        '--ph', '7.2',
        '--workers', '8'
    ]
    
    # 运行脚本
    script_path = Path(__file__).parent / 'pdb2pqrjson.py'
    return run_script(script_path, args, logger)

def run_fast_feature_extraction(pdb_dir: Path, output_dir: Path, logger: logging.Logger) -> bool:
    """运行快速特征提取"""
    logger.info("=" * 50)
    logger.info("步骤 4: 运行快速特征提取")
    logger.info("=" * 50)
    
    # 构建参数
    args = [
        '--input_dir', str(pdb_dir),
        '--output_dir', str(output_dir),
        '--probe_radius', '1.4',
        '--num_workers', '8'
    ]
    
    # 运行脚本
    script_path = Path(__file__).parent / 'extract_features_fast.py'
    return run_script(script_path, args, logger)

def verify_output_files(output_dir: Path, logger: logging.Logger) -> Dict[str, bool]:
    """验证输出文件"""
    expected_files = {
        'prosst_features.json': 'PROSST特征文件',
        'hmm_domain_flag.json': 'HMM结构域标志特征',
        'hmm_pfam_id.json': 'HMM Pfam ID特征',
        'hmm_confidence_score.json': 'HMM置信度分数特征',
        'hmm_emission_prob.json': 'HMM发射概率特征',
        'hmm_core_site_flag.json': 'HMM核心位点标志特征',
        'hmm_boundary_site_flag.json': 'HMM边界位点标志特征',
        'hmm_binding_domain_flag.json': 'HMM结合结构域标志特征',
        'pqr_feature.json': 'PQR残基电荷特征',
        'sasa_features.json': 'SASA特征',
        'secondary_structure_features.json': '二级结构特征',
        'hydrophobicity_features.json': '疏水性特征'
    }
    
    verification_results = {}
    logger.info("=" * 50)
    logger.info("验证输出文件")
    logger.info("=" * 50)
    
    for filename, description in expected_files.items():
        file_path = output_dir / filename
        exists = file_path.exists()
        verification_results[filename] = exists
        
        if exists:
            # 检查文件大小
            file_size = file_path.stat().st_size
            logger.info(f"✓ {description}: {filename} ({file_size} bytes)")
        else:
            logger.warning(f"✗ {description}: {filename} (文件不存在)")
    
    return verification_results

def generate_summary_report(output_dir: Path, verification_results: Dict[str, bool], 
                          execution_time: float, logger: logging.Logger):
    """生成总结报告"""
    logger.info("=" * 50)
    logger.info("处理完成总结")
    logger.info("=" * 50)
    
    total_files = len(verification_results)
    successful_files = sum(verification_results.values())
    failed_files = total_files - successful_files
    
    logger.info(f"总执行时间: {execution_time:.2f} 秒")
    logger.info(f"输出目录: {output_dir}")
    logger.info(f"成功生成文件: {successful_files}/{total_files}")
    logger.info(f"失败文件数: {failed_files}")
    
    if failed_files > 0:
        logger.warning("以下文件生成失败:")
        for filename, success in verification_results.items():
            if not success:
                logger.warning(f"  - {filename}")
    
    # 生成JSON格式的总结报告
    summary = {
        "execution_time_seconds": round(execution_time, 2),
        "output_directory": str(output_dir),
        "total_expected_files": total_files,
        "successful_files": successful_files,
        "failed_files": failed_files,
        "file_status": verification_results
    }
    
    summary_file = output_dir / "processing_summary.json"
    try:
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        logger.info(f"总结报告已保存到: {summary_file}")
    except Exception as e:
        logger.error(f"保存总结报告失败: {e}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="一键式多特征工具 - 自动生成所有mutifeature JSON文件",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    python one_step_mutifeature.py -p predict/rice/pdb -f predict/rice/rice.test.fasta -o predict/rice/mutifeature
    python one_step_mutifeature.py -p predict/rice/pdb -f predict/rice/rice.test.fasta

输出文件:
    - prosst_features.json: PROSST结构特征
    - hmm_*.json: HMM相关特征 (7个文件)
    - pqr_feature.json: PQR残基电荷特征
    - sasa_features.json: SASA特征
    - secondary_structure_features.json: 二级结构特征
    - hydrophobicity_features.json: 疏水性特征
    - processing_summary.json: 处理总结报告
        """
    )
    
    parser.add_argument('-p', '--pdb_dir', type=str, required=True,
                       help='PDB文件目录路径')
    parser.add_argument('-f', '--fasta_file', type=str, required=True,
                       help='FASTA文件路径')
    parser.add_argument('-o', '--output_dir', type=str, default=None,
                       help='输出目录路径 (默认: PDB目录的父目录下的mutifeature文件夹)')
    parser.add_argument('--log_level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='日志级别 (默认: INFO)')
    
    args = parser.parse_args()
    
    # 设置日志
    logger = setup_logging(args.log_level)
    
    # 检查依赖
    logger.info("检查依赖脚本...")
    dependencies = check_dependencies()
    if not all(dependencies.values()):
        logger.error("部分依赖脚本缺失，请确保所有脚本文件存在")
        sys.exit(1)
    
    # 验证输入路径
    pdb_dir = Path(args.pdb_dir)
    fasta_file = Path(args.fasta_file)
    
    if not pdb_dir.exists():
        logger.error(f"PDB目录不存在: {pdb_dir}")
        sys.exit(1)
    
    if not fasta_file.exists():
        logger.error(f"FASTA文件不存在: {fasta_file}")
        sys.exit(1)
    
    # 确定输出目录
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        # 默认在PDB目录的父目录下创建mutifeature文件夹
        output_dir = pdb_dir.parent / "mutifeature"
    
    logger.info(f"PDB目录: {pdb_dir}")
    logger.info(f"FASTA文件: {fasta_file}")
    logger.info(f"输出目录: {output_dir}")
    
    # 创建输出目录结构
    if not create_output_structure(output_dir, logger):
        sys.exit(1)
    
    # 记录开始时间
    start_time = time.time()
    
    # 执行所有特征提取步骤
    steps = [
        ("PROSST特征提取", lambda: run_prosst_feature_extraction(pdb_dir, output_dir, logger)),
        ("HMM特征提取", lambda: run_hmm_feature_extraction(fasta_file, output_dir, logger)),
        ("PQR特征提取", lambda: run_pqr_feature_extraction(pdb_dir, output_dir, logger)),
        ("快速特征提取", lambda: run_fast_feature_extraction(pdb_dir, output_dir, logger))
    ]
    
    step_results = {}
    skipped_steps = []
    
    for step_name, step_func in steps:
        logger.info(f"\n{'='*50}")
        logger.info(f"检查步骤: {step_name}")
        logger.info(f"{'='*50}")
        
        # 检查该步骤的输出文件是否都已存在（PROSST步骤传入pdb_dir做增量检查）
        step_pdb_dir = pdb_dir if step_name == 'PROSST特征提取' else None
        all_exist, existing_files, missing_files = check_step_outputs(output_dir, step_name, logger, pdb_dir=step_pdb_dir)
        
        if all_exist:
            # 所有文件都存在，跳过该步骤
            logger.info(f"⏭ 跳过 {step_name} (所有输出文件已存在)")
            step_results[step_name] = True
            skipped_steps.append(step_name)
            continue
        
        # 文件不完整或不存在，执行该步骤
        logger.info(f"开始执行: {step_name}")
        step_start_time = time.time()
        
        success = step_func()
        step_results[step_name] = success
        
        step_time = time.time() - step_start_time
        if success:
            logger.info(f"✓ {step_name} 完成 (耗时: {step_time:.2f}秒)")
        else:
            logger.error(f"✗ {step_name} 失败 (耗时: {step_time:.2f}秒)")
    
    # 计算总执行时间
    total_time = time.time() - start_time
    
    # 验证输出文件
    verification_results = verify_output_files(output_dir, logger)
    
    # 生成总结报告
    generate_summary_report(output_dir, verification_results, total_time, logger)

    # 校验特征长度并输出失败详情
    validate_feature_lengths(fasta_file, output_dir, verification_results, logger)
    
    # 最终状态
    successful_steps = sum(step_results.values())
    total_steps = len(step_results)
    executed_steps = total_steps - len(skipped_steps)
    
    logger.info("=" * 50)
    logger.info("执行总结")
    logger.info("=" * 50)
    
    if skipped_steps:
        logger.info(f"跳过的步骤 ({len(skipped_steps)}):")
        for step in skipped_steps:
            logger.info(f"  ⏭ {step}")
    
    if executed_steps > 0:
        logger.info(f"实际执行的步骤: {executed_steps}/{total_steps}")
    
    if successful_steps == total_steps:
        logger.info("✓ 所有步骤成功!")
    else:
        logger.warning(f"⚠ 部分步骤失败: {successful_steps}/{total_steps}")
        failed_steps = [name for name, success in step_results.items() if not success]
        for step in failed_steps:
            logger.error(f"  ✗ {step}")
    
    logger.info(f"总耗时: {total_time:.2f} 秒")
    logger.info(f"输出目录: {output_dir}")
    logger.info("=" * 50)

if __name__ == "__main__":
    main() 