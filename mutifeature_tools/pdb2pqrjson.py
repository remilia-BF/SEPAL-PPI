#!/usr/bin/env python3
"""
PDB2PQR批量转换和残基电荷特征提取工具

该脚本用于将指定目录下的所有PDB文件使用PDB2PQR转换为PQR格式，
并提取每个残基的总电荷，生成JSON格式的特征文件。

用法:
    python pdb2pqrjson.py --input_dir /path/to/pdb/files --output_dir /path/to/pqr/files
    python pdb2pqrjson.py -i dataset/S1/testpdb -o mutifeature/PQR

作者: SEPAL Team
版本: 2.0.0
"""

import os
import sys
import argparse
import subprocess
import logging
import json
import re
from pathlib import Path
from typing import List, Optional, Dict, Tuple
import concurrent.futures
import time
from collections import defaultdict

def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """设置日志记录"""
    logger = logging.getLogger("pdb2pqr_converter")
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

def find_pdb_files(input_dir: Path) -> List[Path]:
    """在指定目录中查找所有PDB文件"""
    pdb_files = []
    for ext in ['*.pdb', '*.PDB']:
        pdb_files.extend(input_dir.glob(ext))
    return sorted(pdb_files)

def check_pdb2pqr_available() -> bool:
    """检查PDB2PQR是否可用"""
    try:
        result = subprocess.run(['pdb2pqr', '--help'], 
                              capture_output=True, text=True, timeout=10)
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False

def parse_pqr_file(pqr_file: Path, logger: Optional[logging.Logger] = None) -> Dict[str, float]:
    """
    解析PQR文件，计算每个残基的总电荷
    
    参数:
        pqr_file: PQR文件路径
        logger: 日志记录器
        
    返回:
        dict: 残基ID -> 总电荷的映射
    """
    if logger is None:
        logger = logging.getLogger("pdb2pqr_converter")
    
    residue_charges = defaultdict(float)
    
    # 用于修复坐标字段粘连问题：当坐标值 <= -100 时，PQR 固定宽度格式
    # 会导致相邻字段无空格，如 "-22.464-102.231"，需要插入空格
    _fix_merged = re.compile(r'(\d)(-)')

    try:
        with open(pqr_file, 'r') as f:
            for line in f:
                # 解析ATOM行
                if line.startswith('ATOM') or line.startswith('HETATM'):
                    # PQR格式: ATOM/HETATM 原子序号 原子名 残基名 残基序号 x y z 电荷 半径
                    # 修复坐标粘连：在 "数字-" 之间插入空格，例如 "30.871-42.665" -> "30.871 -42.665"
                    line = _fix_merged.sub(r'\1 \2', line)
                    parts = line.split()
                    if len(parts) >= 10:
                        try:
                            residue_name = parts[3]
                            residue_number = parts[4]  # PQR文件中第5个字段是残基序号
                            charge = float(parts[8])    # 第9个字段是电荷
                            
                            # 创建残基标识符: 残基名_残基序号
                            residue_id = f"{residue_name}_{residue_number}"
                            residue_charges[residue_id] += charge
                            
                        except (ValueError, IndexError) as e:
                            logger.debug(f"跳过无效行: {line.strip()}, 错误: {e}")
                            continue
    
    except Exception as e:
        logger.error(f"解析PQR文件失败 {pqr_file}: {str(e)}")
        return {}
    
    # 将defaultdict转换为常规dict并保留合理精度
    result = {res_id: round(charge, 4) for res_id, charge in residue_charges.items()}
    
    logger.debug(f"从 {pqr_file.name} 解析出 {len(result)} 个残基")
    return result

def extract_protein_id(pdb_file: Path) -> str:
    """从PDB文件名提取蛋白质ID"""
    return pdb_file.stem

def convert_pdb_to_pqr(pdb_file: Path, output_dir: Path, 
                      forcefield: str = "PARSE", 
                      ph: float = 7.2,
                      logger: Optional[logging.Logger] = None) -> Tuple[bool, Dict[str, float]]:
    """
    将单个PDB文件转换为PQR格式
    
    参数:
        pdb_file: 输入PDB文件路径
        output_dir: 输出目录路径
        forcefield: 力场类型 (AMBER, CHARMM, PARSE)
        ph: pH值
        logger: 日志记录器
        
    返回:
        Tuple[bool, Dict[str, float]]: (转换是否成功, 残基电荷字典)
    """
    if logger is None:
        logger = logging.getLogger("pdb2pqr_converter")
    
    # 生成输出文件路径
    pqr_file = output_dir / f"{pdb_file.stem}.pqr"
    
    # 检查输出文件是否已存在
    if pqr_file.exists():
        logger.info(f"跳过 {pdb_file.name} - 输出文件已存在")
        # 解析已存在的PQR文件
        residue_charges = parse_pqr_file(pqr_file, logger)
        return True, residue_charges
    
    # 构建PDB2PQR命令
    cmd = [
        'pdb2pqr',
        '--ff', forcefield,
        '--with-ph', str(ph),
        '--drop-water',  # 删除水分子
        str(pdb_file),
        str(pqr_file)
    ]
    
    try:
        logger.info(f"开始转换 {pdb_file.name} -> {pqr_file.name}")
        
        # 执行转换命令
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            logger.info(f"成功转换 {pdb_file.name}")
            # 解析生成的PQR文件
            residue_charges = parse_pqr_file(pqr_file, logger)
            return True, residue_charges
        else:
            logger.error(f"转换失败 {pdb_file.name}: {result.stderr}")
            return False, {}
            
    except subprocess.TimeoutExpired:
        logger.error(f"转换超时 {pdb_file.name} (>300秒)")
        return False, {}
    except Exception as e:
        logger.error(f"转换出错 {pdb_file.name}: {str(e)}")
        return False, {}

def convert_batch_parallel(pdb_files: List[Path], output_dir: Path,
                         forcefield: str = "PARSE", ph: float = 7.2,
                         max_workers: int = 4, logger: Optional[logging.Logger] = None) -> Tuple[dict, Dict[str, List[float]]]:
    """
    并行批量转换PDB文件
    
    参数:
        pdb_files: PDB文件列表
        output_dir: 输出目录
        forcefield: 力场类型
        ph: pH值
        max_workers: 最大并行工作线程数
        logger: 日志记录器
        
    返回:
        Tuple[dict, Dict[str, List[float]]]: (转换结果统计, 蛋白质残基电荷数据)
    """
    if logger is None:
        logger = logging.getLogger("pdb2pqr_converter")
    
    results = {'success': 0, 'failed': 0, 'skipped': 0}
    protein_charges = {}  # 存储所有蛋白质的残基电荷数据
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有转换任务
        future_to_file = {
            executor.submit(convert_pdb_to_pqr, pdb_file, output_dir, 
                          forcefield, ph, logger): pdb_file 
            for pdb_file in pdb_files
        }
        
        # 处理完成的任务
        for future in concurrent.futures.as_completed(future_to_file):
            pdb_file = future_to_file[future]
            try:
                success, residue_charges = future.result()
                if success:
                    results['success'] += 1
                    # 提取蛋白质ID并存储残基电荷
                    protein_id = extract_protein_id(pdb_file)
                    if residue_charges:
                        # 将残基电荷字典按残基序号排序
                        # 字典键格式: "残基名_残基序号", 例如 "MET_1"
                        sorted_residues = sorted(
                            residue_charges.items(),
                            key=lambda x: int(x[0].split('_')[1])  # 按残基序号排序
                        )
                        # 只取电荷值,丢弃键
                        charge_list = [charge for _, charge in sorted_residues]
                        protein_charges[protein_id] = charge_list
                        logger.debug(f"收集到 {protein_id} 的 {len(charge_list)} 个残基电荷")
                else:
                    results['failed'] += 1
            except Exception as e:
                logger.error(f"处理 {pdb_file.name} 时出现异常: {str(e)}")
                results['failed'] += 1
    
    return results, protein_charges

def save_protein_charges_json(protein_charges: Dict[str, List[float]], 
                             output_path: Path, 
                             logger: Optional[logging.Logger] = None) -> bool:
    """
    保存蛋白质残基电荷数据到JSON文件
    
    参数:
        protein_charges: 蛋白质ID -> 残基电荷列表的映射
        output_path: JSON文件输出路径
        logger: 日志记录器
        
    返回:
        bool: 保存是否成功
    """
    if logger is None:
        logger = logging.getLogger("pdb2pqr_converter")
    
    try:
        # 转换数据格式为数组格式
        json_data = []
        for protein_id, charge_list in protein_charges.items():
            # 将浮点数列表转换为逗号分隔的字符串
            charge_str = ','.join([f"{charge:.4f}" for charge in charge_list])
            json_data.append({
                "protein_id": protein_id,
                "embeddings": charge_str
            })
        
        # 保存到JSON文件
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"成功保存残基电荷数据到 {output_path}")
        logger.info(f"包含 {len(json_data)} 个蛋白质的数据")
        return True
        
    except Exception as e:
        logger.error(f"保存JSON文件失败: {str(e)}")
        return False

def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="批量将PDB文件转换为PQR格式并生成残基电荷特征JSON文件",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    python pdb2pqrjson.py -i dataset/S1/testpdb -o mutifeature/PQR
    python pdb2pqrjson.py -i dataset/S1/testpdb -o mutifeature/PQR --forcefield AMBER --ph 6.5
    python pdb2pqrjson.py -i dataset/S1/testpdb -o mutifeature/PQR --workers 8
    
输出文件:
    - mutifeature/PQR/*.pqr: 转换后的PQR文件
    - mutifeature/pqr_feature.json: 残基电荷特征文件
        """
    )
    
    parser.add_argument('-i', '--input_dir', type=str, required=True,
                       help='输入PDB文件目录路径')
    parser.add_argument('-o', '--output_dir', type=str, required=True,
                       help='输出PQR文件目录路径')
    parser.add_argument('--forcefield', type=str, default='PARSE',
                       choices=['AMBER', 'CHARMM', 'PARSE'],
                       help='力场类型 (默认: PARSE)')
    parser.add_argument('--ph', type=float, default=7.2,
                       help='pH值 (默认: 7.2)')
    parser.add_argument('--workers', type=int, default=4,
                       help='并行工作线程数 (默认: 4)')
    parser.add_argument('--log_level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='日志级别 (默认: INFO)')
    
    args = parser.parse_args()
    
    # 设置日志
    logger = setup_logging(args.log_level)
    
    # 验证输入目录
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        logger.error(f"输入目录不存在: {input_dir}")
        sys.exit(1)
    
    if not input_dir.is_dir():
        logger.error(f"输入路径不是目录: {input_dir}")
        sys.exit(1)
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"输出目录: {output_dir}")
    
    # 检查PDB2PQR是否可用
    if not check_pdb2pqr_available():
        logger.error("PDB2PQR未找到或不可用。请确保已安装PDB2PQR并在PATH中。")
        logger.error("安装命令: conda install -c conda-forge pdb2pqr")
        sys.exit(1)
    
    # 查找PDB文件
    logger.info(f"在 {input_dir} 中搜索PDB文件...")
    pdb_files = find_pdb_files(input_dir)
    
    if not pdb_files:
        logger.warning(f"在 {input_dir} 中未找到任何PDB文件")
        sys.exit(0)
    
    logger.info(f"找到 {len(pdb_files)} 个PDB文件")
    
    # 显示参数信息
    logger.info(f"转换参数:")
    logger.info(f"  - 力场: {args.forcefield}")
    logger.info(f"  - pH值: {args.ph}")
    logger.info(f"  - 并行线程数: {args.workers}")
    
    # 开始批量转换
    start_time = time.time()
    logger.info("开始批量转换...")
    
    results, protein_charges = convert_batch_parallel(
        pdb_files, output_dir, 
        forcefield=args.forcefield,
        ph=args.ph,
        max_workers=args.workers,
        logger=logger
    )
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # 显示转换结果
    logger.info("=" * 50)
    logger.info("转换完成!")
    logger.info(f"总耗时: {elapsed_time:.2f} 秒")
    logger.info(f"成功转换: {results['success']} 个文件")
    logger.info(f"转换失败: {results['failed']} 个文件")
    logger.info(f"输出目录: {output_dir}")
    
    # 生成残基电荷特征JSON文件
    if protein_charges:
        # 使用输出目录的父目录作为JSON文件位置，与one_step_mutifeature.py保持一致
        # 使用绝对路径避免路径解析问题
        json_output_path = output_dir.parent.absolute() / "pqr_feature.json"
        json_output_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info("生成残基电荷特征文件...")
        logger.info(f"JSON文件将保存到: {json_output_path}")
        json_success = save_protein_charges_json(protein_charges, json_output_path, logger)
        
        if json_success:
            logger.info(f"特征文件保存位置: {json_output_path}")
        else:
            logger.error("特征文件保存失败")
    else:
        logger.warning("没有收集到任何残基电荷数据，跳过JSON文件生成")
    
    if results['failed'] > 0:
        logger.warning(f"有 {results['failed']} 个文件转换失败，请查看上方的错误信息")
        sys.exit(1)

if __name__ == "__main__":
    main()