#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
优化的快速蛋白质特征提取脚本
减少SASA计算采样点，使用更少的进程数

使用方法:
python extract_features_fast.py -i /path/to/pdb/files -o /path/to/output

参数:
-i, --input_dir: PDB文件输入目录 (必需)
-o, --output_dir: 输出目录 (可选，默认: mutifeature)
--probe_radius: SASA计算的探针半径 (可选，默认: 1.4)
--num_workers: 进程数量 (可选，默认: 4)

输出:
- {output_dir}/sasa_features.json (字段名保留 sasa，但值为 RSA)
- {output_dir}/secondary_structure_features.json (独热编码格式)
- {output_dir}/hydrophobicity_features.json
"""

import os
import sys
import json
import argparse
import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
warnings.filterwarnings('ignore')

try:
    from Bio import PDB
    from Bio.PDB.DSSP import DSSP, residue_max_acc
    from Bio.PDB.PDBParser import PDBParser
except ImportError as e:
    print(f"错误：无法导入Biopython。请安装: pip install biopython")
    print(f"具体错误: {e}")
    exit(1)


class FastProteinFeatureExtractor:
    """快速蛋白质特征提取器"""
    
    def __init__(self):
        self.parser = PDBParser(QUIET=True)
        self.acc_array = 'Sander'
        self.residue_max_acc = residue_max_acc[self.acc_array]
        
        # 二级结构到独热编码的映射
        self.ss_to_onehot = {
            'H': [1, 0, 0],  # α-螺旋
            'E': [0, 1, 0],  # β-折叠  
            'C': [0, 0, 1]   # 无规卷曲
        }
        
        # 氨基酸疏水性量表 (Kyte-Doolittle)
        self.hydrophobicity_scale = {
            'ALA': 1.8, 'ARG': -4.5, 'ASN': -3.5, 'ASP': -3.5,
            'CYS': 2.5, 'GLN': -3.5, 'GLU': -3.5, 'GLY': -0.4,
            'HIS': -3.2, 'ILE': 4.5, 'LEU': 3.8, 'LYS': -3.9,
            'MET': 1.9, 'PHE': 2.8, 'PRO': -1.6, 'SER': -0.8,
            'THR': -0.7, 'TRP': -0.9, 'TYR': -1.3, 'VAL': 4.2
        }
        
    def calculate_sasa_dssp(self, structure, pdb_file: str) -> Dict[str, float]:
        """
        使用DSSP计算RSA（relative solvent accessibility）
        """
        sasa_dict = {}
        
        try:
            model = structure[0]
            
            # 检查PDB文件格式，如果缺少CRYST1记录，添加一个
            temp_pdb = self._prepare_pdb_for_dssp(pdb_file)
            
            try:
                # DSSP参数设置
                # 使用默认参数，这是合理的设置
                # DSSP会自动处理探针半径、原子半径等参数
                # 如果需要自定义参数，可以这样设置：
                # dssp = DSSP(model, temp_pdb, dssp='mkdssp', acc_array='Miller')
                dssp = DSSP(model, temp_pdb, acc_array=self.acc_array)
                
                for key in dssp.keys():
                    chain_id = key[0]
                    res_num = key[1][1]
                    sasa_value = dssp[key][3]  # Biopython DSSP 第4列是相对ASA（RSA）
                    
                    res_id = f"{chain_id}_{res_num}"
                    sasa_dict[res_id] = round(sasa_value, 3)
                
                # 清理临时文件
                if os.path.exists(temp_pdb):
                    os.remove(temp_pdb)
                    
            except Exception as dssp_error:
                print(f"DSSP分析失败: {dssp_error}")
                print("使用内部SASA计算...")
                sasa_dict = self._calculate_sasa_fallback(structure)
                
        except Exception as e:
            print(f"SASA提取失败: {e}")
            print("使用内部SASA计算...")
            sasa_dict = self._calculate_sasa_fallback(structure)
            
        return sasa_dict
    
    def _calculate_sasa_fallback(self, structure) -> Dict[str, float]:
        """
        内部SASA计算作为备用方案，并按 DSSP 的 Sander 标准归一化为 RSA
        """
        sasa_dict = {}
        
        # 获取所有原子
        atoms = list(structure.get_atoms())
        
        # 原子半径字典 (van der Waals半径)
        atom_radii = {
            'C': 1.7, 'N': 1.55, 'O': 1.52, 'S': 1.8,
            'P': 1.8, 'H': 1.2, 'F': 1.47, 'CL': 1.75,
            'BR': 1.85, 'I': 1.98, 'MG': 1.73, 'CA': 2.31,
            'FE': 2.23, 'ZN': 2.29, 'MN': 2.24
        }
        
        for model in structure:
            for chain in model:
                for residue in chain:
                    if residue.get_id()[0] == ' ':  # 只处理标准氨基酸
                        try:
                            residue_sasa = 0.0
                            residue_atoms = list(residue.get_atoms())
                            
                            for atom in residue_atoms:
                                # 获取原子半径
                                element = atom.element.upper() if atom.element else 'C'
                                radius = atom_radii.get(element, 1.7)
                                
                                # 快速计算SASA
                                atom_sasa = self._calculate_atom_sasa_fast(
                                    atom, atoms, radius, 1.4
                                )
                                residue_sasa += atom_sasa
                            
                            # 残基ID: 链_残基号
                            res_id = f"{chain.id}_{residue.id[1]}"
                            residue_name = residue.get_resname().strip().upper()
                            sasa_dict[res_id] = self._normalize_sasa_to_rsa(residue_name, residue_sasa)
                            
                        except Exception as e:
                            print(f"计算残基 {residue} SASA时出错: {e}")
                            
        return sasa_dict

    def _normalize_sasa_to_rsa(self, residue_name: str, absolute_sasa: float) -> float:
        """将绝对SASA按 DSSP 的最大ASA表归一化为 RSA。"""
        max_acc = self.residue_max_acc.get(residue_name)
        if not max_acc or max_acc <= 0:
            return 0.0

        relative_acc = absolute_sasa / max_acc
        if relative_acc > 1.0:
            relative_acc = 1.0
        elif relative_acc < 0.0:
            relative_acc = 0.0

        return round(relative_acc, 3)
    
    def _calculate_atom_sasa_fast(self, target_atom, nearby_atoms: List, 
                                radius: float, probe_radius: float, 
                                n_points: int = 256) -> float:  # 减少采样点
        """
        快速计算单个原子的SASA - 使用更少的采样点
        """
        accessible_points = 0
        total_radius = radius + probe_radius
        
        # 预计算其他原子的信息以提高效率
        other_atoms_info = []
        for other_atom in nearby_atoms:
            if other_atom == target_atom:
                continue
            other_element = other_atom.element.upper() if other_atom.element else 'C'
            other_radius = {
                'C': 1.7, 'N': 1.55, 'O': 1.52, 'S': 1.8,
                'P': 1.8, 'H': 1.2
            }.get(other_element, 1.7)
            
            # 只考虑可能重叠的原子（距离足够近）
            center_distance = np.linalg.norm(target_atom.coord - other_atom.coord)
            if center_distance < (total_radius + other_radius + probe_radius):
                other_atoms_info.append((other_atom.coord, other_radius + probe_radius))
        
        # 使用更少的采样点
        for i in range(n_points):
            # 使用Fibonacci球面点分布，更均匀
            y = 1 - (i / float(n_points - 1)) * 2  # y goes from 1 to -1
            radius_at_y = np.sqrt(1 - y * y)
            
            theta = (i * 2.399963) % (2 * np.pi)  # golden angle increment
            
            x = np.cos(theta) * radius_at_y
            z = np.sin(theta) * radius_at_y
            
            # 测试点的绝对坐标
            test_point = target_atom.coord + total_radius * np.array([x, y, z])
            
            # 检查是否与其他原子冲突
            is_accessible = True
            for other_coord, other_total_radius in other_atoms_info:
                distance = np.linalg.norm(test_point - other_coord)
                if distance < other_total_radius:
                    is_accessible = False
                    break
            
            if is_accessible:
                accessible_points += 1
        
        # 计算表面积
        surface_area = 4 * np.pi * (total_radius ** 2) * (accessible_points / n_points)
        return surface_area
    
    def extract_secondary_structure_simple(self, structure) -> Dict[str, str]:
        """
        简化的二级结构预测（基于phi/psi角度）
        """
        ss_dict = {}
        
        for model in structure:
            for chain in model:
                residues = list(chain)
                
                for i, residue in enumerate(residues):
                    if residue.get_id()[0] == ' ':  # 只处理标准氨基酸
                        res_id = f"{chain.id}_{residue.id[1]}"
                        
                        try:
                            # 计算phi和psi角度
                            phi, psi = self._calculate_angles(residues, i)
                            
                            # 基于Ramachandran图的简单分类
                            if phi is not None and psi is not None:
                                if -180 <= phi <= -30 and -70 <= psi <= 50:
                                    ss_dict[res_id] = 'H'  # α-helix
                                elif -180 <= phi <= -30 and 70 <= psi <= 180:
                                    ss_dict[res_id] = 'E'  # β-strand
                                else:
                                    ss_dict[res_id] = 'C'  # coil
                            else:
                                ss_dict[res_id] = 'C'
                                
                        except Exception:
                            ss_dict[res_id] = 'C'
                            
        return ss_dict
    
    def _calculate_angles(self, residues: List, index: int) -> Tuple[Optional[float], Optional[float]]:
        """计算phi和psi双面角"""
        try:
            current = residues[index]
            
            phi = None
            psi = None
            
            # 计算phi角度 (需要前一个残基)
            if index > 0:
                prev = residues[index - 1]
                if 'C' in prev and 'N' in current and 'CA' in current and 'C' in current:
                    phi = np.degrees(self._calc_dihedral(
                        prev['C'].get_vector(),
                        current['N'].get_vector(),
                        current['CA'].get_vector(),
                        current['C'].get_vector()
                    ))
            
            # 计算psi角度 (需要下一个残基)
            if index < len(residues) - 1:
                next_res = residues[index + 1]
                if 'N' in current and 'CA' in current and 'C' in current and 'N' in next_res:
                    psi = np.degrees(self._calc_dihedral(
                        current['N'].get_vector(),
                        current['CA'].get_vector(),
                        current['C'].get_vector(),
                        next_res['N'].get_vector()
                    ))
            
            return phi, psi
            
        except Exception:
            return None, None
    
    def _calc_dihedral(self, v1, v2, v3, v4):
        """计算二面角"""
        from Bio.PDB.vectors import calc_dihedral
        return calc_dihedral(v1, v2, v3, v4)
    
    def _prepare_pdb_for_dssp(self, pdb_file: str) -> str:
        """
        为DSSP准备PDB文件，添加必要的CRYST1记录
        
        Args:
            pdb_file: 原始PDB文件路径
            
        Returns:
            临时PDB文件路径
        """
        import tempfile
        
        temp_pdb = tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False)
        temp_pdb_path = temp_pdb.name
        
        with open(pdb_file, 'r') as f:
            lines = f.readlines()
        # 过滤掉 DBREF/DBREF1/DBREF2 行（有些解析器在存在该记录时会出错）
        filtered_lines = [line for line in lines if not line.lstrip().startswith('DBREF')]
        removed_dbref = len(lines) - len(filtered_lines)

        # 检查是否已有CRYST1记录（在过滤后的内容上检查）
        has_cryst1 = any(line.startswith('CRYST1') for line in filtered_lines)

        with open(temp_pdb_path, 'w') as f:
            if not has_cryst1:
                # 添加一个默认的CRYST1记录
                f.write("CRYST1    1.000    1.000    1.000  90.00  90.00  90.00 P 1           1\n")

            # 写入过滤后的内容
            f.writelines(filtered_lines)

        
        return temp_pdb_path

    def _collect_standard_residue_indices(self, structure) -> Dict[str, List[int]]:
        """收集每条链的标准残基编号列表"""
        chain_indices: Dict[str, List[int]] = {}
        for model in structure:
            for chain in model:
                indices = [residue.id[1] for residue in chain if residue.get_id()[0] == ' ']
                if indices:
                    chain_indices[chain.id] = sorted(set(indices))
            # 只使用第一个模型
            break
        return chain_indices

    def _interpolate_missing_residues(self, sasa_dict: Dict[str, float], expected_range: range,
                                      chain_id: str, fallback_values: Optional[Dict[str, float]] = None) -> List[int]:
        """
        通过插值/备用方法填充缺失的SASA值

        Args:
            sasa_dict: DSSP返回的字典，形如 {"A_1": 0.5}
            expected_range: 预期的残基编号范围，例如 range(1, 88)
            chain_id: 链ID，默认'A'
            fallback_values: 可选的备用SASA字典
        Returns:
            缺失的残基编号列表
        """
        missing: List[int] = []
        for res_num in expected_range:
            key = f"{chain_id}_{res_num}"
            if key in sasa_dict:
                continue
            missing.append(res_num)
            prev_val = sasa_dict.get(f"{chain_id}_{res_num - 1}")
            next_val = sasa_dict.get(f"{chain_id}_{res_num + 1}")
            if prev_val is not None and next_val is not None:
                interpolated = round((prev_val + next_val) / 2.0, 3)
            elif prev_val is not None:
                interpolated = prev_val
            elif next_val is not None:
                interpolated = next_val
            elif fallback_values and key in fallback_values:
                interpolated = fallback_values[key]
            else:
                interpolated = 0.0
            sasa_dict[key] = interpolated
        return missing

    def fill_missing_sasa_values(self, structure, sasa_dict: Dict[str, float]) -> Dict[str, List[int]]:
        """
        填充缺失的SASA并返回缺失残基信息
        
        策略:
        1. 对比PDB实际残基范围和DSSP返回的残基,找出缺失的
        2. 优先使用邻居插值填充
        3. 只有当无法插值时,才调用fallback计算(且只计算缺失的残基)
        """
        chain_indices = self._collect_standard_residue_indices(structure)
        if not chain_indices:
            return {}

        missing_summary: Dict[str, List[int]] = {}
        present_chains = {key.split('_')[0] for key in sasa_dict.keys()}
        target_chains = present_chains or chain_indices.keys()

        for chain_id in target_chains:
            indices = chain_indices.get(chain_id)
            if not indices:
                continue
            
            # expected_range 使用 PDB 中实际存在的残基范围
            # 例如 PDB 有残基 [1,2,3,...,87], 则 expected_range = range(1, 88)
            expected_range = range(indices[0], indices[-1] + 1)
            
            # 找出DSSP中缺失的残基
            missing_res_nums = [res_num for res_num in expected_range 
                               if f"{chain_id}_{res_num}" not in sasa_dict]
            
            if not missing_res_nums:
                continue
            
            # 检查哪些缺失残基无法通过插值填充(前后邻居都不存在)
            needs_fallback = []
            for res_num in missing_res_nums:
                prev_key = f"{chain_id}_{res_num - 1}"
                next_key = f"{chain_id}_{res_num + 1}"
                if prev_key not in sasa_dict and next_key not in sasa_dict:
                    needs_fallback.append(res_num)
            
            # 只在需要时才调用备用SASA计算,并且只计算需要的残基
            fallback_values = None
            if needs_fallback:
                # 注意: 这里仍然计算整个结构,但只会用到needs_fallback中的值
                # 如果想进一步优化,可以修改_calculate_sasa_fallback只计算特定残基
                fallback_values = self._calculate_sasa_fallback(structure)
            
            missing = self._interpolate_missing_residues(
                sasa_dict, expected_range, chain_id, fallback_values
            )
            
            if missing:
                missing_summary[chain_id] = missing

        return missing_summary
    
    def calculate_hydrophobicity(self, structure) -> Dict[str, float]:
        """
        计算每个残基的疏水性值
        使用Kyte-Doolittle疏水性量表
        """
        hydrophobicity_dict = {}
        
        for model in structure:
            for chain in model:
                for residue in chain:
                    if residue.get_id()[0] == ' ':  # 只处理标准氨基酸
                        try:
                            # 获取残基名称
                            res_name = residue.get_resname()
                            
                            # 获取疏水性值
                            hydrophobicity = self.hydrophobicity_scale.get(res_name, 0.0)
                            
                            # 残基ID: 链_残基号
                            res_id = f"{chain.id}_{residue.id[1]}"
                            hydrophobicity_dict[res_id] = round(hydrophobicity, 3)
                            
                        except Exception as e:
                            print(f"计算残基 {residue} 疏水性时出错: {e}")
                            
        return hydrophobicity_dict
    
    def ss_to_onehot_encoding(self, ss_dict: Dict[str, str]) -> Dict[str, str]:
        """
        将二级结构转换为独热编码格式
        """
        onehot_dict = {}
        
        for res_id, ss_code in ss_dict.items():
            # 获取独热编码
            onehot = self.ss_to_onehot.get(ss_code, [0, 0, 1])  # 默认使用coil编码
            
            # 转换为逗号分隔的字符串
            onehot_str = ",".join(map(str, onehot))
            onehot_dict[res_id] = onehot_str
            
        return onehot_dict


def process_single_file_fast(pdb_file: str) -> Optional[Dict]:
    """
    快速处理单个PDB文件
    
    Args:
        pdb_file: PDB文件路径
        
    Returns:
        包含处理结果的字典
    """
    try:
        protein_id = os.path.splitext(os.path.basename(pdb_file))[0]
        
        # 创建提取器实例
        extractor = FastProteinFeatureExtractor()
        
        # 解析PDB文件
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('protein', pdb_file)
        
        # 使用DSSP提取SASA
        sasa_dict = extractor.calculate_sasa_dssp(structure, pdb_file)
        missing_residues = extractor.fill_missing_sasa_values(structure, sasa_dict)
        if missing_residues:
            print(f"[{protein_id}] DSSP跳过残基: {missing_residues}，已使用插值/备用方法填充")
        
        # 简化二级结构预测
        ss_dict = extractor.extract_secondary_structure_simple(structure)
        
        # 计算疏水性
        hydrophobicity_dict = extractor.calculate_hydrophobicity(structure)
        
        # 处理SASA结果
        sasa_str = ""
        if sasa_dict:
            res_numbers = []
            for key in sasa_dict.keys():
                if '_' in key:
                    try:
                        res_num = int(key.split('_')[1])
                        res_numbers.append(res_num)
                    except:
                        continue
            
            if res_numbers:
                min_res = min(res_numbers)
                max_res = max(res_numbers)
                
                sasa_values = []
                for i in range(min_res, max_res + 1):
                    sasa_val = sasa_dict.get(f"A_{i}", 0.0)
                    sasa_values.append(str(sasa_val))
                
                sasa_str = ",".join(sasa_values)
        
        # 处理二级结构结果
        ss_str = ""
        if ss_dict:
            onehot_dict = extractor.ss_to_onehot_encoding(ss_dict)
            
            res_numbers = []
            for key in ss_dict.keys():
                if '_' in key:
                    try:
                        res_num = int(key.split('_')[1])
                        res_numbers.append(res_num)
                    except:
                        continue
            
            if res_numbers:
                min_res = min(res_numbers)
                max_res = max(res_numbers)
                
                onehot_sequence = []
                for i in range(min_res, max_res + 1):
                    onehot_code = onehot_dict.get(f"A_{i}", "0,0,1")
                    # 将独热编码用方括号框起来
                    onehot_sequence.append(f"[{onehot_code}]")
                
                ss_str = ",".join(onehot_sequence)
        
        # 处理疏水性结果
        hydrophobicity_str = ""
        if hydrophobicity_dict:
            res_numbers = []
            for key in hydrophobicity_dict.keys():
                if '_' in key:
                    try:
                        res_num = int(key.split('_')[1])
                        res_numbers.append(res_num)
                    except:
                        continue
            
            if res_numbers:
                min_res = min(res_numbers)
                max_res = max(res_numbers)
                
                hydrophobicity_values = []
                for i in range(min_res, max_res + 1):
                    hydrophobicity_val = hydrophobicity_dict.get(f"A_{i}", 0.0)
                    hydrophobicity_values.append(str(hydrophobicity_val))
                
                hydrophobicity_str = ",".join(hydrophobicity_values)
        
        return {
            'protein_id': protein_id,
            'sasa_dict': sasa_dict,
            'ss_dict': ss_dict,
            'hydrophobicity_dict': hydrophobicity_dict,
            'sasa_str': sasa_str,
            'ss_str': ss_str,
            'hydrophobicity_str': hydrophobicity_str,
            'missing_sasa_residues': missing_residues
        }
        
    except Exception as e:
        print(f"处理文件 {pdb_file} 时出错: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description='快速提取蛋白质 RSA 和二级结构特征')
    parser.add_argument('-i', '--input_dir', required=True, help='PDB文件输入目录')
    parser.add_argument('-o', '--output_dir', default='mutifeature', help='输出目录 (默认: mutifeature)')
    parser.add_argument('--probe_radius', type=float, default=1.4, help='SASA计算的探针半径 (默认: 1.4)')
    parser.add_argument('--num_workers', type=int, default=4, help='进程数量 (默认: 4)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_dir):
        print(f"错误：输入目录不存在: {args.input_dir}")
        return
    
    print("=== 快速蛋白质特征提取器 ===")
    print(f"输入目录: {args.input_dir}")
    print(f"输出目录: {args.output_dir}")
    print(f"探针半径: {args.probe_radius}")
    print(f"进程数量: {args.num_workers}")
    
    # 确保输出目录存在
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 收集所有PDB文件
    pdb_files = []
    for file in os.listdir(args.input_dir):
        if file.endswith('.pdb'):
            pdb_files.append(os.path.join(args.input_dir, file))
    
    print(f"找到 {len(pdb_files)} 个PDB文件，使用 {args.num_workers} 个进程处理…")
    
    # 使用进程池处理文件
    results = []
    start_time = time.time()
    
    failed_proteins = []

    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        # 提交所有任务
        future_to_file = {
            executor.submit(process_single_file_fast, pdb_file): pdb_file 
            for pdb_file in pdb_files
        }
        
        # 收集结果
        for future in as_completed(future_to_file):
            pdb_file = future_to_file[future]
            protein_id = os.path.splitext(os.path.basename(pdb_file))[0]
            try:
                result = future.result()
                if result:
                    results.append(result)
                else:
                    failed_proteins.append(protein_id)
            except Exception:
                failed_proteins.append(protein_id)
    
    # 分离SASA、二级结构和疏水性结果
    sasa_results = []
    ss_results = []
    hydrophobicity_results = []
    
    for result in results:
        if result['sasa_dict']:
            sasa_entry = {
                "protein_id": result['protein_id'],
                "sasa": result['sasa_str']
            }
            if result.get('missing_sasa_residues'):
                sasa_entry["missing_residues"] = result['missing_sasa_residues']
            sasa_results.append(sasa_entry)
        
        if result['ss_dict']:
            ss_results.append({
                "protein_id": result['protein_id'],
                "secondary_structure": result['ss_str']
            })
        
        if result['hydrophobicity_dict']:
            hydrophobicity_results.append({
                "protein_id": result['protein_id'],
                "hydrophobicity": result['hydrophobicity_str']
            })
    
    # 保存失败列表
    failed_list_path = os.path.join(output_dir, 'failed_proteins.txt')
    with open(failed_list_path, 'w', encoding='utf-8') as f:
        for pid in failed_proteins:
            f.write(pid + "\n")

    # 保存SASA结果
    sasa_output = os.path.join(output_dir, 'sasa_features.json')
    with open(sasa_output, 'w', encoding='utf-8') as f:
        json.dump(sasa_results, f, indent=2, ensure_ascii=False)
    print(f"SASA特征已保存到: {sasa_output}")
    
    # 保存二级结构结果
    ss_output = os.path.join(output_dir, 'secondary_structure_features.json')
    with open(ss_output, 'w', encoding='utf-8') as f:
        json.dump(ss_results, f, indent=2, ensure_ascii=False)
    print(f"二级结构特征已保存到: {ss_output}")
    
    # 保存疏水性结果
    hydrophobicity_output = os.path.join(output_dir, 'hydrophobicity_features.json')
    with open(hydrophobicity_output, 'w', encoding='utf-8') as f:
        json.dump(hydrophobicity_results, f, indent=2, ensure_ascii=False)
    print(f"疏水性特征已保存到: {hydrophobicity_output}")
    
    # 显示统计信息
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"=== 提取完成 ===")
    print(f"处理文件数: {len(pdb_files)}")
    print(f"成功提取SASA: {len(sasa_results)}")
    print(f"成功提取二级结构: {len(ss_results)}")
    print(f"成功提取疏水性: {len(hydrophobicity_results)}")
    print(f"失败文件数: {len(failed_proteins)} (列表: {failed_list_path})")
    print(f"总耗时: {total_time:.2f} 秒")
    print(f"平均每文件: {total_time/len(pdb_files):.2f} 秒")
    print(f"输出目录: {output_dir}")
    print("特征提取完成!")


if __name__ == "__main__":
    main() 