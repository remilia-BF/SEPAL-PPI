#!/usr/bin/env python3
"""
HMM特征生成脚本
基于HMM结构域识别结果生成7个特征维度：
1. 结构域存在标志 (1维)
2. Pfam ID (1维) 
3. HMMer置信度分数 (1维)
4. 氨基酸发射概率 (1维)
5. 核心位点二进制标志 (1维)
6. 边界位点二进制标志 (1维)
7. 结合相关结构域二进制标志 (1维)
"""

import os
import sys
import json
import math
import re
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
from typing import Union, Optional

# 添加父目录到路径以导入HMM缓存管理器
#sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


from hmm_cache_manager import HMMCacheManager



class HMMFeatureGenerator:
    """HMM特征生成器"""
    
    def __init__(self, hmm_data_dir="./cache/hmm/domains", hmm_db_path="./data/weights/Pfam-A.hmm"):
        self.hmm_data_dir = Path(hmm_data_dir)
        self.hmm_db_path = hmm_db_path
        # 预加载 Pfam 短名->accession 数字ID 映射（无版本、无PF前缀）
        self.name_to_pfam_id_map = self._load_pfam_mapping()
        
        # 初始化HMM缓存管理器
        if HMMCacheManager is not None:
            self.hmm_cache_manager = HMMCacheManager(
                hmm_data_dir=str(self.hmm_data_dir),
                hmm_db_path=hmm_db_path
            )
        else:
            self.hmm_cache_manager = None
        
        # 结合相关结构域关键词
        self.binding_keywords = [
            'protein-protein interaction', 'PPI', 'binding', 'interface', 
            'dimerization', 'multimerization', 'complex', 'association',
            'recognition', 'docking', 'molecular recognition', 'interaction domain',
            'SH2', 'SH3', 'PDZ', 'BTB', 'WW', 'bromodomain', 'chromodomain', 
            'Pleckstrin homology', 'Src homology 2', 'Src homology 3',
            'PTB domain', 'FHA domain', 'binding domain', 'interaction',
            'receptor', 'ligand', 'adapter', 'scaffold', 'regulatory'
        ]
        
        # 核心位点关键词
        self.core_keywords = [
            'active site', 'catalytic', 'enzyme', 'kinase', 'phosphatase',
            'transferase', 'hydrolase', 'ligase', 'isomerase', 'lyase',
            'oxidoreductase', 'catalytic domain', 'active center', 'binding pocket',
            'substrate binding', 'cofactor binding', 'metal binding', 'nucleotide binding'
        ]
    
    def _default_pfam_mapping_path(self) -> Path:
        # 仓库根目录/data/weights/pfam_name_to_acc.noversion.noPF.tsv
        repo_root = Path(__file__).resolve().parents[1]
        return repo_root / 'data' / 'weights' / 'pfam_name_to_acc.noversion.noPF.tsv'

    def _load_pfam_mapping(self, mapping_path: Optional[Union[str, Path]] = None):
        """加载 Pfam 短名 -> accession 数字ID 的映射。
        文件格式：两列分隔（空白或制表），如：
          C2\t00168
        返回 dict，例如 { 'C2': 168 }
        """
        if mapping_path is None:
            mapping_path = self._default_pfam_mapping_path()
        mapping_path = Path(mapping_path)

        mapping = {}
        if not mapping_path.exists():
            raise FileNotFoundError(
                f"未找到Pfam映射文件: {mapping_path}. 请先生成 data/weights/pfam_name_to_acc.noversion.noPF.tsv 或提供正确路径。"
            )

        try:
            with open(mapping_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    parts = line.split()
                    if len(parts) < 2:
                        continue
                    name, acc_no_pf = parts[0], parts[1]
                    # 保留数字，去除非数字字符
                    digits = ''.join(ch for ch in acc_no_pf if ch.isdigit())
                    if digits:
                        try:
                            mapping[name] = int(digits)
                        except Exception:
                            # 忽略不可转换的条目
                            pass
        except Exception as e:
            raise RuntimeError(f"读取Pfam映射文件失败: {mapping_path}: {e}")

        if mapping:
            print(f"已加载Pfam映射 {len(mapping)} 条: {mapping_path}")
        else:
            print(f"警告: Pfam映射文件为空或未解析到条目: {mapping_path}")
        return mapping

    def extract_pfam_id(self, domain_name):
        """获取 Pfam ID（整数）。仅使用映射；若该短名未出现在映射中，则返回 0。"""
        if self.name_to_pfam_id_map and domain_name in self.name_to_pfam_id_map:
            return self.name_to_pfam_id_map[domain_name]
        return 0
    
    def is_binding_domain(self, domain_name):
        """判断是否为结合相关结构域"""
        domain_lower = domain_name.lower()
        return any(keyword in domain_lower for keyword in self.binding_keywords)
    
    def is_core_site(self, domain_name, position, domain_start, domain_end):
        """判断是否为核心位点"""
        domain_lower = domain_name.lower()
        
        # 检查是否为催化相关结构域
        is_catalytic = any(keyword in domain_lower for keyword in self.core_keywords)
        
        if is_catalytic:
            # 对于催化结构域，中心区域更可能是核心位点
            domain_center = (domain_start + domain_end) // 2
            center_margin = max(5, (domain_end - domain_start + 1) // 4)
            
            # 在中心区域内的位点更可能是核心位点
            if abs(position - domain_center) <= center_margin:
                return 1.0
            else:
                # 距离中心越远，核心位点概率越低
                distance = abs(position - domain_center)
                return max(0.0, 1.0 - (distance - center_margin) / center_margin)
        
        return 0.0
    
    def is_boundary_site(self, position, domain_start, domain_end):
        """判断是否为边界位点"""
        # 定义边界区域大小
        boundary_margin = min(5, max(2, (domain_end - domain_start + 1) // 6))
        
        # 检查是否在起始或结束边界附近
        if (domain_start - boundary_margin <= position <= domain_start + boundary_margin or
            domain_end - boundary_margin <= position <= domain_end + boundary_margin):
            return 1.0
        
        return 0.0
    
    def calculate_emission_probability(self, amino_acid, domain_name, confidence):
        """计算氨基酸发射概率（简化版本）"""
        # 这是一个简化的实现，实际应该基于HMM模型的发射概率矩阵
        # 这里我们基于置信度和氨基酸类型给出一个合理的估计
        
        # 常见氨基酸的基础概率
        common_aa = {'A', 'L', 'G', 'V', 'E', 'S', 'I', 'K', 'D', 'T', 'R', 'P', 'N', 'Q', 'F', 'Y', 'M', 'H', 'C', 'W'}
        
        if amino_acid in common_aa:
            # 基于置信度调整概率
            base_prob = 0.7  # 常见氨基酸的基础概率
            return base_prob + confidence * 0.3
        else:
            # 罕见氨基酸
            return 0.3 + confidence * 0.4
    
    def generate_residue_features(self, sequence, domains):
        """为序列中的每个残基生成特征"""
        seq_length = len(sequence)
        
        # 初始化特征数组
        domain_flag = np.zeros(seq_length, dtype=np.int32)      # 结构域存在标志 (整数)
        pfam_id = np.zeros(seq_length, dtype=np.int32)          # Pfam ID (整数)
        confidence_score = np.zeros(seq_length, dtype=np.float32) # HMMer置信度分数 (浮点数)
        emission_prob = np.zeros(seq_length, dtype=np.float32)    # 氨基酸发射概率 (浮点数)
        core_site_flag = np.zeros(seq_length, dtype=np.int32)   # 核心位点标志 (整数)
        boundary_site_flag = np.zeros(seq_length, dtype=np.int32) # 边界位点标志 (整数)
        binding_domain_flag = np.zeros(seq_length, dtype=np.int32) # 结合相关结构域标志 (整数)
        
        # 为每个结构域生成特征
        for start, end, domain_name, evalue, score, confidence in domains:
            # 确保索引在合理范围内
            start = max(0, min(start, seq_length - 1))
            end = max(start, min(end, seq_length - 1))
            
            # 提取Pfam ID
            pfam_id_value = self.extract_pfam_id(domain_name)
            
            # 判断是否为结合相关结构域
            is_binding = self.is_binding_domain(domain_name)
            
            # 为结构域内的每个残基生成特征
            for pos in range(start, end + 1):
                if pos < seq_length:
                    # 1. 结构域存在标志 (整数: 0或1)
                    domain_flag[pos] = 1
                    
                    # 2. Pfam ID (整数)
                    pfam_id[pos] = pfam_id_value
                    
                    # 3. HMMer置信度分数 (浮点数: 0.0-1.0)
                    confidence_score[pos] = confidence
                    
                    # 4. 氨基酸发射概率 (浮点数: 0.0-1.0)
                    if pos < len(sequence):
                        aa = sequence[pos]
                        emission_prob[pos] = self.calculate_emission_probability(aa, domain_name, confidence)
                    
                    # 5. 核心位点标志 (整数: 0或1)
                    core_site_flag[pos] = int(self.is_core_site(domain_name, pos, start, end) > 0.5)
                    
                    # 6. 边界位点标志 (整数: 0或1)
                    boundary_site_flag[pos] = int(self.is_boundary_site(pos, start, end) > 0.5)
                    
                    # 7. 结合相关结构域标志 (整数: 0或1)
                    binding_domain_flag[pos] = 1 if is_binding else 0
        
        return {
            'domain_flag': domain_flag,
            'pfam_id': pfam_id,
            'confidence_score': confidence_score,
            'emission_prob': emission_prob,
            'core_site_flag': core_site_flag,
            'boundary_site_flag': boundary_site_flag,
            'binding_domain_flag': binding_domain_flag
        }
    
    def features_to_string(self, features):
        """将特征数组转换为逗号分隔的字符串"""
        if features.dtype == np.int32:
            # 整数特征：直接转换为整数
            return ','.join([f"{int(val)}" for val in features])
        else:
            # 浮点数特征：保留6位小数
            return ','.join([f"{val:.6f}" for val in features])
    
    def load_hmm_cache(self, fasta_path):
        """加载HMM缓存"""
        if self.hmm_cache_manager is None:
            print("警告: HMM缓存管理器不可用，将使用空缓存")
            return {}
        
        try:
            # 首先检查FASTA文件哈希值对应的缓存是否存在
            fasta_hash = self.hmm_cache_manager.get_fasta_hash(fasta_path)
            expected_cache_path = self.hmm_cache_manager.get_cache_path(fasta_path)
            print(f"FASTA文件哈希值: {fasta_hash}")
            print(f"期望的缓存文件: {expected_cache_path}")
            
            if expected_cache_path.exists():
                print(f"找到对应的缓存文件: {expected_cache_path}")
                try:
                    import pickle
                    with open(expected_cache_path, 'rb') as f:
                        cache = pickle.load(f)
                    print(f"成功加载对应缓存，包含 {len(cache)} 个序列")
                    
                    # 检查缓存是否包含结构域信息
                    has_domain_info = False
                    for seq_id, domains in cache.items():
                        if domains:  # 如果有结构域信息
                            has_domain_info = True
                            break
                    
                    if has_domain_info:
                        print("缓存包含结构域信息，使用现有缓存")
                        return cache
                    else:
                        print("缓存文件存在但不包含结构域信息，需要重新创建")
                        
                except Exception as e:
                    print(f"加载对应缓存文件失败: {e}")
            
            # 如果没有对应的缓存文件，尝试查找现有的缓存文件
            existing_cache_files = list(self.hmm_data_dir.glob("hmm_cache_*.pkl"))
            if existing_cache_files:
                print(f"找到 {len(existing_cache_files)} 个现有缓存文件:")
                for cache_file in existing_cache_files:
                    print(f"  - {cache_file.name}")
                
                # 检查是否有包含当前序列的缓存文件
                current_sequences = set()
                try:
                    with open(fasta_path, 'r') as f:
                        current_id = None
                        for line in f:
                            line = line.strip()
                            if line.startswith('>'):
                                current_id = line[1:]
                                current_sequences.add(current_id)
                except Exception as e:
                    print(f"读取FASTA文件失败: {e}")
                    current_sequences = set()
                
                print(f"当前FASTA文件包含 {len(current_sequences)} 个序列")
                
                # 尝试从现有缓存中找到包含当前序列的缓存
                for cache_file in existing_cache_files:
                    try:
                        with open(cache_file, 'rb') as f:
                            cache = pickle.load(f)
                        
                        # 检查缓存中是否包含当前序列
                        cache_sequences = set(cache.keys())
                        common_sequences = current_sequences & cache_sequences
                        
                        if len(common_sequences) > 0:
                            print(f"缓存文件 {cache_file.name} 包含 {len(common_sequences)} 个当前序列")
                            if len(common_sequences) == len(current_sequences):
                                print(f"使用现有缓存文件: {cache_file.name}")
                                return cache
                            else:
                                print(f"缓存文件不完整，需要创建新缓存")
                                break
                    except Exception as e:
                        print(f"检查缓存文件 {cache_file.name} 失败: {e}")
                        continue
            
            # 如果没有找到合适的缓存，创建新的缓存
            print("未找到合适的缓存文件，开始创建新的HMM缓存...")
            cache = self.hmm_cache_manager.load_or_create_cache(fasta_path)
            if cache is None:
                print("警告: 无法创建HMM缓存，将使用空缓存")
                return {}
            
            print(f"成功创建新缓存，包含 {len(cache)} 个序列")
            return cache
            
        except Exception as e:
            print(f"加载HMM缓存时出错: {e}")
            return {}
    
    def process_fasta_file(self, fasta_path, output_dir="./mutifeature"):
        """处理FASTA文件并生成HMM特征"""
        print(f"处理FASTA文件: {fasta_path}")
        
        # 创建输出目录
        output_dir = Path(output_dir)
        # 递归创建输出目录，避免父目录不存在时报错
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 加载HMM缓存
        print("加载HMM缓存...")
        hmm_cache = self.load_hmm_cache(fasta_path)
        
        print(f"HMM缓存包含 {len(hmm_cache)} 个序列的结构域信息")
        
        # 读取FASTA文件
        sequences = {}
        try:
            with open(fasta_path, 'r') as f:
                current_id = None
                current_seq = []
                
                for line in f:
                    line = line.strip()
                    if line.startswith('>'):
                        if current_id is not None:
                            sequences[current_id] = ''.join(current_seq)
                        current_id = line[1:]
                        current_seq = []
                    else:
                        current_seq.append(line)
                
                if current_id is not None:
                    sequences[current_id] = ''.join(current_seq)
        
        except Exception as e:
            print(f"读取FASTA文件失败: {e}")
            return
        
        print(f"读取了 {len(sequences)} 个蛋白质序列")
        
        # 初始化结果字典
        results = {
            'domain_flag': [],
            'pfam_id': [],
            'confidence_score': [],
            'emission_prob': [],
            'core_site_flag': [],
            'boundary_site_flag': [],
            'binding_domain_flag': []
        }
        
        # 处理每个序列
        print("生成HMM特征...")
        for protein_id, sequence in tqdm(sequences.items(), desc="处理蛋白质"):
            # 获取结构域信息
            domains = hmm_cache.get(protein_id, [])
            
            # 生成特征
            features = self.generate_residue_features(sequence, domains)
            
            # 转换为字符串格式
            feature_strings = {
                'protein_id': protein_id,
                'domain_flag': self.features_to_string(features['domain_flag']),
                'pfam_id': self.features_to_string(features['pfam_id']),
                'confidence_score': self.features_to_string(features['confidence_score']),
                'emission_prob': self.features_to_string(features['emission_prob']),
                'core_site_flag': self.features_to_string(features['core_site_flag']),
                'boundary_site_flag': self.features_to_string(features['boundary_site_flag']),
                'binding_domain_flag': self.features_to_string(features['binding_domain_flag'])
            }
            
            # 添加到结果中
            for key in results:
                if key in feature_strings:
                    results[key].append(feature_strings)
        
        # 保存结果到JSON文件
        print("保存特征到JSON文件...")
        
        # 保存每个特征维度到单独的文件
        feature_files = {
            'domain_flag': 'hmm_domain_flag.json',
            'pfam_id': 'hmm_pfam_id.json', 
            'confidence_score': 'hmm_confidence_score.json',
            'emission_prob': 'hmm_emission_prob.json',
            'core_site_flag': 'hmm_core_site_flag.json',
            'boundary_site_flag': 'hmm_boundary_site_flag.json',
            'binding_domain_flag': 'hmm_binding_domain_flag.json'
        }
        
        for feature_name, filename in feature_files.items():
            if feature_name in results and results[feature_name]:
                output_file = output_dir / filename
                
                # 重新格式化数据以匹配目标格式
                formatted_data = []
                for item in results[feature_name]:
                    formatted_data.append({
                        'protein_id': item['protein_id'],
                        'embeddings': item[feature_name]
                    })
                
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(formatted_data, f, indent=2, ensure_ascii=False)
                
                print(f"已保存 {feature_name} 特征到: {output_file}")
        
        print(f"HMM特征生成完成！共处理 {len(sequences)} 个蛋白质序列")
        print(f"特征文件保存在: {output_dir}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="HMM特征生成脚本")
    parser.add_argument("--fasta", type=str, default="protein.fasta", 
                       help="输入FASTA文件路径")
    parser.add_argument("--output", type=str, default="./mutifeature",
                       help="输出目录路径")
    parser.add_argument("--hmm_data_dir", type=str, default="./cache/hmm/domains",
                       help="HMM数据目录路径")
    parser.add_argument("--hmm_db_path", type=str, default="./data/weights/Pfam-A.hmm",
                       help="HMM数据库路径")
    
    args = parser.parse_args()
    
    # 检查输入文件是否存在
    if not os.path.exists(args.fasta):
        print(f"错误: FASTA文件不存在: {args.fasta}")
        return
    
    # 检查HMM数据库是否存在
    if not os.path.exists(args.hmm_db_path):
        print(f"警告: HMM数据库不存在: {args.hmm_db_path}")
        print("将尝试使用现有缓存文件")
    
    # 创建特征生成器
    generator = HMMFeatureGenerator(
        hmm_data_dir=args.hmm_data_dir,
        hmm_db_path=args.hmm_db_path
    )
    
    # 处理FASTA文件
    generator.process_fasta_file(args.fasta, args.output)


if __name__ == "__main__":
    main() 