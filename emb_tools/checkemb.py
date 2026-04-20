#!/usr/bin/env python3

import lmdb
import numpy as np
import torch
import sys
import os
import struct
import pickle

# --- 可配置参数 ---
# 设置 LMDB 数据库文件的路径
lmdb_file_path = 'emb/Human/esm2_15b.Human.avg.1280.lmdb/noCLSeos.lmdb'

# 设置您要搜索的蛋白ID子字符串
search_string = "Q9HCD5"

# 指定嵌入的精度：'fp32', 'fp16', 或 'bf16'
# 确保这个设置与生成嵌入时使用的精度一致
embedding_precision = 'fp32' 

# 指定嵌入的维度，例如 ESM-2/650M 是 1280，ESM-2/150M 是 640
hidden_dim = 5120

def parse_embedding_data(value, precision, hidden_dim):
    """
    解析嵌入数据，支持多种格式：
    1. 原始二进制格式（fp32, fp16, bf16）
    2. int8格式（带scale前缀）
    3. pickle格式（向后兼容）
    """
    try:
        # 首先尝试作为原始二进制数据解析
        if precision == 'fp32':
            embedding = np.frombuffer(value, dtype=np.float32)
        elif precision == 'fp16':
            embedding = np.frombuffer(value, dtype=np.float16)
        elif precision == 'bf16':
            # bf16存储为uint16
            uint16_data = np.frombuffer(value, dtype=np.uint16)
            embedding_tensor = torch.from_numpy(uint16_data).view(torch.bfloat16)
            embedding = embedding_tensor.float().numpy()
        elif precision == 'int8':
            # int8格式：前4字节是scale，后面是量化数据
            if len(value) < 4:
                raise ValueError("int8数据长度不足")
            scale = struct.unpack('f', value[:4])[0]
            quantized_data = np.frombuffer(value[4:], dtype=np.int8)
            # 反量化
            embedding = quantized_data.astype(np.float32) * scale / 127.0
        else:
            raise ValueError(f"不支持的精度: {precision}")
        
        # 检查数据长度是否合理
        if len(embedding) == 0:
            raise ValueError("嵌入数据为空")
        
        return embedding
        
    except Exception as e:
        # 如果原始二进制解析失败，尝试pickle格式（向后兼容）
        try:
            embedding = pickle.loads(value)
            if hasattr(embedding, 'shape'):
                # 如果是numpy数组，转换为flatten格式
                embedding = embedding.flatten()
            else:
                # 如果是其他格式，尝试转换为numpy数组
                embedding = np.array(embedding, dtype=np.float32).flatten()
            return embedding
        except Exception as pickle_error:
            raise ValueError(f"无法解析嵌入数据: 原始格式错误 {e}, pickle格式错误 {pickle_error}")

# --- 核心逻辑 ---
print(f"已知蛋白Q9SUV6长度为 142，嵌入维度为 {hidden_dim}，目标精度为 {embedding_precision}")

try:
    # 以只读模式打开 LMDB 环境
    env = lmdb.open(lmdb_file_path, readonly=True, map_size=int(1.5 * 10**12))
    
    print(f"正在打开 LMDB 数据库: {lmdb_file_path}")

    with env.begin() as txn:
        cursor = txn.cursor()
        
        print(f"\n正在搜索所有包含 '{search_string}' 的蛋白ID...")

        found_count = 0
        for key, value in cursor:
            protein_id = key.decode('utf-8')

            if search_string in protein_id:
                found_count += 1
                try:
                    # 使用新的解析函数
                    embedding = parse_embedding_data(value, embedding_precision, hidden_dim)

                    # 检查嵌入的总长度是否为 hidden_dim 的整数倍
                    if len(embedding) % hidden_dim != 0:
                        raise ValueError(f"嵌入总长度 {len(embedding)} 不是隐藏维度 {hidden_dim} 的整数倍。数据可能已损坏或精度设置不正确。")

                    # 重新计算嵌入的形状
                    seq_len = len(embedding) // hidden_dim
                    
                    print(f"找到匹配项: {protein_id}")
                    print(f" - 嵌入形状: ({seq_len}, {hidden_dim})")
                    print(f" - 嵌入数据类型: {embedding.dtype}")
                    print(f" - 嵌入数据示例 (前5个值): {embedding.flatten()[:5]}")
                
                except ValueError as e:
                    print(f"处理蛋白ID {protein_id} 时出错: {e}")
                except Exception as e:
                    print(f"处理蛋白ID {protein_id} 时发生未知错误: {e}")
    
    if found_count == 0:
        print(f"在数据库中没有找到包含 '{search_string}' 的条目。")
        
    env.close()
    
    print("\n检查完成。")

except lmdb.Error as e:
    print(f"LMDB 数据库操作出错: {e}")
except FileNotFoundError:
    print(f"错误: 找不到 LMDB 文件 {lmdb_file_path}。请检查路径是否正确。")
except Exception as e:
    print(f"发生未知错误: {e}")