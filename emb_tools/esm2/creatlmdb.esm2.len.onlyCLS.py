import os
import torch
import numpy as np
import lmdb
from transformers import AutoTokenizer, AutoModel
from Bio import SeqIO
import time
from tqdm import tqdm
import shutil



# 设置设备，优先使用 GPU
os.environ["INFRA_PROVIDER"] = "True"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 初始化 Hugging Face 的 ESM-2 模型和 tokenizer (全局初始化)
# 这里的模型名称 "esm2_t48_15B_UR50D" 将从 Hugging Face Hub 下载
#MODEL_NAME = "data/esm2/esm2_t48_15B_UR50D"
#MODEL_NAME = "facebook/esm2_t36_3B_UR50D"
#MODEL_NAME = "facebook/esm2_t30_150M_UR50D"
#MODEL_NAME = "facebook/esm2_t33_650M_UR50D"
MODEL_NAME = "facebook/esm2_t48_15B_UR50D"
#MODEL_NAME = "facebook/esm1b_t33_650M_UR50S"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16).to(device)
print(f"Loaded model's hidden_size: {model.config.hidden_size}")

def read_fasta_sequences(fasta_file_path):
    """读取FASTA文件并返回字典 {seq_id: sequence}"""
    sequences = {}
    try:
        with open(fasta_file_path, 'r') as f:
            for record in SeqIO.parse(f, 'fasta'):
                sequences[record.id] = str(record.seq)
    except FileNotFoundError:
        print(f"错误: 文件 {fasta_file_path} 不存在。")
    except Exception as e:
        print(f"读取 FASTA 文件时发生错误: {e}")
    return sequences


def get_esm_embedding(model, tokenizer, device, sequence):
    """计算 ESM-2 的 token 级别嵌入"""
    try:
        with torch.no_grad():
            inputs = tokenizer(sequence, return_tensors="pt", add_special_tokens=True).to(device)

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                outputs = model(**inputs)

            embeddings = outputs.last_hidden_state
            # 添加此行来打印实际的形状
            #print(f"Embedding tensor shape: {embeddings.shape}") 
            last_layer_embedding = embeddings[0, 0].float().cpu().numpy()
            #print(f"Final embedding dimension: {last_layer_embedding.shape[-1]}")
            return last_layer_embedding
    except Exception as e:
        print(f"嵌入计算错误: {e}")
        return None


def calculate_embedding_and_save(model, tokenizer, device, txn, seq_id, seq):
    """计算嵌入并保存到 LMDB。"""
    embedding = get_esm_embedding(model, tokenizer, device, seq)
    if embedding is not None:
        txn.put(seq_id.encode(), embedding.tobytes())
        return True
    return False


def estimate_remaining_time(start_time, processed_count, total_sequences, time_per_seq):
    """估算剩余时间。"""
    avg_time = sum(time_per_seq) / len(time_per_seq) if time_per_seq else 0
    remaining_time = avg_time * (total_sequences - processed_count)
    elapsed_time = time.time() - start_time
    return elapsed_time, remaining_time


def process_and_save_sequences(fasta_file_path, output_file_path, model, tokenizer, device, save_interval=10, commit_interval=100):
    """逐步处理序列，计算嵌入并流式保存"""
    sequences = read_fasta_sequences(fasta_file_path)
    total_sequences = len(sequences)
    processed_count = 0
    start_time = time.time()
    time_per_seq = []
    
    print(f"批量提交间隔: 每{commit_interval}个序列提交一次事务")

    # 检查输出 LMDB 数据库路径是否存在
    if os.path.exists(output_file_path):
        print(f"数据库 {output_file_path} 已存在。")
        while True:
            choice = input("您想 (1) 删除重建，还是 (2) 追加到现有数据库？ 请输入 1 或 2: ")
            if choice == '1':
                try:
                    if os.path.isdir(output_file_path):
                        shutil.rmtree(output_file_path)
                    else:
                        os.remove(output_file_path)
                    print(f"已删除现有数据库 {output_file_path}，将重新创建。")
                    break
                except Exception as e:
                    print(f"删除数据库时出错: {e}")
                    print("请检查是否有其他程序正在使用该数据库，或检查文件权限。")
                    return
            elif choice == '2':
                print(f"将追加到现有数据库 {output_file_path}。")
                break
            else:
                print("输入无效，请输入 1 或 2。")

    print(f"总共 {total_sequences} 个序列, 将全部重新处理。")

    try:
        with lmdb.open(output_file_path, map_size=int(1.5 * 10**12)) as env:
            progress_bar = tqdm(sequences.items(), desc="Processing sequences", total=total_sequences)
            
            # 批量处理缓冲区
            batch_buffer = []
            
            for seq_id, seq in progress_bar:
                seq_start = time.time()
                embedding = get_esm_embedding(model, tokenizer, device, seq)
                
                if embedding is not None:
                    # 添加到缓冲区
                    batch_buffer.append((seq_id.encode(), embedding.tobytes()))
                    
                    seq_time = time.time() - seq_start
                    time_per_seq.append(seq_time)
                    if len(time_per_seq) > 10:
                        time_per_seq.pop(0)
                    processed_count += 1
                    
                    elapsed_time, remaining_time = estimate_remaining_time(start_time, processed_count, total_sequences, time_per_seq)
                    progress_bar.set_postfix_str(f"用时: {elapsed_time:.2f}s, 预计剩余: {remaining_time / 60:.2f} 分钟, 缓冲: {len(batch_buffer)}")

                    # 达到提交间隔或者是最后一个序列时，执行批量写入
                    if len(batch_buffer) >= commit_interval or processed_count == total_sequences:
                        with env.begin(write=True) as txn:
                            for key, value in batch_buffer:
                                txn.put(key, value)
                        batch_buffer.clear()  # 清空缓冲区
                        
                        # 清理GPU缓存
                        if processed_count % save_interval == 0:
                            torch.cuda.empty_cache()

    except KeyboardInterrupt:
        print("\n检测到手动中断，正在保存已处理的数据...")
        # 保存剩余的缓冲区数据
        if batch_buffer:
            try:
                with lmdb.open(output_file_path, map_size=int(1.5 * 10**12)) as env:
                    with env.begin(write=True) as txn:
                        for key, value in batch_buffer:
                            txn.put(key, value)
                print(f"已保存 {len(batch_buffer)} 个剩余序列的嵌入")
            except Exception as e:
                print(f"保存剩余数据时出错: {e}")
        print("程序停止。")
        exit()

    print(f"\n所有序列已计算完成，嵌入已重新保存到 {output_file_path}")


if __name__ == "__main__":
    fasta_file_path = './dataset/S1/protein.fasta'  # 请替换为您的 FASTA 文件路径
    output_file_path = './emb/esm2_15b.5120.lmdb/esm2_15b.5120.lmdb.onlyCLS.lmdb'  # 请替换为您希望保存 LMDB 文件的路径
    
    # commit_interval 参数控制批量提交频率：每处理100个序列提交一次事务
    process_and_save_sequences(fasta_file_path, output_file_path, model, tokenizer, device, commit_interval=100)
