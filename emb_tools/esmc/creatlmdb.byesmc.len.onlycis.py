import os
import torch
import numpy as np
import lmdb
from esm.models.esmc import ESMC, ESMProteinTensor
from esm.sdk.api import LogitsConfig
from esm.tokenization import EsmSequenceTokenizer
from Bio import SeqIO
import time
from tqdm import tqdm
import shutil  # 导入 shutil 模块，用于删除目录


# 设置设备
os.environ["INFRA_PROVIDER"] = "True"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 初始化ESMC客户端和分词器 (全局初始化，在主程序入口处调用)
client = ESMC.from_pretrained("esmc_600m", device=device)
tokenizer = EsmSequenceTokenizer()


def read_fasta_sequences(fasta_file_path):
    """读取FASTA文件并返回字典 {seq_id: sequence}"""
    sequences = {}
    try:
        with open(fasta_file_path, 'r') as f:
            for record in SeqIO.parse(f, 'fasta'):
                sequences[record.id] = str(record.seq)
    except FileNotFoundError:
        print(f"Error: The file {fasta_file_path} does not exist.")
    except Exception as e:
        print(f"An error occurred while reading the FASTA file: {e}")
    return sequences


def get_esm_embedding(client, tokenizer, device, sequence):
    """计算 ESMC 的 token 级别嵌入"""
    try:
        with torch.no_grad():
            tokens = tokenizer.encode(sequence)
            protein_tensor = ESMProteinTensor(sequence=torch.tensor(tokens).to(device))
            logits_output = client.logits(protein_tensor, LogitsConfig(sequence=True, return_embeddings=True))
            embeddings = logits_output.embeddings
            last_layer_embedding = embeddings[-1, 0].cpu().numpy()  # 只保留第一个token (CLS/cis)
            return last_layer_embedding
    except Exception as e:
        print(f"Embedding error: {e}")
        return None


def calculate_embedding_and_save(client, tokenizer, device, txn, seq_id, seq):
    """计算嵌入并保存到 LMDB。"""
    embedding = get_esm_embedding(client, tokenizer, device, seq)
    if embedding is not None:
        txn.put(seq_id.encode(), embedding.tobytes())
    return embedding is not None


def estimate_remaining_time(start_time, processed_count, total_sequences, time_per_seq):  # 修改 total_remaining 为 total_sequences
    """估算剩余时间。"""
    avg_time = sum(time_per_seq) / len(time_per_seq) if time_per_seq else 0
    remaining_time = avg_time * (total_sequences - processed_count)  # 修改 total_remaining 为 total_sequences
    elapsed_time = time.time() - start_time
    return elapsed_time, remaining_time


def process_and_save_sequences(fasta_file_path, output_file_path, client, tokenizer, device, save_interval=10):
    """逐步处理序列，计算嵌入并流式保存,  **取消断点续传 & 数据完整性检查**"""
    sequences = read_fasta_sequences(fasta_file_path)
    total_sequences = len(sequences)  # 总序列数直接从 sequences 字典获取
    processed_count = 0  # 每次都从 0 开始计数
    start_time = time.time()
    time_per_seq = []

    # 检查输出 LMDB 数据库路径是否存在
    if os.path.exists(output_file_path):
        print(f"数据库 {output_file_path} 已存在。")
        while True:
            choice = input("您想 (1) 删除重建，还是 (2) 追加到现有数据库？ 请输入 1 或 2: ")
            if choice == '1':
                try:
                    if os.path.isdir(output_file_path): # 检查是否为目录
                        shutil.rmtree(output_file_path) # 删除目录及其内容
                    else:
                        os.remove(output_file_path) # 删除文件
                    print(f"已删除现有数据库 {output_file_path}，将重新创建。")
                    break # 删除重建后跳出循环，继续后续处理
                except Exception as e:
                    print(f"删除数据库时出错: {e}")
                    print("请检查是否有其他程序正在使用该数据库，或检查文件权限。")
                    return  # 发生错误，退出函数
            elif choice == '2':
                print(f"将追加到现有数据库 {output_file_path}。")
                break # 追加到数据库，跳出循环，继续后续处理
            else:
                print("输入无效，请输入 1 或 2。")

    print(f"总共 {total_sequences} 个序列, 将全部重新处理。")  # 更加明确的提示信息

    try:
        with lmdb.open(output_file_path, map_size=int(1.5 * 10**12)) as env:
            with env.begin(write=True) as txn:

                progress_bar = tqdm(sequences.items(), desc="Processing sequences", total=total_sequences)  # total 直接使用 total_sequences
                for seq_id, seq in progress_bar:
                    seq_start = time.time()
                    if calculate_embedding_and_save(client, tokenizer, device, txn, seq_id, seq):
                        seq_time = time.time() - seq_start
                        time_per_seq.append(seq_time)
                        if len(time_per_seq) > 10:
                            time_per_seq.pop(0)
                        processed_count += 1
                        elapsed_time, remaining_time = estimate_remaining_time(start_time, processed_count, total_sequences, time_per_seq)  # total_remaining 修改为 total_sequences
                        progress_bar.postfix = f"用时: {elapsed_time:.2f}s, 预计剩余: {remaining_time / 60:.2f} 分钟"

                        if processed_count % save_interval == 0:
                            torch.cuda.empty_cache()

    except KeyboardInterrupt:
        print("\n检测到手动中断，程序停止。")
        exit()

    print(f"\n所有序列已计算完成，嵌入已重新保存到 {output_file_path}")


if __name__ == "__main__":
    fasta_file_path = './dataset/S1/protein.fasta'  # 请替换为您的 FASTA 文件路径
    output_file_path = './emb/S1/esm600m_S1_all.NOcls_NOeos.lmdb'  # 请替换为您希望保存 LMDB 文件的路径
    process_and_save_sequences(fasta_file_path, output_file_path, client, tokenizer, device)