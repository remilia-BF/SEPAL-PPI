import os
import math
import pickle
import hashlib
import tempfile
import subprocess
import multiprocessing as mp
from pathlib import Path
from collections import defaultdict
from typing import Optional

from tqdm import tqdm
from functools import partial

class HMMCacheManager:
    def __init__(self, hmm_data_dir: str = "./hmm_data", hmm_db_path: str = "./data/weights/Pfam-A.hmm"):
        self.hmm_data_dir = Path(hmm_data_dir)
        self.hmm_db_path = hmm_db_path
        # 递归创建目录，避免父目录不存在时报错
        self.hmm_data_dir.mkdir(parents=True, exist_ok=True)

    def get_fasta_hash(self, fasta_path):
        with open(fasta_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()[:16]

    def get_cache_path(self, fasta_path):
        fasta_hash = self.get_fasta_hash(fasta_path)
        cache_filename = f"hmm_cache_{fasta_hash}.pkl"
        return self.hmm_data_dir / cache_filename

    def cache_exists(self, fasta_path):
        cache_path = self.get_cache_path(fasta_path)
        return cache_path.exists()

    def load_cache(self, fasta_path):
        cache_path = self.get_cache_path(fasta_path)
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"加载HMM缓存失败: {e}")
                return None
        return None

    def save_cache(self, fasta_path, hmm_results):
        cache_path = self.get_cache_path(fasta_path)
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(hmm_results, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"HMM缓存已保存: {cache_path}")
        except Exception as e:
            print(f"保存HMM缓存失败: {e}")

    def create_hmm_database(self, fasta_path, batch_size=200, num_processes=None):
        print(f"开始为 {fasta_path} 创建HMM数据库...")
        if self.cache_exists(fasta_path):
            print("HMM缓存已存在，跳过创建")
            return self.load_cache(fasta_path)
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
            return {}
        print(f"共读取 {len(sequences)} 个蛋白质序列")
        if num_processes is None:
            num_processes = min(mp.cpu_count(), 32)
        print(f"使用 {num_processes} 个进程进行并行HMM分析")
        seq_ids = list(sequences.keys())
        batches = []
        for i in range(0, len(seq_ids), batch_size):
            batch_ids = seq_ids[i:i + batch_size]
            batch_sequences = [(seq_id, sequences[seq_id]) for seq_id in batch_ids]
            batches.append((i // batch_size, batch_sequences))
        print(f"创建了 {len(batches)} 个批次，每批次 {batch_size} 个序列")
        hmm_results = {}
        try:
            with mp.Pool(processes=num_processes) as pool:
                # 使用partial而非lambda，避免多进程pickle错误
                process_func = partial(self._process_batch_hmm_wrapper, hmm_db_path=self.hmm_db_path)
                with tqdm(total=len(batches), desc="HMM并行分析进度", unit="batch") as pbar:
                    results = []
                    for batch in batches:
                        result = pool.apply_async(process_func, (batch,))
                        results.append(result)
                    for result in results:
                        try:
                            batch_idx, batch_results = result.get(timeout=600)
                            hmm_results.update(batch_results)
                            pbar.update(1)
                            pbar.set_postfix({
                                'processed': f"{len(hmm_results)}/{len(sequences)}",
                                'batch': f"{batch_idx+1}/{len(batches)}"
                            })
                            if (batch_idx + 1) % 10 == 0:
                                self._save_intermediate_cache(fasta_path, hmm_results)
                        except Exception as e:
                            # 避免未定义变量引用，输出真实错误信息
                            print(f"批次处理失败: {e}")
                            continue
        except Exception as e:
            print(f"并行处理过程中出现错误: {e}")
            print("回退到串行处理...")
            return self._create_hmm_database_serial(sequences, batch_size, fasta_path)
        self.save_cache(fasta_path, hmm_results)
        self._cleanup_intermediate_cache(fasta_path)
        print(f"HMM数据库创建完成，共处理 {len(hmm_results)} 个蛋白质")
        return hmm_results

    def _process_batch_hmm_wrapper(self, batch_data, hmm_db_path):
        batch_idx, batch_sequences = batch_data
        batch_results = self._process_batch_hmm_parallel(batch_sequences, hmm_db_path)
        return batch_idx, batch_results

    def _process_batch_hmm_parallel(self, batch_sequences, hmm_db_path):
        batch_results = {}
        with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as batch_fasta:
            for seq_id, sequence in batch_sequences:
                batch_fasta.write(f">{seq_id}\n{sequence}\n")
            batch_fasta_path = batch_fasta.name
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.out', delete=False) as out_file:
                out_path = out_file.name
            cmd = [
                'hmmscan',
                '--domtblout', out_path,
                '--noali',
                '--cut_ga',
                '--cpu', '1',
                '--notextw',
                hmm_db_path,
                batch_fasta_path
            ]
            try:
                env = os.environ.copy()
                env['HMMER_NCPU'] = '1'
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=900,
                    env=env
                )
                if result.returncode == 0:
                    batch_results = self._parse_batch_hmmscan_output(out_path)
                else:
                    if result.stderr:
                        print(f"HMMSCAN警告 (PID {os.getpid()}): {result.stderr[:200]}")
            except subprocess.TimeoutExpired:
                print(f"HMMSCAN批次处理超时 (PID {os.getpid()})，跳过当前批次")
            except Exception as e:
                print(f"HMMSCAN执行错误 (PID {os.getpid()}): {e}")
            try:
                os.unlink(out_path)
            except:
                pass
        except Exception as e:
            print(f"批次HMM处理错误 (PID {os.getpid()}): {e}")
        finally:
            try:
                os.unlink(batch_fasta_path)
            except:
                pass
        for seq_id, _ in batch_sequences:
            if seq_id not in batch_results:
                batch_results[seq_id] = []
        return batch_results

    def _save_intermediate_cache(self, fasta_path, hmm_results):
        temp_cache_path = self.get_cache_path(fasta_path).with_suffix('.tmp')
        try:
            with open(temp_cache_path, 'wb') as f:
                pickle.dump(hmm_results, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print(f"保存临时缓存失败: {e}")

    def _cleanup_intermediate_cache(self, fasta_path):
        temp_cache_path = self.get_cache_path(fasta_path).with_suffix('.tmp')
        if temp_cache_path.exists():
            try:
                temp_cache_path.unlink()
            except:
                pass

    def _create_hmm_database_serial(self, sequences, batch_size, fasta_path):
        print("使用串行方法处理HMM分析...")
        hmm_results = {}
        seq_ids = list(sequences.keys())
        progress_bar = tqdm(range(0, len(seq_ids), batch_size), desc="HMM串行分析进度", unit="batch")
        for i in progress_bar:
            batch_ids = seq_ids[i:i + batch_size]
            batch_sequences = [(seq_id, sequences[seq_id]) for seq_id in batch_ids]
            batch_results = self._process_batch_hmm(batch_sequences)
            hmm_results.update(batch_results)
            progress_bar.set_postfix({
                'processed': f"{len(hmm_results)}/{len(sequences)}",
                'current_batch': len(batch_results)
            })
            if (i // batch_size + 1) % 10 == 0:
                self._save_intermediate_cache(fasta_path, hmm_results)
        progress_bar.close()
        return hmm_results

    def _process_batch_hmm(self, batch_sequences):
        batch_results = {}
        with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as batch_fasta:
            for seq_id, sequence in batch_sequences:
                batch_fasta.write(f">{seq_id}\n{sequence}\n")
            batch_fasta_path = batch_fasta.name
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.out', delete=False) as out_file:
                out_path = out_file.name
            cmd = [
                'hmmscan',
                '--domtblout', out_path,
                '--noali',
                '--cut_ga',
                '--cpu', '1',
                self.hmm_db_path,
                batch_fasta_path
            ]
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                if result.returncode == 0:
                    batch_results = self._parse_batch_hmmscan_output(out_path)
                else:
                    print(f"HMMSCAN批次处理警告: {result.stderr}")
            except subprocess.TimeoutExpired:
                print(f"HMMSCAN批次处理超时，跳过当前批次")
            os.unlink(out_path)
        except Exception as e:
            print(f"批次HMM处理错误: {e}")
        finally:
            if os.path.exists(batch_fasta_path):
                os.unlink(batch_fasta_path)
        for seq_id, _ in batch_sequences:
            if seq_id not in batch_results:
                batch_results[seq_id] = []
        return batch_results

    def _parse_batch_hmmscan_output(self, output_path):
        results = defaultdict(list)
        try:
            with open(output_path, 'r') as f:
                for line in f:
                    if line.startswith('#') or not line.strip():
                        continue
                    fields = line.split()
                    if len(fields) < 23:
                        continue
                    domain_name = fields[0]
                    seq_id = fields[3]
                    target_start = int(fields[17])
                    target_end = int(fields[18])
                    evalue = float(fields[12])
                    score = float(fields[13])
                    confidence = min(1.0, max(0.1, -math.log10(max(evalue, 1e-100)) / 50.0))
                    domain_info = (target_start-1, target_end-1, domain_name, evalue, score, confidence)
                    results[seq_id].append(domain_info)
        except Exception as e:
            print(f"解析批次HMMSCAN输出错误: {e}")
        return dict(results)

    def get_domains(self, seq_id, hmm_cache=None):
        if hmm_cache and seq_id in hmm_cache:
            return hmm_cache[seq_id]
        return []

    def load_or_create_cache(self, fasta_path=None, batch_size: int = 200, num_processes: Optional[int] = None):
        """加载或创建HMM缓存。如果没有提供fasta_path，则返回一个空的缓存。
        可通过 batch_size 与 num_processes 控制 hmmscan 构建参数。
        """
        if fasta_path is None:
            print("未提供FASTA文件路径，将返回空缓存")
            return {}
        
        # 尝试加载缓存
        cache = self.load_cache(fasta_path)
        if cache is not None:
            #print(f"已从缓存加载HMM数据，包含 {len(cache)} 个条目")
            return cache
        
        # 如果缓存不存在，则创建
        print(f"未找到HMM缓存，将为 {fasta_path} 创建新的缓存 (batch_size={batch_size}, num_processes={num_processes})")
        return self.create_hmm_database(fasta_path, batch_size=batch_size, num_processes=num_processes)
