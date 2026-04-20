import os
import numpy as np
import argparse
from tqdm import tqdm
import time
import shutil
import torch
import threading
import queue
import multiprocessing as mp
from functools import partial
import contextlib
import gc
import warnings
import traceback
import sys
import json
import torch.multiprocessing as mp
import signal
import psutil
from concurrent.futures import ProcessPoolExecutor, as_completed
import tempfile
import pickle
# Capture warnings instead of printing them
warnings.filterwarnings("ignore")

# 全局变量用于信号处理
interrupted = False

def signal_handler(signum, frame):
    """处理Ctrl+C信号"""
    global interrupted
    print(f"\n接收到中断信号 {signum}，正在安全退出...")
    interrupted = True
    # 强制退出程序
    sys.exit(1)

# 注册信号处理器
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def check_interrupted():
    """检查是否被中断"""
    global interrupted
    return interrupted


def force_cleanup_memory():
    """强制清理内存"""
    try:
        # 强制垃圾回收
        gc.collect()
        
        # 清理PyTorch缓存
        if 'torch' in sys.modules:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            # 清理CPU缓存
            if hasattr(torch, 'clear_cache'):
                torch.clear_cache()
        
        # 清理numpy缓存（如果存在）
        if 'numpy' in sys.modules:
            import numpy as np
            # NumPy没有clear_cache方法，但可以尝试其他清理方式
            try:
                # 清理NumPy的内部缓存（如果存在）
                if hasattr(np, 'clear_cache'):
                    np.clear_cache()
                elif hasattr(np, '_no_nep50_warning'):
                    # 清理一些NumPy的内部状态
                    pass
            except:
                pass
        
        # 强制Python垃圾回收多次
        for _ in range(3):
            gc.collect()
            
    except Exception as e:
        print(f"内存清理时出错: {e}")


def write_results_to_json(successful_results, output_json):
    """将结果写入JSON文件"""
    try:
        # 准备JSON数据
        json_data = []
        
        for protein_name, sst_seq in tqdm(successful_results, desc="写入JSON文件"):
            if protein_name and sst_seq is not None:
                # 将SST序列转换为整数token字符串（用逗号分隔）
                # 确保token是整数，没有小数点
                integer_tokens = [int(round(token)) for token in sst_seq.tolist()]
                token_string = ",".join(map(str, integer_tokens))
                
                # 创建JSON条目
                entry = {
                    "protein_id": protein_name,
                    "embeddings": token_string
                }
                json_data.append(entry)
        
        # 写入JSON文件
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        
        print(f"成功保存SST序列到 {output_json}，共 {len(json_data)} 个蛋白质")
    
    except Exception as e:
        print(f"写入JSON文件时出错: {e}")
        traceback.print_exc()


def load_existing_json(output_json):
    """读取已有JSON文件，返回 (existing_data, existing_ids)"""
    if not os.path.exists(output_json):
        return [], set()
    try:
        with open(output_json, 'r', encoding='utf-8') as f:
            existing_data = json.load(f)
        existing_ids = {entry['protein_id'] for entry in existing_data if 'protein_id' in entry}
        print(f"已有JSON文件包含 {len(existing_ids)} 个蛋白质")
        return existing_data, existing_ids
    except Exception as e:
        print(f"读取已有JSON文件时出错: {e}，将视为空文件")
        return [], set()


def merge_and_write_json(new_results, output_json, existing_data=None):
    """将新结果与已有数据合并后写入JSON文件（按protein_id去重，新结果覆盖旧结果）"""
    try:
        # 将新结果转为JSON条目
        new_entries = []
        for protein_name, sst_seq in tqdm(new_results, desc="准备新结果"):
            if protein_name and sst_seq is not None:
                integer_tokens = [int(round(token)) for token in sst_seq.tolist()]
                token_string = ",".join(map(str, integer_tokens))
                new_entries.append({
                    "protein_id": protein_name,
                    "embeddings": token_string
                })

        # 合并：新结果覆盖旧结果
        new_ids = {entry['protein_id'] for entry in new_entries}
        if existing_data:
            # 保留旧数据中不在新结果里的条目
            kept_old = [entry for entry in existing_data if entry.get('protein_id') not in new_ids]
            merged = kept_old + new_entries
            print(f"合并: 保留旧数据 {len(kept_old)} 条 + 新增/更新 {len(new_entries)} 条 = 共 {len(merged)} 条")
        else:
            merged = new_entries

        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(merged, f, indent=2, ensure_ascii=False)

        print(f"成功保存SST序列到 {output_json}，共 {len(merged)} 个蛋白质")

    except Exception as e:
        print(f"合并写入JSON文件时出错: {e}")
        traceback.print_exc()


def get_optimal_parallel_settings():
    """获取最优的并行化设置"""
    # 获取系统信息
    cpu_count = psutil.cpu_count(logical=True)
    memory_gb = psutil.virtual_memory().total / (1024**3)
    
    # 根据系统资源动态调整并行化设置
    if memory_gb >= 32:  # 大内存系统
        num_processes = min(cpu_count, 8)  # 最多8个进程
        num_threads = min(cpu_count // num_processes, 16)  # 每个进程最多16个线程
        num_workers = min(cpu_count // 2, 4)  # DataLoader工作进程
    elif memory_gb >= 16:  # 中等内存系统
        num_processes = min(cpu_count // 2, 4)
        num_threads = min(cpu_count // num_processes, 8)
        num_workers = min(cpu_count // 4, 2)
    else:  # 小内存系统
        num_processes = min(cpu_count // 4, 2)
        num_threads = min(cpu_count // num_processes, 4)
        num_workers = 1
    
    return {
        'num_processes': num_processes,
        'num_threads': num_threads,
        'num_workers': num_workers,
        'cpu_count': cpu_count,
        'memory_gb': memory_gb
    }


def process_single_file_safe(args):
    """安全处理单个PDB文件并返回结果 - 适用于WSL环境"""
    pdb_path, structure_vocab_size, process_id, parallel_settings = args
    
    # 设置环境变量以确保每个进程都有独立的模型缓存路径
    model_cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_cache")
    os.environ['TORCH_HOME'] = model_cache_dir
    os.environ['HF_HOME'] = model_cache_dir
    os.environ['HF_CACHE_HOME'] = model_cache_dir
    os.environ['TRANSFORMERS_CACHE'] = os.path.join(model_cache_dir, "transformers")
    os.environ['DISABLE_TQDM'] = '1' # 在子进程中禁用tqdm
    
    # 优化并行计算设置 - 使用更保守的设置
    os.environ['OMP_NUM_THREADS'] = str(parallel_settings['num_threads'])
    os.environ['MKL_NUM_THREADS'] = str(parallel_settings['num_threads'])
    os.environ['OPENBLAS_NUM_THREADS'] = str(parallel_settings['num_threads'])
    os.environ['VECLIB_MAXIMUM_THREADS'] = str(parallel_settings['num_threads'])
    os.environ['NUMEXPR_NUM_THREADS'] = str(parallel_settings['num_threads'])
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # 禁用Hugging Face tokenizer并行性
    os.environ['PYTHONWARNINGS'] = 'ignore'  # 忽略Python警告
    os.environ['TF_NUM_INTEROP_THREADS'] = str(parallel_settings['num_threads'])
    os.environ['TF_NUM_INTRAOP_THREADS'] = str(parallel_settings['num_threads'])
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'  # 启用CUDA异步操作以提高性能
    
    try:
        # 将当前脚本的父目录添加到sys.path以导入prosst模块
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        # 标准库和PyTorch导入
        import torch
        import torch.multiprocessing
        
        # 设置PyTorch线程数 - 在导入任何其他库之前设置
        torch.set_num_threads(parallel_settings['num_threads'])
        # 注意：不要在子进程中设置interop线程，这会导致错误
        
        # 设置CUDA设备，强制使用第一个GPU（如果可用）
        if torch.cuda.is_available():
            torch.cuda.set_device(0) 
        
        # 在导入任何库之前设置共享策略为'file_system'
        torch.multiprocessing.set_sharing_strategy('file_system')
        
        # 设置多处理启动方法为'spawn'以避免fork问题
        try:
            torch.multiprocessing.set_start_method('spawn', force=True)
        except RuntimeError:
            # 如果已经设置，不要再次尝试设置
            pass
        
        # 导入SST预测器
        from prosst.structure.get_sst_seq import SSTPredictor
        
        # 通过重定向stdout和stderr到/dev/null来抑制不必要的输出
        with open(os.devnull, 'w') as devnull:
            with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                # 创建SST预测器实例 - 使用优化的并行设置
                predictor = SSTPredictor(
                    structure_vocab_size=structure_vocab_size,
                    num_processes=0,  # 在子进程中设为0，避免嵌套多进程
                    num_threads=parallel_settings['num_threads']
                )
                
                # 在torch.no_grad()上下文中执行预测以禁用梯度计算并减少内存消耗
                with torch.no_grad():
                    result = predictor.predict_from_pdb(pdb_path)
                
                # 检查预测结果是否有效
                if result and len(result) > 0 and not isinstance(result, str):
                    protein_data = result[0]
                    protein_name = protein_data['name'].split('.')[0]
                    # 将SST序列转换为整数数组，确保token是整数
                    sst_seq = np.array(protein_data[f'{structure_vocab_size}_sst_seq'], dtype=np.int32)
                    
                    # 明确删除不再需要的变量，并清除GPU缓存和Python垃圾
                    del predictor # 删除预测器实例
                    del result
                    gc.collect() # 强制垃圾回收
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache() # 清除GPU内存缓存
                    
                    return (protein_name, sst_seq, os.path.basename(pdb_path), None)  # 修复：添加None作为error_msg
                else:
                    # 如果结果无效，也要执行清理
                    del predictor
                    del result
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    return (None, None, os.path.basename(pdb_path), "无效结果")
        
    except Exception as e:
        # 捕获并打印处理过程中发生的错误
        error_msg = f"处理 {os.path.basename(pdb_path)} 时出错: {str(e)}"
        print(error_msg)
        traceback.print_exc() # 打印完整堆栈跟踪
        return (None, None, os.path.basename(pdb_path), str(e))


def process_pdb_files_multiprocess_optimized(input_dir, output_json, structure_vocab_size=2048, max_workers=None, append=True, overwrite_last=0):
    """使用优化的多进程处理PDB文件并创建JSON格式的SST token文件
    
    Args:
        append: 默认True，自动追加模式（比较差异，仅处理新增蛋白）
        overwrite_last: 强制重处理排序后最后N个PDB文件（默认0，不重处理）
    """
    
    # 获取最优并行化设置
    parallel_settings = get_optimal_parallel_settings()
    print(f"系统信息: CPU核心数={parallel_settings['cpu_count']}, 内存={parallel_settings['memory_gb']:.1f}GB")
    
    # 如果没有指定max_workers，使用优化设置
    if max_workers is None:
        max_workers = parallel_settings['num_processes']
    
    print(f"多进程优化设置: 进程数={max_workers}, 线程数={parallel_settings['num_threads']}")
    
    # 检查输入目录
    if not os.path.exists(input_dir):
        print(f"错误: 输入目录 {input_dir} 不存在")
        return
    
    # 获取所有PDB文件
    pdb_files_basenames = [f for f in os.listdir(input_dir) if f.endswith('.pdb')]
    total_initial_files = len(pdb_files_basenames)
    
    if total_initial_files == 0:
        print(f"警告: 在 {input_dir} 中未找到PDB文件")
        return
    
    print(f"在输入目录中找到 {total_initial_files} 个PDB文件。")
    
    full_pdb_paths = [os.path.join(input_dir, f) for f in pdb_files_basenames]
    pdb_paths_to_process = set()
    existing_data = []  # 用于追加模式下保留旧数据

    if append and os.path.exists(output_json):
        # 追加模式：读取已有JSON，计算差异
        existing_data, existing_ids = load_existing_json(output_json)
        pdb_ids = {os.path.basename(p).split('.')[0] for p in full_pdb_paths}
        new_ids = pdb_ids - existing_ids
        
        # 添加不存在于JSON中的PDB
        for pdb_path in full_pdb_paths:
            protein_name = os.path.basename(pdb_path).split('.')[0]
            if protein_name in new_ids:
                pdb_paths_to_process.add(pdb_path)
        
        # 如有需要，强制重处理排序后最后N个PDB文件
        if overwrite_last > 0:
            sorted_full_pdb_paths = sorted(full_pdb_paths)
            for pdb_path in sorted_full_pdb_paths[max(0, len(sorted_full_pdb_paths) - overwrite_last):]:
                protein_name = os.path.basename(pdb_path).split('.')[0]
                if pdb_path not in pdb_paths_to_process:
                    pdb_paths_to_process.add(pdb_path)
                    print(f"包含 {protein_name}.pdb 进行重新处理（最后 {overwrite_last} 个文件的一部分）。")
        
        print(f"追加模式: 已有 {len(existing_ids)} 个蛋白, 需新增 {len(new_ids)} 个, 共 {total_initial_files} 个PDB文件")
        
        if not pdb_paths_to_process:
            print("所有蛋白已处理完毕，无需追加。")
            return
    elif not append and os.path.exists(output_json):
        # 重建模式：删除已有文件
        try:
            os.remove(output_json)
            print(f"重建模式: 已删除现有文件 {output_json}，将重新创建。")
        except Exception as e:
            print(f"删除文件时出错: {e}")
            return
        for p in full_pdb_paths:
            pdb_paths_to_process.add(p)
        existing_data = []
    else:
        # JSON文件不存在，处理所有文件
        for p in full_pdb_paths:
            pdb_paths_to_process.add(p)
        existing_data = []
    
    # 将集合转换为有序列表
    pdb_paths_to_process_list = sorted(list(pdb_paths_to_process))
    total_files_to_process = len(pdb_paths_to_process_list)
    
    if total_files_to_process == 0:
        print("在JSON检查和覆盖策略考虑后，没有PDB文件需要处理。退出。")
        return
    
    print(f"在检查JSON和覆盖策略后，找到 {total_files_to_process} 个PDB文件需要处理。")
    
    # 设置模型缓存目录
    model_cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_cache")
    os.makedirs(model_cache_dir, exist_ok=True)
    print(f"模型将缓存在: {model_cache_dir}")
    
    # 设置环境变量以指定模型缓存目录并优化并行计算
    os.environ['TORCH_HOME'] = model_cache_dir
    os.environ['HF_HOME'] = model_cache_dir
    os.environ['HF_CACHE_HOME'] = model_cache_dir
    os.environ['TRANSFORMERS_CACHE'] = os.path.join(model_cache_dir, "transformers")
    os.environ['DISABLE_TQDM'] = '1'
    
    # 优化并行计算设置
    os.environ['OMP_NUM_THREADS'] = str(parallel_settings['num_threads'])
    os.environ['MKL_NUM_THREADS'] = str(parallel_settings['num_threads'])
    os.environ['OPENBLAS_NUM_THREADS'] = str(parallel_settings['num_threads'])
    os.environ['VECLIB_MAXIMUM_THREADS'] = str(parallel_settings['num_threads'])
    os.environ['NUMEXPR_NUM_THREADS'] = str(parallel_settings['num_threads'])
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['PYTHONWARNINGS'] = 'ignore'
    os.environ['TF_NUM_INTEROP_THREADS'] = str(parallel_settings['num_threads'])
    os.environ['TF_NUM_INTRAOP_THREADS'] = str(parallel_settings['num_threads'])
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'  # 启用CUDA异步操作

    # 跟踪处理进度的变量
    global_start_time = time.time()
    total_successful = 0
    total_errors = 0
    all_successful_results = []  # 存储所有成功的结果
    
    try:
        # 准备任务参数
        task_args = []
        for i, pdb_path in enumerate(pdb_paths_to_process_list):
            task_args.append((pdb_path, structure_vocab_size, i, parallel_settings))
        
        print(f"开始多进程处理，使用 {max_workers} 个进程...")
        
        # 使用ProcessPoolExecutor进行多进程处理
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_file = {
                executor.submit(process_single_file_safe, args): args[0] 
                for args in task_args
            }
            
            # 使用tqdm显示进度
            with tqdm(total=total_files_to_process, desc="处理PDB文件") as pbar:
                for future in as_completed(future_to_file):
                    pdb_path = future_to_file[future]
                    
                    # 检查是否被中断
                    if check_interrupted():
                        print("\n检测到中断信号，正在安全退出...")
                        break
                    
                    try:
                        result = future.result()
                        protein_name, sst_seq, filename, error_msg = result
                        
                        if protein_name and sst_seq is not None:
                            all_successful_results.append((protein_name, sst_seq))
                            total_successful += 1
                        else:
                            total_errors += 1
                            if error_msg:
                                print(f"处理失败 {filename}: {error_msg}")
                        
                        # 更新进度条
                        pbar.update(1)
                        pbar.set_description(f"进度: {total_successful + total_errors}/{total_files_to_process} "
                                            f"| 成功: {total_successful} | 错误: {total_errors}")
                        
                    except Exception as e:
                        total_errors += 1
                        print(f"处理 {os.path.basename(pdb_path)} 时出现异常: {e}")
                        pbar.update(1)
        
        # 处理完成或被中断
        if check_interrupted():
            print("\n程序被用户中断。")
        else:
            total_time = time.time() - global_start_time
            print(f"\n所有处理完成！用时: {total_time:.2f}秒")
            print(f"总成功: {total_successful}, 总失败: {total_errors}")
        
        # 写入JSON文件（即使被中断也保存已处理的结果）
        if all_successful_results:
            print("写入JSON文件...")
            merge_and_write_json(all_successful_results, output_json, existing_data)
    
    except KeyboardInterrupt:
        print("\n检测到手动中断，程序已停止。")
    except Exception as e:
        print(f"发生错误: {e}")
        traceback.print_exc()
    finally:
        # 最终清理资源
        print("执行最终清理...")
        try:
            force_cleanup_memory()
        except Exception as e:
            print(f"最终清理时出错: {e}")


def process_pdb_files_hybrid_optimized(input_dir, output_json, structure_vocab_size=2048, max_workers=None, batch_size=50, append=True, overwrite_last=0):
    """使用混合优化策略处理PDB文件 - 结合多进程和多线程的优势"""
    
    # 获取最优并行化设置
    parallel_settings = get_optimal_parallel_settings()
    print(f"系统信息: CPU核心数={parallel_settings['cpu_count']}, 内存={parallel_settings['memory_gb']:.1f}GB")
    
    # 如果没有指定max_workers，使用优化设置
    if max_workers is None:
        max_workers = min(parallel_settings['num_processes'], 4)  # 限制最大进程数
    
    print(f"混合优化设置: 进程数={max_workers}, 线程数={parallel_settings['num_threads']}, 批处理大小={batch_size}")
    
    # 检查输入目录
    if not os.path.exists(input_dir):
        print(f"错误: 输入目录 {input_dir} 不存在")
        return
    
    # 获取所有PDB文件
    pdb_files_basenames = [f for f in os.listdir(input_dir) if f.endswith('.pdb')]
    total_initial_files = len(pdb_files_basenames)
    
    if total_initial_files == 0:
        print(f"警告: 在 {input_dir} 中未找到PDB文件")
        return
    
    print(f"在输入目录中找到 {total_initial_files} 个PDB文件。")
    
    full_pdb_paths = [os.path.join(input_dir, f) for f in pdb_files_basenames]
    pdb_paths_to_process = set()
    existing_data = []  # 用于追加模式下保留旧数据

    if append and os.path.exists(output_json):
        # 追加模式：读取已有JSON，计算差异
        existing_data, existing_ids = load_existing_json(output_json)
        pdb_ids = {os.path.basename(p).split('.')[0] for p in full_pdb_paths}
        new_ids = pdb_ids - existing_ids
        
        # 添加不存在于JSON中的PDB
        for pdb_path in full_pdb_paths:
            protein_name = os.path.basename(pdb_path).split('.')[0]
            if protein_name in new_ids:
                pdb_paths_to_process.add(pdb_path)
        
        # 如有需要，强制重处理排序后最后N个PDB文件
        if overwrite_last > 0:
            sorted_full_pdb_paths = sorted(full_pdb_paths)
            for pdb_path in sorted_full_pdb_paths[max(0, len(sorted_full_pdb_paths) - overwrite_last):]:
                protein_name = os.path.basename(pdb_path).split('.')[0]
                if pdb_path not in pdb_paths_to_process:
                    pdb_paths_to_process.add(pdb_path)
                    print(f"包含 {protein_name}.pdb 进行重新处理（最后 {overwrite_last} 个文件的一部分）。")
        
        print(f"追加模式: 已有 {len(existing_ids)} 个蛋白, 需新增 {len(new_ids)} 个, 共 {total_initial_files} 个PDB文件")
        
        if not pdb_paths_to_process:
            print("所有蛋白已处理完毕，无需追加。")
            return
    elif not append and os.path.exists(output_json):
        # 重建模式：删除已有文件
        try:
            os.remove(output_json)
            print(f"重建模式: 已删除现有文件 {output_json}，将重新创建。")
        except Exception as e:
            print(f"删除文件时出错: {e}")
            return
        for p in full_pdb_paths:
            pdb_paths_to_process.add(p)
        existing_data = []
    else:
        # JSON文件不存在，处理所有文件
        for p in full_pdb_paths:
            pdb_paths_to_process.add(p)
        existing_data = []
    
    # 将集合转换为有序列表
    pdb_paths_to_process_list = sorted(list(pdb_paths_to_process))
    total_files_to_process = len(pdb_paths_to_process_list)
    
    if total_files_to_process == 0:
        print("在JSON检查和覆盖策略考虑后，没有PDB文件需要处理。退出。")
        return
    
    print(f"在检查JSON和覆盖策略后，找到 {total_files_to_process} 个PDB文件需要处理。")
    
    # 设置模型缓存目录
    model_cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_cache")
    os.makedirs(model_cache_dir, exist_ok=True)
    print(f"模型将缓存在: {model_cache_dir}")
    
    # 设置环境变量以指定模型缓存目录并优化并行计算
    os.environ['TORCH_HOME'] = model_cache_dir
    os.environ['HF_HOME'] = model_cache_dir
    os.environ['HF_CACHE_HOME'] = model_cache_dir
    os.environ['TRANSFORMERS_CACHE'] = os.path.join(model_cache_dir, "transformers")
    os.environ['DISABLE_TQDM'] = '1'
    
    # 优化并行计算设置
    os.environ['OMP_NUM_THREADS'] = str(parallel_settings['num_threads'])
    os.environ['MKL_NUM_THREADS'] = str(parallel_settings['num_threads'])
    os.environ['OPENBLAS_NUM_THREADS'] = str(parallel_settings['num_threads'])
    os.environ['VECLIB_MAXIMUM_THREADS'] = str(parallel_settings['num_threads'])
    os.environ['NUMEXPR_NUM_THREADS'] = str(parallel_settings['num_threads'])
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['PYTHONWARNINGS'] = 'ignore'
    os.environ['TF_NUM_INTEROP_THREADS'] = str(parallel_settings['num_threads'])
    os.environ['TF_NUM_INTRAOP_THREADS'] = str(parallel_settings['num_threads'])
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'  # 启用CUDA异步操作

    # 将文件分成批次
    file_batches = [pdb_paths_to_process_list[i:i+batch_size] 
                   for i in range(0, len(pdb_paths_to_process_list), batch_size)]
    print(f"将 {total_files_to_process} 个文件分成 {len(file_batches)} 批，每批最多 {batch_size} 个文件。")
    
    # 跟踪处理进度的变量
    global_start_time = time.time()
    total_successful = 0
    total_errors = 0
    all_successful_results = []  # 存储所有成功的结果
    
    try:
        # 处理每一批文件
        for batch_idx, file_batch in enumerate(file_batches):
            # 检查是否被中断
            if check_interrupted():
                print("\n检测到中断信号，正在安全退出...")
                break
                
            print(f"\n--- 处理第 {batch_idx+1}/{len(file_batches)} 批 ({len(file_batch)} 个文件) ---")
            
            # 显示当前内存使用情况
            try:
                process = psutil.Process()
                memory_info = process.memory_info()
                print(f"当前内存使用: {memory_info.rss / 1024 / 1024:.1f} MB")
            except:
                pass
            
            # 准备当前批次的任务参数
            batch_task_args = []
            for i, pdb_path in enumerate(file_batch):
                batch_task_args.append((pdb_path, structure_vocab_size, i, parallel_settings))
            
            batch_successful_results = []
            batch_errors = 0
            
            # 使用ProcessPoolExecutor处理当前批次
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                # 提交当前批次的所有任务
                future_to_file = {
                    executor.submit(process_single_file_safe, args): args[0] 
                    for args in batch_task_args
                }
                
                # 使用tqdm显示当前批次的进度
                with tqdm(total=len(file_batch), desc=f"处理第 {batch_idx+1} 批") as pbar:
                    for future in as_completed(future_to_file):
                        pdb_path = future_to_file[future]
                        
                        # 检查是否被中断
                        if check_interrupted():
                            print("\n检测到中断信号，停止处理当前批次...")
                            break
                        
                        try:
                            result = future.result()
                            protein_name, sst_seq, filename, error_msg = result
                            
                            if protein_name and sst_seq is not None:
                                batch_successful_results.append((protein_name, sst_seq))
                                total_successful += 1
                            else:
                                batch_errors += 1
                                total_errors += 1
                                if error_msg:
                                    print(f"处理失败 {filename}: {error_msg}")
                            
                            # 更新进度条
                            pbar.update(1)
                            pbar.set_description(f"批次 {batch_idx+1}: {total_successful + total_errors}/{total_files_to_process} "
                                                f"| 成功: {total_successful} | 错误: {total_errors}")
                            
                        except Exception as e:
                            batch_errors += 1
                            total_errors += 1
                            print(f"处理 {os.path.basename(pdb_path)} 时出现异常: {e}")
                            pbar.update(1)
            
            # 将当前批次结果添加到总结果中
            all_successful_results.extend(batch_successful_results)
            
            # 显示批次处理统计
            print(f"第 {batch_idx+1} 批完成: 处理了 {len(file_batch)} 个文件, "
                  f"成功: {len(file_batch)-batch_errors}, 错误: {batch_errors}")
            
            # 清理当前批次的内存
            print(f"清理第 {batch_idx+1} 批的内存...")
            try:
                del batch_successful_results
                force_cleanup_memory()
                
                # 显示清理后的内存使用情况
                try:
                    process = psutil.Process()
                    memory_info = process.memory_info()
                    print(f"清理后内存使用: {memory_info.rss / 1024 / 1024:.1f} MB")
                except:
                    pass
                    
            except Exception as e:
                print(f"清理内存时出错: {e}")
            
            # 检查是否被中断
            if check_interrupted():
                print("\n检测到中断信号，停止处理...")
                break
            
            # 短暂暂停，确保资源释放
            time.sleep(0.5)
        
        # 处理完成或被中断
        if check_interrupted():
            print("\n程序被用户中断。")
        else:
            total_time = time.time() - global_start_time
            print(f"\n所有处理完成！用时: {total_time:.2f}秒")
            print(f"总成功: {total_successful}, 总失败: {total_errors}")
        
        # 写入JSON文件（即使被中断也保存已处理的结果）
        if all_successful_results:
            print("写入JSON文件...")
            merge_and_write_json(all_successful_results, output_json, existing_data)
    
    except KeyboardInterrupt:
        print("\n检测到手动中断，程序已停止。")
    except Exception as e:
        print(f"发生错误: {e}")
        traceback.print_exc()
    finally:
        # 最终清理资源
        print("执行最终清理...")
        try:
            force_cleanup_memory()
        except Exception as e:
            print(f"最终清理时出错: {e}")


def main():
    parser = argparse.ArgumentParser(description='将PDB文件转换为SST序列并存储为JSON格式（多进程优化版本）')
    parser.add_argument('--input', '-i', required=True, help='包含PDB文件的输入目录')
    parser.add_argument('--output', '-o', required=True, help='输出JSON文件路径')
    parser.add_argument('--vocab_size', '-v', type=int, default=2048, 
                        choices=[20, 128, 512, 1024, 2048, 4096],
                        help='SST结构词汇表大小 (20, 128, 512, 1024, 2048, 4096)')
    parser.add_argument('--max_workers', '-w', type=int, default=None,
                        help='最大并行进程数 (默认: 根据系统资源自动调整)')
    parser.add_argument('--mode', '-m', choices=['multiprocess', 'hybrid'], default='multiprocess',
                        help='处理模式: multiprocess(纯多进程) 或 hybrid(混合模式)')
    parser.add_argument('--batch_size', '-b', type=int, default=50,
                        help='混合模式下的批处理大小 (默认: 50)')
    parser.add_argument('--append', '-a', action='store_true', default=True,
                        help='追加模式: 自动比较差异，仅处理新增蛋白 (默认: True)')
    parser.add_argument('--no-append', dest='append', action='store_false',
                        help='重建模式: 删除已有文件并重新处理所有PDB文件')
    parser.add_argument('--overwrite-last', type=int, default=0,
                        help='强制重处理排序后最后N个PDB文件 (默认: 0)')
    
    args = parser.parse_args()
    
    print("使用多进程优化版本处理PDB文件")
    print("按 Ctrl+C 可以安全中断程序")
    if args.append:
        print("模式: 追加 (自动比较差异，仅处理新增蛋白)")
    else:
        print("模式: 重建 (删除已有文件，重新处理所有PDB文件)")
    
    if args.mode == 'hybrid':
        process_pdb_files_hybrid_optimized(args.input, args.output, args.vocab_size, args.max_workers, args.batch_size, args.append, args.overwrite_last)
    else:
        process_pdb_files_multiprocess_optimized(args.input, args.output, args.vocab_size, args.max_workers, args.append, args.overwrite_last)

if __name__ == "__main__":
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        # 如果已经设置，不要再次尝试设置
        pass
    main() 