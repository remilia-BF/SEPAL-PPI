import json
import csv
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any, Set
import logging


class HTMLGenerator:
    """生成SEPAL-PPI预测结果的HTML界面"""
    
    def __init__(self, output_dir: str, config_path: Optional[str] = None, logger=None):
        self.output_dir = Path(output_dir)
        self.config_path = config_path
        # 修复：正确处理不同类型的logger对象
        if logger is None:
            self.logger = logging.getLogger(__name__)
        elif hasattr(logger, 'logger'):
            # 如果是SEPALLogger对象，使用其内部的logger
            self.logger = logger.logger
        else:
            # 如果是普通的logging.Logger对象
            self.logger = logger
        
        # 从配置文件中加载注意力头数
        # 直接使用SEPAL模型的注意力头数：2
        self.attention_heads = 2
        
        # 复制预测使用的FASTA序列到结果目录
        self._copy_prediction_fasta_files()
    
    def _load_attention_heads_from_config(self) -> int:
        """从配置文件中加载注意力头数"""
        if not self.config_path:
            # 尝试自动找到配置文件
            config_file = self._find_config_file()
            if config_file:
                self.config_path = config_file
            else:
                self.logger.warning("未找到配置文件，使用默认注意力头数: 2")
                return 2
        
        try:
            import yaml
            # 修复：确保config_path是字符串类型
            config_path_str = str(self.config_path) if isinstance(self.config_path, Path) else self.config_path
            with open(config_path_str, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # 尝试从不同的可能位置获取注意力头数
            attention_heads = None
            
            # 尝试从不同的可能位置获取注意力头数
            attention_heads = None
            
            # 方法1: 检查model.model_config.pooling.attention_num (resolved_config.yaml的结构)
            if 'model' in config and 'model_config' in config['model'] and 'pooling' in config['model']['model_config']:
                pooling_config = config['model']['model_config']['pooling']
                attention_heads = pooling_config.get('attention_num')
                self.logger.debug(f"在model.model_config.pooling中找到attention_num: {attention_heads}")
            
            # 方法2: 检查model.pooling.attention_num (普通配置文件结构)
            if attention_heads is None and 'model' in config and 'pooling' in config['model']:
                pooling_config = config['model']['pooling']
                attention_heads = pooling_config.get('attention_num')
                self.logger.debug(f"在model.pooling中找到attention_num: {attention_heads}")
            
            # 方法3: 检查preprocessing配置中的transformer heads
            if attention_heads is None and 'model' in config and 'model_config' in config['model'] and 'preprocessing' in config['model']['model_config']:
                preprocessing_config = config['model']['model_config']['preprocessing']
                attention_heads = preprocessing_config.get('transformer', {}).get('num_heads')
                if attention_heads is None:
                    attention_heads = preprocessing_config.get('transformer_heads')
                self.logger.debug(f"在preprocessing中找到num_heads: {attention_heads}")
            
            # 方法4: 检查顶级配置
            if attention_heads is None:
                attention_heads = config.get('attention_heads')
                if attention_heads is None:
                    attention_heads = config.get('transformer', {}).get('num_heads')
                self.logger.debug(f"在顶级配置中找到attention_heads: {attention_heads}")
            
            if attention_heads is not None:
                self.logger.info(f"从配置文件中加载注意力头数: {attention_heads}")
                return int(attention_heads)
            else:
                self.logger.warning("配置文件中未找到注意力头数，使用默认值: 2")
                return 2
                
        except Exception as e:
            self.logger.warning(f"加载配置文件失败: {e}，使用默认注意力头数: 2")
            return 2
    
    def _get_config_paths(self) -> Dict[str, str]:
        """从配置文件中获取数据路径"""
        if not self.config_path:
            # 尝试自动找到配置文件
            config_file = self._find_config_file()
            if config_file:
                self.config_path = config_file
        
        try:
            import yaml
            # 修复：确保config_path是字符串类型
            config_path_str = str(self.config_path) if isinstance(self.config_path, Path) else self.config_path
            with open(config_path_str, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # 首先检查是否是自定义预测配置文件格式（直接包含test_protein_fasta字段）
            if 'test_protein_fasta' in config:
                self.logger.debug("检测到自定义预测配置文件格式")
                return {
                    'fasta_file': config.get('test_protein_fasta', 'dataset/S1/protein.fasta'),
                    'test_id_file': config.get('test_id_file', 'predict/S1/predict_id.csv')
                }
            
            # 其次检查是否是Hydra配置格式（包含defaults字段）
            defaults = config.get('defaults', [])
            data_config = None
            
            # 查找data配置
            for default in defaults:
                if isinstance(default, dict) and 'data' in default:
                    data_name = default['data']
                    # 修复：确保config_path是字符串类型
                    config_path_str = str(self.config_path) if isinstance(self.config_path, Path) else self.config_path
                    data_config_path = Path(config_path_str).parent / 'data' / f'{data_name}.yaml'
                    if data_config_path.exists():
                        with open(data_config_path, 'r', encoding='utf-8') as f:
                            data_config = yaml.safe_load(f)
                        break
                elif isinstance(default, str) and default.startswith('data:'):
                    data_name = default.split(':')[1].strip()
                    # 修复：确保config_path是字符串类型
                    config_path_str = str(self.config_path) if isinstance(self.config_path, Path) else self.config_path
                    data_config_path = Path(config_path_str).parent / 'data' / f'{data_name}.yaml'
                    if data_config_path.exists():
                        with open(data_config_path, 'r', encoding='utf-8') as f:
                            data_config = yaml.safe_load(f)
                        break
            
            if data_config:
                return {
                    'fasta_file': data_config.get('fasta_file', 'dataset/S1/protein.fasta'),
                    'test_id_file': 'predict/S1/predict_id.csv'  # 默认预测ID文件路径
                }
            else:
                self.logger.warning("未找到data配置，使用默认路径")
                return {
                    'fasta_file': 'dataset/S1/protein.fasta',
                    'test_id_file': 'predict/S1/predict_id.csv'
                }
                
        except Exception as e:
            self.logger.warning(f"读取配置文件失败: {e}，使用默认路径")
            return {
                'fasta_file': 'dataset/S1/protein.fasta',
                'test_id_file': 'predict/S1/predict_id.csv'
            }
    
    def _extract_predicted_protein_ids(self, test_id_file: str) -> Set[str]:
        """从预测ID文件中提取所有蛋白质ID"""
        protein_ids = set()
        
        try:
            test_id_path = Path(test_id_file)
            if not test_id_path.exists():
                self.logger.warning(f"预测ID文件不存在: {test_id_file}")
                return protein_ids
            
            with open(test_id_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and ',' in line:
                        # 分割蛋白质对，支持2列或3列格式
                        parts = line.split(',')
                        if len(parts) >= 2:
                            protein1, protein2 = parts[0], parts[1]
                            protein_ids.add(protein1.strip())
                            protein_ids.add(protein2.strip())
            
            self.logger.info(f"从{test_id_file}中提取了{len(protein_ids)}个唯一蛋白质ID")
            
        except Exception as e:
            self.logger.error(f"读取预测ID文件失败: {e}")
        
        return protein_ids
    
    def _read_fasta_sequences(self, fasta_file: str) -> Dict[str, str]:
        """读取FASTA文件中的蛋白质序列"""
        sequences = {}
        
        try:
            fasta_path = Path(fasta_file)
            if not fasta_path.exists():
                self.logger.warning(f"FASTA文件不存在: {fasta_file}")
                return sequences
            
            current_id = None
            current_seq = []
            
            with open(fasta_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('>'):
                        # 保存前一个序列
                        if current_id and current_seq:
                            sequences[current_id] = ''.join(current_seq)
                        
                        # 开始新序列
                        current_id = line[1:]  # 去掉'>'
                        current_seq = []
                    elif current_id:
                        current_seq.append(line)
                
                # 保存最后一个序列
                if current_id and current_seq:
                    sequences[current_id] = ''.join(current_seq)
            
            self.logger.info(f"从{fasta_file}中读取了{len(sequences)}个蛋白质序列")
            
        except Exception as e:
            self.logger.error(f"读取FASTA文件失败: {e}")
        
        return sequences
    
    def _write_filtered_fasta(self, sequences: Dict[str, str], protein_ids: Set[str], output_file: str):
        """将筛选后的蛋白质序列写入FASTA文件"""
        try:
            filtered_count = 0
            with open(output_file, 'w', encoding='utf-8') as f:
                for protein_id in protein_ids:
                    if protein_id in sequences:
                        f.write(f'>{protein_id}\n')
                        # 每行60个字符
                        seq = sequences[protein_id]
                        for i in range(0, len(seq), 60):
                            f.write(seq[i:i+60] + '\n')
                        filtered_count += 1
                    else:
                        self.logger.warning(f"蛋白质{protein_id}在FASTA文件中未找到")
            
            self.logger.debug(f"已写入{filtered_count}个预测蛋白质序列到{output_file}")
            
        except Exception as e:
            self.logger.error(f"写入筛选后的FASTA文件失败: {e}")
    
    def _copy_prediction_fasta_files(self):
        """复制预测中使用的FASTA序列到结果目录"""
        try:
            # 获取配置路径
            config_paths = self._get_config_paths()
            fasta_file = config_paths['fasta_file']
            test_id_file = config_paths['test_id_file']
            
            # 提取预测中使用的蛋白质ID
            protein_ids = self._extract_predicted_protein_ids(test_id_file)
            if not protein_ids:
                self.logger.warning("未找到预测蛋白质ID，跳过FASTA文件复制")
                return
            
            # 读取完整的FASTA文件
            all_sequences = self._read_fasta_sequences(fasta_file)
            if not all_sequences:
                self.logger.warning("未读取到蛋白质序列，跳过FASTA文件复制")
                return
            
            # 创建筛选后的FASTA文件
            output_fasta = self.output_dir / "prediction_proteins.fasta"
            self._write_filtered_fasta(all_sequences, protein_ids, str(output_fasta))
            
            # 同时复制原始predict_id.csv文件
            test_id_path = Path(test_id_file)
            if test_id_path.exists():
                target_id_file = self.output_dir / "predict_id.csv"
                shutil.copy2(test_id_path, target_id_file)
                self.logger.debug(f"已复制预测ID文件: {test_id_file} -> {target_id_file}")
            
        except Exception as e:
            self.logger.error(f"复制预测FASTA文件失败: {e}")
    
    def _find_config_file(self) -> Optional[str]:
        """自动查找配置文件"""
        possible_paths = [
            self.output_dir / "resolved_config.yaml",  # 优先使用resolved_config
            self.output_dir / "best_config.yaml",
            self.output_dir / "config.yaml",
            self.output_dir / "model_config.yaml",
        ]
        
        # 检查是否是ensemble输出目录
        if self._is_ensemble_output_dir():
            # 添加ensemble相关的配置路径
            ensemble_paths = self._get_ensemble_config_paths()
            possible_paths.extend(ensemble_paths)
        
        # 添加通用后备路径
        possible_paths.extend([
            Path("results/sepal_ppi_training_20250821_005451/resolved_config.yaml"),
            Path("results/sepal_ppi_training_20250821_005451/best_config.yaml"),
            Path("config/config.yaml"),
        ])
        
        for path in possible_paths:
            if path.exists():
                self.logger.info(f"找到配置文件: {path}")
                return str(path)
        
        return None
    
    def _is_ensemble_output_dir(self) -> bool:
        """检查是否是ensemble输出目录"""
        # 检查是否包含ensemble特征文件
        ensemble_indicators = [
            "ensemble_predictions.csv",
            "prediction_summary.json"
        ]
        
        for indicator in ensemble_indicators:
            if (self.output_dir / indicator).exists():
                return True
        return False
    
    def _get_ensemble_config_paths(self) -> List[Path]:
        """获取ensemble相关的配置文件路径"""
        ensemble_paths = []
        
        # 查找ensemble配置文件
        ensemble_config_patterns = [
            "results/sepal_ppi_ensemble_training_*/ensemble_inference_config.yaml",
            "results/*/ensemble_inference_config.yaml"
        ]
        
        import glob
        for pattern in ensemble_config_patterns:
            for config_file in glob.glob(pattern):
                ensemble_paths.append(Path(config_file))
        
        # 从ensemble配置中提取模型路径
        if ensemble_paths:
            try:
                import yaml
                with open(ensemble_paths[0], 'r', encoding='utf-8') as f:
                    ensemble_config = yaml.safe_load(f)
                
                # 提取模型配置路径
                models = ensemble_config.get('ensemble', {}).get('models', [])
                for model in models:
                    config_path = model.get('config_path')
                    if config_path:
                        # 优先查找resolved_config.yaml
                        model_dir = Path(config_path).parent
                        resolved_config = model_dir / "resolved_config.yaml"
                        if resolved_config.exists():
                            ensemble_paths.append(resolved_config)
                        else:
                            ensemble_paths.append(Path(config_path))
                            
            except Exception as e:
                self.logger.warning(f"解析ensemble配置文件失败: {e}")
        
        return ensemble_paths
        
    def generate_prediction_results_html(self, results: Dict[str, Any]) -> str:
        """
        生成预测结果的HTML界面（基于外部模板文件）
        
        Args:
            results: ensemble_predict_mode返回的结果字典
            
        Returns:
            生成的HTML文件路径
        """
        try:
            # 输出HTML文件路径
            html_file_path = self.output_dir / "prediction_results.html"

            # 收集数据源文件
            csv_file = self._find_csv_file()
            jsonl_files = self._find_jsonl_files()
            json_files = self._find_json_files()

            if not csv_file:
                self.logger.error("未找到预测结果CSV文件")
                return ""

            # 加载数据
            csv_data = self._load_csv_data(csv_file)
            protein_sequences = self._load_protein_sequences(limit=1000)
            
            # 先将可解释性数据分割成小文件用于动态加载
            self._create_interpretability_chunks(jsonl_files, json_files)
            
            # 然后从chunk文件创建索引
            interpretability_index = self._create_interpretability_index(jsonl_files, json_files)
            self.logger.info(f"创建了包含 {len(interpretability_index)} 个蛋白对的可解释性数据索引")

            # 模板路径
            template_path = Path(__file__).parent / "templates" / "prediction_results.html"
            if not template_path.exists():
                self.logger.warning("未找到模板文件，回退到内联生成方式")
                # 回退到旧的内联生成逻辑（传递索引而不是全部数据）
                html_content = self._generate_html_content(csv_file, jsonl_files, json_files, csv_data, {})
            else:
                # 读取模板并替换占位符
                with open(template_path, 'r', encoding='utf-8') as tf:
                    template_html = tf.read()

                # 将数据注入到模板中（不再嵌入可解释性数据）
                html_content = (
                    template_html
                        .replace('__CSV_DATA__', json.dumps(csv_data))
                        .replace('__INTERPRETABILITY_INDEX__', json.dumps(interpretability_index))
                        .replace('__PROTEIN_SEQUENCES__', json.dumps(protein_sequences))
                )

            # 写入HTML文件
            with open(html_file_path, 'w', encoding='utf-8') as f:
                f.write(html_content)

            self.logger.info(f"预测结果HTML界面已生成: {html_file_path}")
            return str(html_file_path)

        except Exception as e:
            self.logger.error(f"生成HTML界面时出错: {e}")
            return ""
    
    def _find_csv_file(self) -> Optional[str]:
        """查找CSV预测结果文件"""
        csv_files = list(self.output_dir.glob("*predictions.csv"))
        if csv_files:
            return csv_files[0].name
        return None
    
    def _find_jsonl_files(self) -> List[str]:
        """查找JSONL可解释性文件"""
        jsonl_files = list(self.output_dir.glob("*_interpretability.jsonl"))
        # 添加注意力权重文件
        attention_files = list(self.output_dir.glob("*_predict_attention_weights.jsonl"))
        jsonl_files.extend(attention_files)
        return [f.name for f in jsonl_files]
    
    def _find_json_files(self) -> List[str]:
        """查找JSON可解释性文件"""
        json_files = list(self.output_dir.glob("*_interpretability.json"))
        json_files.extend(list(self.output_dir.glob("prediction_summary.json")))
        return [f.name for f in json_files]
    
    def _load_csv_data(self, filename: str) -> List[Dict[str, Any]]:
        """加载CSV数据并内嵌到HTML中"""
        csv_data = []
        try:
            # 使用完整路径
            csv_path = self.output_dir / filename
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    csv_data.append(row)
        except FileNotFoundError:
            self.logger.error(f"CSV文件 {filename} 未找到")
        except Exception as e:
            self.logger.error(f"加载CSV文件 {filename} 失败: {e}")
        return csv_data
    
    def _create_interpretability_index(self, jsonl_files: List[str], json_files: List[str]) -> Dict[str, Dict[str, str]]:
        """
        创建可解释性数据索引，记录每个蛋白对的数据来源（指向chunk文件）
        
        Returns:
            格式: {"protein1_protein2": {"attention": "chunk_0.json", "cis": "chunk_1.json"}}
        """
        index = {}
        chunks_dir = self.output_dir / "interpretability_chunks"
        
        try:
            # 如果chunks目录存在，从chunk文件创建索引
            if chunks_dir.exists():
                # 扫描所有chunk文件
                chunk_files = list(chunks_dir.glob("*.json"))
                
                for chunk_file in chunk_files:
                    try:
                        with open(chunk_file, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            
                            # 处理数组格式
                            if isinstance(data, list):
                                for item in data:
                                    key = f"{item.get('protein1_id')}_{item.get('protein2_id')}"
                                    if key not in index:
                                        index[key] = {}
                                    
                                    # 根据文件名判断数据类型
                                    if 'cis_interpretability' in chunk_file.name:
                                        index[key]['cis'] = chunk_file.name
                                    elif 'predict_attention_weights' in chunk_file.name or 'attention' in chunk_file.name:
                                        index[key]['attention'] = chunk_file.name
                            # 处理单个对象格式
                            else:
                                key = f"{data.get('protein1_id')}_{data.get('protein2_id')}"
                                if key not in index:
                                    index[key] = {}
                                
                                if 'cis_interpretability' in chunk_file.name:
                                    index[key]['cis'] = chunk_file.name
                                elif 'predict_attention_weights' in chunk_file.name or 'attention' in chunk_file.name:
                                    index[key]['attention'] = chunk_file.name
                    
                    except Exception as e:
                        self.logger.warning(f"扫描chunk文件 {chunk_file.name} 失败: {e}")
                
                self.logger.info(f"从 {len(chunk_files)} 个chunk文件创建了索引")
            
            else:
                # 回退：如果没有chunk目录，从原始文件创建索引（旧逻辑）
                self.logger.warning("未找到chunks目录，使用原始JSONL文件")
                for file in jsonl_files:
                    try:
                        file_path = self.output_dir / file
                        with open(file_path, 'r', encoding='utf-8') as f:
                            for line in f:
                                if line.strip():
                                    data = json.loads(line)
                                    key = f"{data.get('protein1_id')}_{data.get('protein2_id')}"
                                    if key not in index:
                                        index[key] = {}
                                    
                                    if 'cis_interpretability' in file:
                                        index[key]['cis'] = file
                                    elif 'predict_attention_weights' in file:
                                        index[key]['attention'] = file
                    except Exception as e:
                        self.logger.warning(f"扫描JSONL文件 {file} 失败: {e}")
        
        except Exception as e:
            self.logger.error(f"创建可解释性索引失败: {e}")
        
        return index
    
    def _create_interpretability_chunks(self, jsonl_files: List[str], json_files: List[str], chunk_size: int = 100):
        """
        将可解释性数据分割成小块，用于按需加载
        
        Args:
            jsonl_files: JSONL文件列表
            json_files: JSON文件列表
            chunk_size: 每个分块的最大记录数
        """
        try:
            # 创建分块目录
            chunks_dir = self.output_dir / "interpretability_chunks"
            chunks_dir.mkdir(exist_ok=True)
            
            # 处理JSONL文件
            for file in jsonl_files:
                try:
                    file_path = self.output_dir / file
                    chunk_data = []
                    chunk_index = 0
                    
                    with open(file_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            if line.strip():
                                data = json.loads(line)
                                chunk_data.append(data)
                                
                                # 达到分块大小时写入文件
                                if len(chunk_data) >= chunk_size:
                                    chunk_file = chunks_dir / f"{Path(file).stem}_chunk_{chunk_index}.json"
                                    with open(chunk_file, 'w', encoding='utf-8') as cf:
                                        json.dump(chunk_data, cf)
                                    chunk_data = []
                                    chunk_index += 1
                    
                    # 写入剩余数据
                    if chunk_data:
                        chunk_file = chunks_dir / f"{Path(file).stem}_chunk_{chunk_index}.json"
                        with open(chunk_file, 'w', encoding='utf-8') as cf:
                            json.dump(chunk_data, cf)
                    
                    self.logger.debug(f"已将 {file} 分割为 {chunk_index + 1} 个分块")
                
                except Exception as e:
                    self.logger.warning(f"分块JSONL文件 {file} 失败: {e}")
            
            # JSON文件通常较小，复制到chunks目录即可
            for file in json_files:
                try:
                    src_path = self.output_dir / file
                    dst_path = chunks_dir / file
                    if src_path.exists():
                        shutil.copy2(src_path, dst_path)
                except Exception as e:
                    self.logger.warning(f"复制JSON文件 {file} 失败: {e}")
        
        except Exception as e:
            self.logger.error(f"创建可解释性分块失败: {e}")
    
    def _load_interpretability_data(self, jsonl_files: List[str], json_files: List[str]) -> Dict[str, Any]:
        """加载可解释性数据并内嵌到HTML中"""
        interpretability_data = {}
        try:
            # 加载JSONL文件（残基级注意力权重和CIS可解释性）
            for file in jsonl_files:
                try:
                    # 使用完整路径
                    file_path = self.output_dir / file
                    with open(file_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            if line.strip():
                                data = json.loads(line)
                                key = f"{data.get('protein1_id')}_{data.get('protein2_id')}"
                                if key not in interpretability_data:
                                    interpretability_data[key] = {}
                                
                                # 判断是注意力权重数据还是CIS数据
                                if 'cis_interpretability' in file:
                                    # CIS可解释性数据
                                    interpretability_data[key]['cis'] = data
                                elif 'predict_attention_weights' in file:
                                    # 注意力权重数据
                                    # 直接使用正确的注意力头数（SEPAL模型使用2个头）
                                    data['attention_heads'] = 2
                                    interpretability_data[key]['attention'] = data
                except FileNotFoundError:
                    self.logger.warning(f"JSONL文件 {file} 未找到")
                except Exception as e:
                    self.logger.warning(f"加载JSONL文件 {file} 失败: {e}")

            # 加载JSON文件（CIS可解释性）
            for file in json_files:
                try:
                    # 使用完整路径
                    file_path = self.output_dir / file
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if 'cis_interpretability' in file:
                            for item in data:
                                key = f"{item.get('protein1_id')}_{item.get('protein2_id')}"
                                if key not in interpretability_data:
                                    interpretability_data[key] = {}
                                interpretability_data[key]['cis'] = item
                except FileNotFoundError:
                    self.logger.warning(f"JSON文件 {file} 未找到")
                except Exception as e:
                    self.logger.warning(f"加载JSON文件 {file} 失败: {e}")
        except Exception as e:
            self.logger.error(f"加载可解释性数据失败: {e}")
        return interpretability_data
    
    def _generate_html_content(self, csv_file: str, jsonl_files: List[str], json_files: List[str], csv_data: List[Dict[str, Any]], interpretability_data: Dict[str, Any]) -> str:
        """生成HTML内容"""
        
        # 读取蛋白序列信息（如果存在）- 限制大小以避免HTML文件过大
        protein_sequences = self._load_protein_sequences(limit=1000)
        
        html_content = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SEPAL-PPI 预测结果</title>
    <style>
        {self._get_css_styles()}
    </style>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.26.0/plotly.min.js"></script>
</head>
<body>
    <div class="container">
        <header class="header">
            <h1>SEPAL-PPI 蛋白质相互作用预测结果</h1>
            <div class="search-container">
                <input type="text" id="searchInput" placeholder="搜索蛋白质ID或蛋白对 (例如: P12345 或 P12345,Q67890)" class="search-input">
                <button id="searchBtn" class="search-btn">搜索</button>
                <button id="clearBtn" class="clear-btn">清除</button>
            </div>
        </header>

        <main class="main-content">
            <!-- 统计概览 -->
            <section class="stats-section">
                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="stat-number" id="totalPairs">-</div>
                        <div class="stat-label">总蛋白对数</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number" id="positivePairs">-</div>
                        <div class="stat-label">阳性预测</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number" id="avgProbability">-</div>
                        <div class="stat-label">阴性预测</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number" id="interpretableCount">-</div>
                        <div class="stat-label">可解释结果</div>
                    </div>
                </div>
            </section>

            <!-- 筛选控制 -->
            <section class="filter-section">
                <div class="filter-controls">
                    <label>
                        预测结果筛选:
                        <select id="predictionFilter">
                            <option value="all">全部</option>
                            <option value="positive">阳性预测</option>
                            <option value="negative">阴性预测</option>
                        </select>
                    </label>
                    <label>
                        概率范围:
                        <input type="range" id="probabilityRange" min="0" max="1" step="0.01" value="0">
                        <span id="probabilityValue">≥ 0.00</span>
                    </label>
                    <button id="resetFilters" class="filter-btn">重置筛选</button>
                </div>
            </section>

            <!-- 预测结果表格 -->
            <section class="results-section">
                <h2>预测结果</h2>
                <div class="table-container">
                    <table id="resultsTable" class="results-table">
                        <thead>
                            <tr>
                                <th class="sortable" data-column="protein1">蛋白质1 ↕</th>
                                <th class="sortable" data-column="protein2">蛋白质2 ↕</th>
                                <th class="sortable" data-column="ensemble_probability">集成概率 ↕</th>
                                <th class="sortable" data-column="ensemble_prediction">预测结果 ↕</th>
                                <th>模型详情</th>
                                <th>可解释性分析</th>
                            </tr>
                        </thead>
                        <tbody id="resultsTableBody">
                            <!-- 动态填充 -->
                        </tbody>
                    </table>
                </div>
                <div class="pagination">
                    <button id="prevPage" class="page-btn">← 上一页</button>
                    <span id="pageInfo">第 1 页，共 1 页</span>
                    <button id="nextPage" class="page-btn">下一页 →</button>
                    <select id="pageSize">
                        <option value="20">20/页</option>
                        <option value="50" selected>50/页</option>
                        <option value="100">100/页</option>
                    </select>
                </div>
            </section>
        </main>

        <!-- 可解释性分析模态框 -->
        <div id="interpretabilityModal" class="modal">
            <div class="modal-content">
                <div class="modal-header">
                    <h3 id="modalTitle">可解释性分析</h3>
                    <span class="close" id="closeModal">&times;</span>
                </div>
                <div class="modal-body" id="modalBody">
                    <!-- 动态填充 -->
                </div>
            </div>
        </div>

        <!-- 蛋白详情模态框 -->
        <div id="proteinModal" class="modal">
            <div class="modal-content">
                <div class="modal-header">
                    <h3 id="proteinModalTitle">蛋白质详情</h3>
                    <span class="close" id="closeProteinModal">&times;</span>
                </div>
                <div class="modal-body" id="proteinModalBody">
                    <!-- 动态填充 -->
                </div>
            </div>
        </div>
    </div>

    <script>
        {self._get_javascript_code(csv_file, jsonl_files, json_files, csv_data, interpretability_data)}
    </script>
</body>
</html>"""
        
        return html_content
    
    def _load_protein_sequences(self, limit: int = 1000) -> Dict[str, str]:
        """加载蛋白质序列信息（限制数量以避免HTML过大）"""
        protein_sequences = {}
        
        # 按优先级搜索FASTA文件
        fasta_files = []
        
        # 1. 首先检查筛选后的预测蛋白质文件（最高优先级）
        prediction_fasta = self.output_dir / "prediction_proteins.fasta"
        if prediction_fasta.exists():
            fasta_files.append(prediction_fasta)
        
        # 2. 检查结果目录中的其他FASTA文件
        result_dir_fastas = list(self.output_dir.glob("*.fasta"))
        for fasta in result_dir_fastas:
            if fasta.name != "prediction_proteins.fasta":  # 避免重复添加
                fasta_files.append(fasta)
        
        # 3. 检查已知的源数据路径
        potential_paths = [
            Path("dataset/S1/protein.fasta"),  # 主要的蛋白质序列文件
            Path("dataset/S1/S1_ext_lossPDB.fasta"),  # 备用文件
        ]
        
        for path in potential_paths:
            if path.exists():
                fasta_files.append(path)
        
        # 4. 添加其他可能的位置
        fasta_files.extend(list(self.output_dir.parent.glob("*.fasta")))
        fasta_files.extend(list(Path(".").glob("**/protein.fasta")))
        
        # 去重，保持顺序
        seen = set()
        unique_fasta_files = []
        for f in fasta_files:
            if f not in seen:
                seen.add(f)
                unique_fasta_files.append(f)
        
        for fasta_file in unique_fasta_files[:1]:  # 只读取第一个找到的FASTA文件
            try:
                with open(fasta_file, 'r', encoding='utf-8') as f:
                    current_id = None
                    current_seq = []
                    
                    for line in f:
                        line = line.strip()
                        if line.startswith('>'):
                            if current_id and current_seq:
                                protein_sequences[current_id] = ''.join(current_seq)
                            current_id = line[1:].split()[0]  # 获取ID，忽略描述
                            current_seq = []
                        elif current_id:
                            current_seq.append(line)
                    
                    # 处理最后一个序列
                    if current_id and current_seq:
                        protein_sequences[current_id] = ''.join(current_seq)
                    
                    # 限制序列数量
                    if len(protein_sequences) >= limit:
                        break
                        
                self.logger.info(f"已加载 {len(protein_sequences)} 个蛋白质序列（限制: {limit}）")
                break
                
            except Exception as e:
                self.logger.warning(f"读取FASTA文件 {fasta_file} 失败: {e}")
        
        return protein_sequences
    
    def _load_results_from_dir(self) -> Dict[str, Any]:
        """从输出目录加载结果数据"""
        results = {}
        
        try:
            # 加载ensemble预测结果
            ensemble_file = self.output_dir / "ensemble_predictions.csv"
            if ensemble_file.exists():
                with open(ensemble_file, 'r', encoding='utf-8') as f:
                    import csv
                    reader = csv.DictReader(f)
                    ensemble_predictions = list(reader)
                    results['ensemble_predictions'] = ensemble_predictions
                    self.logger.info(f"加载了 {len(ensemble_predictions)} 个ensemble预测结果")
            
            # 加载预测摘要
            summary_file = self.output_dir / "prediction_summary.json"
            if summary_file.exists():
                with open(summary_file, 'r', encoding='utf-8') as f:
                    summary = json.load(f)
                    results['prediction_summary'] = summary
                    self.logger.info("加载了预测摘要数据")
            
            # 如果没有ensemble结果，从摘要中构建
            if 'ensemble_predictions' not in results and 'prediction_summary' in results:
                predictions = []
                summary = results['prediction_summary']
                
                if 'predictions' in summary:
                    for pred in summary['predictions']:
                        predictions.append({
                            'protein1_id': pred.get('protein1_id', ''),
                            'protein2_id': pred.get('protein2_id', ''),
                            'ensemble_score': pred.get('ensemble_score', 0.0),
                            'ensemble_prediction': pred.get('ensemble_prediction', 0),
                            'confidence': pred.get('confidence', 'medium')
                        })
                
                results['ensemble_predictions'] = predictions
                self.logger.info(f"从摘要中构建了 {len(predictions)} 个预测结果")
            
        except Exception as e:
            self.logger.error(f"加载结果数据失败: {e}")
        
        return results
    
    def _get_css_styles(self) -> str:
        """返回CSS样式"""
        return """
        :root {
            --primary-color: #6366f1;
            --secondary-color: #8b5cf6;
            --background-color: #f8fafc;
            --surface-color: #ffffff;
            --text-primary: #1e293b;
            --text-secondary: #64748b;
            --border-color: #e2e8f0;
            --success-color: #10b981;
            --warning-color: #f59e0b;
            --error-color: #ef4444;
            --gradient-bg: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            --box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: var(--background-color);
            color: var(--text-primary);
            line-height: 1.6;
        }

        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }

        .header {
            background: var(--gradient-bg);
            padding: 30px;
            border-radius: 12px;
            margin-bottom: 30px;
            box-shadow: var(--box-shadow);
        }

        .header h1 {
            color: white;
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 20px;
            text-align: center;
        }

        .search-container {
            display: flex;
            gap: 10px;
            max-width: 600px;
            margin: 0 auto;
        }

        .search-input {
            flex: 1;
            padding: 12px 16px;
            border: none;
            border-radius: 8px;
            font-size: 16px;
            box-shadow: var(--box-shadow);
        }

        .search-btn, .clear-btn {
            padding: 12px 20px;
            border: none;
            border-radius: 8px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
        }

        .search-btn {
            background-color: var(--success-color);
            color: white;
        }

        .clear-btn {
            background-color: var(--error-color);
            color: white;
        }

        .search-btn:hover {
            background-color: #059669;
            transform: translateY(-2px);
        }

        .clear-btn:hover {
            background-color: #dc2626;
            transform: translateY(-2px);
        }

        .stats-section {
            margin-bottom: 30px;
        }

        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
        }

        .stat-card {
            background: var(--surface-color);
            padding: 25px;
            border-radius: 12px;
            text-align: center;
            box-shadow: var(--box-shadow);
            border: 1px solid var(--border-color);
        }

        .stat-number {
            font-size: 2.5rem;
            font-weight: 700;
            color: var(--primary-color);
            margin-bottom: 8px;
        }

        .stat-label {
            color: var(--text-secondary);
            font-weight: 500;
        }

        .filter-section {
            background: var(--surface-color);
            padding: 20px;
            border-radius: 12px;
            margin-bottom: 30px;
            box-shadow: var(--box-shadow);
            border: 1px solid var(--border-color);
        }

        .filter-controls {
            display: flex;
            gap: 20px;
            align-items: center;
            flex-wrap: wrap;
        }

        .filter-controls label {
            display: flex;
            align-items: center;
            gap: 8px;
            font-weight: 500;
        }

        .filter-controls select, .filter-controls input {
            padding: 8px 12px;
            border: 1px solid var(--border-color);
            border-radius: 6px;
            font-size: 14px;
        }

        .filter-btn {
            padding: 8px 16px;
            background-color: var(--secondary-color);
            color: white;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-weight: 500;
            transition: all 0.3s ease;
        }

        .filter-btn:hover {
            background-color: #7c3aed;
            transform: translateY(-1px);
        }

        .results-section {
            background: var(--surface-color);
            border-radius: 12px;
            padding: 25px;
            box-shadow: var(--box-shadow);
            border: 1px solid var(--border-color);
        }

        .results-section h2 {
            color: var(--text-primary);
            margin-bottom: 20px;
            font-size: 1.5rem;
        }

        .table-container {
            overflow-x: auto;
            margin-bottom: 20px;
        }

        .results-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 14px;
        }

        .results-table th {
            background: var(--primary-color);
            color: white;
            padding: 15px 10px;
            font-weight: 600;
            text-align: left;
            position: sticky;
            top: 0;
            cursor: pointer;
            user-select: none;
        }

        .results-table th:hover {
            background: rgba(255, 255, 255, 0.1);
        }

        .results-table td {
            padding: 12px 10px;
            border-bottom: 1px solid var(--border-color);
        }

        .results-table tr:hover {
            background-color: #f1f5f9;
        }

        .protein-link {
            color: var(--primary-color);
            text-decoration: none;
            font-weight: 500;
            cursor: pointer;
        }

        .protein-link:hover {
            text-decoration: underline;
        }

        .probability-cell {
            font-weight: 600;
        }

        .prediction-positive {
            background-color: #dcfce7;
            color: #166534;
            padding: 4px 8px;
            border-radius: 4px;
            font-weight: 600;
        }

        .prediction-negative {
            background-color: #fee2e2;
            color: #991b1b;
            padding: 4px 8px;
            border-radius: 4px;
            font-weight: 600;
        }

        .interpretability-btn {
            padding: 6px 12px;
            background-color: var(--primary-color);
            color: white;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-size: 12px;
            font-weight: 500;
            transition: all 0.3s ease;
        }

        .interpretability-btn:hover {
            background-color: #4f46e5;
            transform: translateY(-1px);
        }

        .interpretability-btn:disabled {
            background-color: var(--text-secondary);
            cursor: not-allowed;
            transform: none;
        }

        .pagination {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 15px;
            margin-top: 20px;
        }

        .page-btn {
            padding: 8px 16px;
            background-color: var(--primary-color);
            color: white;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-weight: 500;
            transition: all 0.3s ease;
        }

        .page-btn:hover:not(:disabled) {
            background-color: #4f46e5;
            transform: translateY(-1px);
        }

        .page-btn:disabled {
            background-color: var(--text-secondary);
            cursor: not-allowed;
            transform: none;
        }

        .modal {
            display: none;
            position: fixed;
            z-index: 1000;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0, 0, 0, 0.5);
        }

        .modal-content {
            background-color: var(--surface-color);
            margin: 2% auto;
            padding: 0;
            border-radius: 12px;
            width: 90%;
            max-width: 1000px;
            max-height: 90%;
            overflow: hidden;
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.25);
        }

        .modal-header {
            background: var(--primary-color);
            color: white;
            padding: 20px 30px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .modal-header h3 {
            margin: 0;
            font-size: 1.5rem;
        }

        .close {
            font-size: 28px;
            font-weight: bold;
            cursor: pointer;
            transition: color 0.3s ease;
        }

        .close:hover {
            color: #f1f5f9;
        }

        .modal-body {
            padding: 30px;
            max-height: 70vh;
            overflow-y: auto;
        }

        .sequence-viewer {
            background: #f8fafc;
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 20px;
            margin: 15px 0;
        }

        .sequence-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }

        .sequence-title {
            font-weight: 600;
            color: var(--text-primary);
        }

        .sequence-length {
            color: var(--text-secondary);
            font-size: 14px;
        }

        .sequence-display {
            font-family: 'Courier New', monospace;
            font-size: 12px;
            line-height: 1.8;
            background: white;
            padding: 15px;
            border-radius: 6px;
            border: 1px solid var(--border-color);
            word-break: break-all;
            max-height: 200px;
            overflow-y: auto;
        }

        /* 改进的序列显示 */
        .sequence-display-container {
            font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', 'Courier New', monospace;
            font-size: 14px;
            line-height: 1.4;
            background: white;
            padding: 15px;
            border-radius: 6px;
            border: 1px solid var(--border-color);
            max-height: 300px;
            overflow-y: auto;
        }

        .sequence-line {
            display: flex;
            align-items: flex-start;
            margin-bottom: 5px;
        }

        .residue-index {
            font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', 'Courier New', monospace;
            font-size: 12px;
            color: var(--text-secondary);
            background: #f1f5f9;
            padding: 2px 8px;
            border-radius: 4px;
            margin-right: 10px;
            min-width: 60px;
            text-align: right;
            flex-shrink: 0;
        }

        .sequence-residues {
            font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', 'Courier New', monospace;
            letter-spacing: 0.5px;
            flex: 1;
        }

        .attention-residue {
            display: inline-block;
            padding: 1px 2px;
            margin: 0;
            border-radius: 2px;
            transition: all 0.3s ease;
            cursor: pointer;
            background-color: transparent;
            font-weight: normal;
        }

        .attention-residue:hover {
            background-color: rgba(0, 0, 0, 0.1);
            transform: scale(1.1);
        }

        /* 注意力控制面板 */
        .attention-controls {
            display: flex;
            gap: 20px;
            align-items: center;
            margin: 15px 0;
            padding: 15px;
            background: #f8fafc;
            border-radius: 8px;
            border: 1px solid var(--border-color);
            flex-wrap: wrap;
        }

        .attention-controls label {
            display: flex;
            align-items: center;
            gap: 8px;
            font-weight: 500;
            color: var(--text-primary);
        }

        .attention-controls input, .attention-controls select {
            padding: 6px 10px;
            border: 1px solid var(--border-color);
            border-radius: 4px;
            font-size: 14px;
        }

        .attention-legend {
            display: flex;
            align-items: center;
            gap: 15px;
            margin: 15px 0;
            padding: 10px 15px;
            background: white;
            border-radius: 6px;
            border: 1px solid var(--border-color);
        }

        .legend-gradient {
            width: 200px;
            height: 20px;
            border: 1px solid #ccc;
            border-radius: 4px;
        }

        .legend-labels {
            display: flex;
            justify-content: space-between;
            width: 200px;
            font-size: 12px;
            color: var(--text-secondary);
        }

        /* CIS表格样式改进 */
        .cis-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
            font-size: 14px;
        }

        .cis-table th {
            background: var(--primary-color);
            color: white;
            padding: 12px 8px;
            text-align: left;
            font-weight: 600;
        }

        .cis-table td {
            padding: 10px 8px;
            border-bottom: 1px solid var(--border-color);
        }

        .cis-table tr:hover {
            background-color: #f8fafc;
        }

        .similarity-score {
            font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', 'Courier New', monospace;
            font-weight: 600;
            color: var(--primary-color);
        }

        .scientific-notation {
            font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', 'Courier New', monospace;
        }

        .positive-label {
            background-color: #dcfce7;
            color: #166534;
            padding: 4px 8px;
            border-radius: 4px;
            font-weight: 600;
            font-size: 12px;
        }

        .negative-label {
            background-color: #fee2e2;
            color: #991b1b;
            padding: 4px 8px;
            border-radius: 4px;
            font-weight: 600;
            font-size: 12px;
        }

        /* 注意力头选择器样式 */
        .attention-head-selector {
            margin: 10px 0;
            padding: 10px;
            background: #f1f5f9;
            border-radius: 8px;
            border: 1px solid var(--border-color);
        }

        .attention-head-selector label {
            font-weight: 500;
            color: var(--text-primary);
            margin-right: 10px;
        }

        .attention-head-selector select {
            padding: 8px 12px;
            border: 1px solid var(--border-color);
            border-radius: 6px;
            font-size: 14px;
            background: white;
            min-width: 150px;
        }

        /* 序列复制样式 */
        .sequence-copy-container {
            margin: 10px 0;
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .copy-sequence-btn {
            padding: 8px 16px;
            background-color: var(--secondary-color);
            color: white;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-weight: 500;
            font-size: 14px;
            transition: all 0.3s ease;
        }

        .copy-sequence-btn:hover {
            background-color: #7c3aed;
            transform: translateY(-1px);
        }

        .copy-feedback {
            color: var(--success-color);
            font-weight: 500;
            font-size: 14px;
            opacity: 0;
            transition: opacity 0.3s ease;
        }

        .copy-feedback.show {
            opacity: 1;
        }

        /* 序列显示改进 - 类似FASTA格式 */
        .sequence-display-fasta {
            font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', 'Courier New', monospace;
            font-size: 14px;
            line-height: 1.6;
            background: white;
            padding: 15px;
            border-radius: 6px;
            border: 1px solid var(--border-color);
            max-height: 400px;
            overflow-y: auto;
            user-select: text;
        }

        .sequence-fasta-line {
            display: block;
            margin-bottom: 2px;
            white-space: nowrap;
        }

        .sequence-position-header {
            color: var(--text-secondary);
            font-size: 12px;
            margin-bottom: 5px;
            font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', 'Courier New', monospace;
            display: flex;
            justify-content: flex-start;
            align-items: center;
            white-space: nowrap;
        }

        .position-marker {
            display: inline-block;
            width: 10px;
            text-align: center;
            margin-right: 0;
            color: var(--text-secondary);
            font-size: 10px;
            font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', 'Courier New', monospace;
        }

        .attention-residue-fasta {
            display: inline-block;
            width: 10px;
            text-align: center;
            transition: all 0.3s ease;
            cursor: pointer;
            font-weight: normal;
            letter-spacing: 0;
            font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', 'Courier New', monospace;
        }

        .attention-residue-fasta:hover {
            background-color: rgba(0, 0, 0, 0.1);
            border-radius: 2px;
        }

        .sequence-spacer {
            display: inline-block;
            width: 50px;
        }

        @media (max-width: 768px) {
            .container {
                padding: 10px;
            }

            .header h1 {
                font-size: 1.8rem;
            }

            .search-container {
                flex-direction: column;
            }

            .stats-grid {
                grid-template-columns: repeat(2, 1fr);
            }

            .filter-controls {
                flex-direction: column;
                align-items: stretch;
            }

            .results-table {
                font-size: 12px;
            }

            .modal-content {
                width: 95%;
                margin: 5% auto;
            }

            .modal-body {
                padding: 20px;
            }
        }
        """
    
    def _get_javascript_code(self, csv_file: str, jsonl_files: List[str], json_files: List[str], csv_data: List[Dict[str, Any]], interpretability_data: Dict[str, Any]) -> str:
        """返回JavaScript代码"""
        return f"""
        // 全局变量
        let allData = {json.dumps(csv_data)}; // 内嵌CSV数据
        let filteredData = [...allData]; // 当前页数据
        let currentPage = 1;
        let pageSize = 50;
        let sortColumn = '';
        let sortDirection = 'asc';
        let interpretabilityData = {json.dumps(interpretability_data)}; // 内嵌可解释性数据
        let proteinSequences = {json.dumps(self._load_protein_sequences(1000))}; // 内嵌蛋白质序列

        // 初始化
        document.addEventListener('DOMContentLoaded', function() {{
            loadData();
            setupEventListeners();
        }});

        // 加载数据 (此函数不再需要，数据已内嵌)
        async function loadData() {{
            // 数据已内嵌，直接更新统计信息
            updateStats();
            renderTable();
        }}

        // 加载CSV数据 (此函数不再需要，数据已内嵌)

        // 加载可解释性数据 (此函数不再需要，数据已内嵌)
        async function loadInterpretabilityData(jsonlFiles, jsonFiles) {{
            // 数据已内嵌，直接使用
        }}

        // 设置事件监听器
        function setupEventListeners() {{
            // 搜索功能
            document.getElementById('searchBtn').addEventListener('click', performSearch);
            document.getElementById('searchInput').addEventListener('keypress', function(e) {{
                if (e.key === 'Enter') {{
                    performSearch();
                }}
            }});
            document.getElementById('clearBtn').addEventListener('click', clearSearch);

            // 筛选功能
            document.getElementById('predictionFilter').addEventListener('change', applyFilters);
            document.getElementById('probabilityRange').addEventListener('input', function() {{
                document.getElementById('probabilityValue').textContent = `≥ ${{this.value}}`;
                applyFilters();
            }});
            document.getElementById('resetFilters').addEventListener('click', resetFilters);

            // 分页功能
            document.getElementById('prevPage').addEventListener('click', () => changePage(-1));
            document.getElementById('nextPage').addEventListener('click', () => changePage(1));
            document.getElementById('pageSize').addEventListener('change', function() {{
                pageSize = parseInt(this.value);
                currentPage = 1;
                renderTable();
            }});

            // 表格排序
            document.querySelectorAll('.sortable').forEach(th => {{
                th.addEventListener('click', () => sortTable(th.dataset.column));
            }});

            // 模态框
            document.getElementById('closeModal').addEventListener('click', () => {{
                document.getElementById('interpretabilityModal').style.display = 'none';
            }});
            document.getElementById('closeProteinModal').addEventListener('click', () => {{
                document.getElementById('proteinModal').style.display = 'none';
            }});

            // 点击模态框外部关闭
            window.addEventListener('click', function(event) {{
                const modal1 = document.getElementById('interpretabilityModal');
                const modal2 = document.getElementById('proteinModal');
                if (event.target === modal1) {{
                    modal1.style.display = 'none';
                }}
                if (event.target === modal2) {{
                    modal2.style.display = 'none';
                }}
            }});
        }}

        // 执行搜索
        function performSearch() {{
            const query = document.getElementById('searchInput').value.trim().toLowerCase();
            if (!query) {{
                filteredData = [...allData];
            }} else {{
                filteredData = allData.filter(row => {{
                    const protein1 = row.protein1.toLowerCase();
                    const protein2 = row.protein2.toLowerCase();
                    
                    // 支持单个蛋白质搜索
                    if (protein1.includes(query) || protein2.includes(query)) {{
                        return true;
                    }}
                    
                    // 支持蛋白对搜索（逗号分隔）
                    if (query.includes(',')) {{
                        const [p1, p2] = query.split(',').map(p => p.trim());
                        return (protein1.includes(p1) && protein2.includes(p2)) ||
                               (protein1.includes(p2) && protein2.includes(p1));
                    }}
                    
                    return false;
                }});
            }}
            
            currentPage = 1;
            applyFilters();
        }}

        // 清除搜索
        function clearSearch() {{
            document.getElementById('searchInput').value = '';
            filteredData = [...allData];
            currentPage = 1;
            applyFilters();
        }}

        // 应用筛选
        function applyFilters() {{
            const predictionFilter = document.getElementById('predictionFilter').value;
            const probabilityThreshold = parseFloat(document.getElementById('probabilityRange').value);
            
            let filtered = [...filteredData];
            
            // 预测结果筛选
            if (predictionFilter === 'positive') {{
                filtered = filtered.filter(row => parseInt(row.ensemble_prediction) === 1);
            }} else if (predictionFilter === 'negative') {{
                filtered = filtered.filter(row => parseInt(row.ensemble_prediction) === 0);
            }}
            
            // 概率筛选
            filtered = filtered.filter(row => parseFloat(row.ensemble_probability) >= probabilityThreshold);
            
            filteredData = filtered;
            currentPage = 1;
            updateStats();
            renderTable();
        }}

        // 重置筛选
        function resetFilters() {{
            document.getElementById('predictionFilter').value = 'all';
            document.getElementById('probabilityRange').value = '0';
            document.getElementById('probabilityValue').textContent = '≥ 0.00';
            document.getElementById('searchInput').value = '';
            
            filteredData = [...allData];
            currentPage = 1;
            updateStats();
            renderTable();
        }}

        // 更新统计信息
        function updateStats() {{
            const totalPairs = allData.length;
            const positivePairs = allData.filter(row => parseInt(row.ensemble_prediction) === 1).length;
            const negativePairs = allData.filter(row => parseInt(row.ensemble_prediction) === 0).length;
            const interpretableCount = Object.keys(interpretabilityData).length;
            
            document.getElementById('totalPairs').textContent = totalPairs.toLocaleString();
            document.getElementById('positivePairs').textContent = positivePairs.toLocaleString();
            document.getElementById('avgProbability').textContent = negativePairs.toLocaleString();
            document.getElementById('interpretableCount').textContent = interpretableCount.toLocaleString();
        }}

        // 表格排序
        function sortTable(column) {{
            if (sortColumn === column) {{
                sortDirection = sortDirection === 'asc' ? 'desc' : 'asc';
            }} else {{
                sortColumn = column;
                sortDirection = 'asc';
            }}
            
            filteredData.sort((a, b) => {{
                let aVal = a[column];
                let bVal = b[column];
                
                // 数值排序
                if (column === 'ensemble_probability' || column === 'ensemble_prediction') {{
                    aVal = parseFloat(aVal);
                    bVal = parseFloat(bVal);
                }}
                
                if (aVal < bVal) return sortDirection === 'asc' ? -1 : 1;
                if (aVal > bVal) return sortDirection === 'asc' ? 1 : -1;
                return 0;
            }});
            
            renderTable();
            updateSortIcons();
        }}

        // 更新排序图标
        function updateSortIcons() {{
            document.querySelectorAll('.sortable').forEach(th => {{
                const column = th.dataset.column;
                if (column === sortColumn) {{
                    th.textContent = th.textContent.replace(/ [↕↑↓]/, '') + (sortDirection === 'asc' ? ' ↑' : ' ↓');
                }} else {{
                    th.textContent = th.textContent.replace(/ [↕↑↓]/, '') + ' ↕';
                }}
            }});
        }}

        // 渲染表格
        function renderTable() {{
            const tbody = document.getElementById('resultsTableBody');
            const startIndex = (currentPage - 1) * pageSize;
            const endIndex = startIndex + pageSize;
            const pageData = filteredData.slice(startIndex, endIndex);
            
            tbody.innerHTML = '';
            
            pageData.forEach(row => {{
                const tr = document.createElement('tr');
                
                // 获取模型列名（动态检测）
                const modelColumns = Object.keys(row).filter(key => 
                    key.includes('_probability') || key.includes('_weight')
                );
                
                const modelDetails = modelColumns
                    .filter(col => col.includes('_probability'))
                    .map(col => {{
                        const modelName = col.replace('_probability', '');
                        const prob = parseFloat(row[col]).toFixed(3);
                        // 对于ensemble模型，不显示权重
                        if (modelName === 'ensemble') {{
                            return `${{modelName}}: ${{prob}}`;
                        }} else {{
                            const weight = row[modelName + '_weight'] ? parseFloat(row[modelName + '_weight']).toFixed(3) : 'N/A';
                            return `${{modelName}}: ${{prob}} (权重: ${{weight}})`;
                        }}
                    }})
                    .join('<br>');
                
                const hasInterpretability = interpretabilityData[`${{row.protein1}}_${{row.protein2}}`] ? 
                    '' : 'disabled';
                
                tr.innerHTML = `
                    <td><a href="#" class="protein-link" onclick="showProteinDetails('${{row.protein1}}')">${{row.protein1}}</a></td>
                    <td><a href="#" class="protein-link" onclick="showProteinDetails('${{row.protein2}}')">${{row.protein2}}</a></td>
                    <td class="probability-cell">${{parseFloat(row.ensemble_probability).toFixed(3)}}</td>
                    <td>
                        <span class="prediction-${{parseInt(row.ensemble_prediction) === 1 ? 'positive' : 'negative'}}">
                            ${{parseInt(row.ensemble_prediction) === 1 ? '阳性' : '阴性'}}
                        </span>
                    </td>
                    <td style="font-size: 12px;">${{modelDetails}}</td>
                    <td>
                        <button class="interpretability-btn" ${{hasInterpretability}} 
                                onclick="showInterpretability('${{row.protein1}}', '${{row.protein2}}')">
                            查看分析
                        </button>
                    </td>
                `;
                
                tbody.appendChild(tr);
            }});
            
            updatePagination();
        }}

        // 更新分页
        function updatePagination() {{
            const totalPages = Math.ceil(filteredData.length / pageSize);
            
            document.getElementById('prevPage').disabled = currentPage === 1;
            document.getElementById('nextPage').disabled = currentPage === totalPages || totalPages === 0;
            document.getElementById('pageInfo').textContent = 
                `第 ${{currentPage}} 页，共 ${{totalPages}} 页 (共 ${{filteredData.length}} 条记录)`;
        }}

        // 切换页面
        function changePage(direction) {{
            const totalPages = Math.ceil(filteredData.length / pageSize);
            const newPage = currentPage + direction;
            
            if (newPage >= 1 && newPage <= totalPages) {{
                currentPage = newPage;
                renderTable();
            }}
        }}

        // 显示蛋白质详情
        function showProteinDetails(proteinId) {{
            const modal = document.getElementById('proteinModal');
            const title = document.getElementById('proteinModalTitle');
            const body = document.getElementById('proteinModalBody');
            
            title.textContent = `蛋白质详情: ${{proteinId}}`;
            
            // 查找所有包含此蛋白质的蛋白对
            const relatedPairs = allData.filter(row => 
                row.protein1 === proteinId || row.protein2 === proteinId
            );
            
            let content = `
                <div class="protein-info">
                    <h4>基本信息</h4>
                    <p><strong>蛋白质ID:</strong> ${{proteinId}}</p>
                    <p><strong>相关蛋白对数量:</strong> ${{relatedPairs.length}}</p>
            `;
            
            // 显示序列信息（如果可用）
            if (proteinSequences[proteinId]) {{
                const sequence = proteinSequences[proteinId];
                content += `
                    <p><strong>序列长度:</strong> ${{sequence.length}} 氨基酸</p>
                    <div class="sequence-viewer">
                        <div class="sequence-header">
                            <span class="sequence-title">蛋白质序列</span>
                        </div>
                        <div class="sequence-display">${{formatSequence(sequence)}}</div>
                    </div>
                `;
            }}
            
            content += `
                </div>
                <div class="related-pairs">
                    <h4>相关蛋白对 (${{relatedPairs.length}} 个)</h4>
                    <table class="cis-table">
                        <thead>
                            <tr>
                                <th>蛋白对</th>
                                <th>集成概率</th>
                                <th>预测结果</th>
                                <th>操作</th>
                            </tr>
                        </thead>
                        <tbody>
            `;
            
            relatedPairs.forEach(pair => {{
                const otherProtein = pair.protein1 === proteinId ? pair.protein2 : pair.protein1;
                const hasInterpretability = interpretabilityData[`${{pair.protein1}}_${{pair.protein2}}`] ? '' : 'disabled';
                
                content += `
                    <tr>
                        <td>${{pair.protein1}} - ${{pair.protein2}}</td>
                        <td>${{parseFloat(pair.ensemble_probability).toFixed(3)}}</td>
                        <td>
                            <span class="prediction-${{parseInt(pair.ensemble_prediction) === 1 ? 'positive' : 'negative'}}">
                                ${{parseInt(pair.ensemble_prediction) === 1 ? '阳性' : '阴性'}}
                            </span>
                        </td>
                        <td>
                            <button class="interpretability-btn" ${{hasInterpretability}} 
                                    onclick="showInterpretability('${{pair.protein1}}', '${{pair.protein2}}')">
                                🔍 查看分析
                            </button>
                        </td>
                    </tr>
                `;
            }});
            
            content += `
                        </tbody>
                    </table>
                </div>
            `;
            
            body.innerHTML = content;
            modal.style.display = 'block';
        }}

        // 格式化序列显示
        function formatSequence(sequence) {{
            return sequence.match(/.{{1,80}}/g).join('<br>');
        }}

        // 显示可解释性分析
        function showInterpretability(protein1, protein2) {{
            const modal = document.getElementById('interpretabilityModal');
            const title = document.getElementById('modalTitle');
            const body = document.getElementById('modalBody');
            
            title.textContent = `可解释性分析: ${{protein1}} - ${{protein2}}`;
            
            const key = `${{protein1}}_${{protein2}}`;
            const data = interpretabilityData[key];
            
            if (!data) {{
                body.innerHTML = '<p>该蛋白对没有可解释性分析数据。</p>';
                modal.style.display = 'block';
                return;
            }}
            
            let content = '';
            
            // 残基级注意力权重可视化
            if (data.attention) {{
                content += generateAttentionVisualization(data.attention);
            }}
            
            // CIS可解释性分析
            if (data.cis) {{
                content += generateCisVisualization(data.cis);
            }}
            
            body.innerHTML = content;
            modal.style.display = 'block';
            
            // 如果有注意力数据，初始化颜色
            if (data.attention) {{
                const modalId = body.querySelector('[id^="attention_"]')?.id?.split('_')[1];
                if (modalId) {{
                    initializeAttentionColors(`attention_${{modalId}}`);
                }}
            }}
        }}

        // 生成注意力权重可视化
        function generateAttentionVisualization(attentionData) {{
            const modalId = `attention_${{Date.now()}}`;
            let html = `
                <div class="attention-section">
                    <h4>残基级注意力权重分析</h4>
                    <p>注意力头数: ${{attentionData.attention_heads}}</p>
                    
                    <div class="attention-controls">
                        <label>
                            <input type="checkbox" id="logScale_${{modalId}}" onchange="toggleLogScale('${{modalId}}')">
                            对数缩放
                        </label>
                        <label>
                            热图颜色方案:
                            <select id="colorScheme_${{modalId}}" onchange="changeColorScheme('${{modalId}}')">
                                <option value="heat">红色热图</option>
                                <option value="viridis">Viridis</option>
                                <option value="plasma">Plasma</option>
                                <option value="blue">蓝色渐变</option>
                            </select>
                        </label>
                    </div>
                    
                    <div class="attention-legend" id="legend_${{modalId}}">
                        <span>注意力强度:</span>
                        <div class="legend-gradient" id="legendGradient_${{modalId}}"></div>
                        <div class="legend-labels">
                            <span>低</span>
                            <span>高</span>
                        </div>
                    </div>
            `;
            
            // 蛋白质1的注意力权重
            if (attentionData.protein1_attention && Object.keys(attentionData.protein1_attention).length > 0) {{
                const protein1Seq = proteinSequences[attentionData.protein1_id] || '';
                html += generateProteinAttentionView(
                    attentionData.protein1_id, 
                    protein1Seq, 
                    attentionData.protein1_attention,
                    attentionData.protein1_length,
                    modalId,
                    1
                );
            }}
            
            // 蛋白质2的注意力权重
            if (attentionData.protein2_attention && Object.keys(attentionData.protein2_attention).length > 0) {{
                const protein2Seq = proteinSequences[attentionData.protein2_id] || '';
                html += generateProteinAttentionView(
                    attentionData.protein2_id, 
                    protein2Seq, 
                    attentionData.protein2_attention,
                    attentionData.protein2_length,
                    modalId,
                    2
                );
            }}
            
            html += '</div>';
            return html;
        }}

        // 生成单个蛋白质的注意力权重视图
        function generateProteinAttentionView(proteinId, sequence, attentionWeights, length, modalId, proteinIndex) {{
            const heads = Object.keys(attentionWeights);
            const firstHead = heads[0];
            const weights = attentionWeights[firstHead] || [];
            const lineLength = 60; // 每行显示60个残基，更接近FASTA格式
            const viewerId = `${{proteinId}}_${{modalId}}_${{proteinIndex}}`;
            
            let html = `
                <div class="sequence-viewer" id="viewer_${{viewerId}}">
                    <div class="sequence-header">
                        <span class="sequence-title">${{proteinId}}</span>
                        <span class="sequence-length">长度: ${{length}} 氨基酸</span>
                    </div>
                    
                    <!-- 注意力头选择器 -->
                    <div class="attention-head-selector">
                        <label>
                            注意力头选择:
                            <select id="headSelector_${{viewerId}}" onchange="changeAttentionHead('${{viewerId}}', '${{modalId}}')">
                                <option value="average">平均值</option>
                                <option value="max">最大值池化</option>`;
            
            // 添加各个注意力头的选项
            heads.forEach((head, index) => {{
                html += `<option value="${{head}}">Head ${{index + 1}}</option>`;
            }});
            
            html += `
                            </select>
                        </label>
                    </div>
                    
                    <!-- 序列复制按钮 -->
                    <div class="sequence-copy-container">
                                            <button class="copy-sequence-btn" onclick="copySequenceToClipboard('${{viewerId}}')">
                        复制序列
                    </button>
                        <span class="copy-feedback" id="copyFeedback_${{viewerId}}">已复制!</span>
                    </div>
                    
                    <div class="sequence-display-fasta" id="sequenceDisplay_${{viewerId}}" data-sequence="${{sequence}}" data-attention='${{JSON.stringify(attentionWeights)}}'>
            `;
            
            // 计算初始权重（默认使用第一个头）
            if (weights && weights.length > 0) {{
                html += generateSequenceLines(viewerId, sequence, weights, lineLength, modalId);
            }} else {{
                html += '<p>无注意力权重数据</p>';
            }}
            
            html += `
                    </div>
                </div>
            `;
            
            return html;
        }}

        // 生成序列行（FASTA格式）
        function generateSequenceLines(viewerId, sequence, weights, lineLength, modalId) {{
            if (!sequence || weights.length === 0) {{
                return '<p>无序列或注意力数据</p>';
            }}
            
            const minWeight = Math.min(...weights);
            const maxWeight = Math.max(...weights);
            const seqLength = Math.min(sequence.length, weights.length);
            let html = '';
            
            // 按行分割序列（类似FASTA格式）
            for (let lineStart = 0; lineStart < seqLength; lineStart += lineLength) {{
                const lineEnd = Math.min(lineStart + lineLength, seqLength);
                
                // 为每行添加位置标记行
                html += '<div class="sequence-position-header">';
                for (let i = lineStart; i < lineEnd; i++) {{
                    if ((i - lineStart) % 10 === 9 && i + 1 <= seqLength) {{
                        // 位置标记应该在第10、20、30...个氨基酸上方
                        const pos = i + 1;
                        html += `<span class="position-marker">${{pos}}</span>`;
                    }} else {{
                        html += '<span class="position-marker"></span>';
                    }}
                    // 每10个氨基酸后添加空白间距
                    if ((i - lineStart + 1) % 10 === 0 && i < lineEnd - 1) {{
                        html += '<span class="sequence-spacer"></span>';
                    }}
                }}
                html += '</div>';
                
                // 序列行
                html += '<div class="sequence-fasta-line">';
                for (let i = lineStart; i < lineEnd; i++) {{
                    const residue = sequence[i];
                    const weight = weights[i];
                    const intensity = maxWeight > minWeight ? (weight - minWeight) / (maxWeight - minWeight) : 0;
                    
                    html += `<span class="attention-residue-fasta" 
                                   data-intensity="${{intensity}}"
                                   data-weight="${{weight}}"
                                   data-position="${{i + 1}}"
                                   data-modal="${{modalId}}"
                                   data-viewer="${{viewerId}}"
                                   title="残基: ${{residue}} (位置: ${{i + 1}}), 注意力: ${{weight.toFixed(4)}}">${{residue}}</span>`;
                    
                    // 每10个氨基酸后添加空白间距
                    if ((i - lineStart + 1) % 10 === 0 && i < lineEnd - 1) {{
                        html += '<span class="sequence-spacer"></span>';
                    }}
                }}
                html += '</div>';
            }}
            
            return html;
        }}
        
        // 更改注意力头
        function changeAttentionHead(viewerId, modalId) {{
            const selector = document.getElementById(`headSelector_${{viewerId}}`);
            const selectedHead = selector.value;
            const sequenceDisplay = document.getElementById(`sequenceDisplay_${{viewerId}}`);
            
            if (!sequenceDisplay) return;
            
            const sequence = sequenceDisplay.getAttribute('data-sequence');
            const attentionData = JSON.parse(sequenceDisplay.getAttribute('data-attention'));
            const heads = Object.keys(attentionData);
            
            let weights = [];
            
            if (selectedHead === 'average') {{
                // 计算平均值
                const numHeads = heads.length;
                const maxLength = Math.max(...heads.map(h => attentionData[h].length));
                weights = new Array(maxLength).fill(0);
                
                heads.forEach(head => {{
                    const headWeights = attentionData[head];
                    headWeights.forEach((weight, i) => {{
                        weights[i] += weight / numHeads;
                    }});
                }});
            }} else if (selectedHead === 'max') {{
                // 计算最大值池化
                const maxLength = Math.max(...heads.map(h => attentionData[h].length));
                weights = new Array(maxLength).fill(-Infinity);
                
                heads.forEach(head => {{
                    const headWeights = attentionData[head];
                    headWeights.forEach((weight, i) => {{
                        weights[i] = Math.max(weights[i], weight);
                    }});
                }});
            }} else {{
                // 选择特定的头
                weights = attentionData[selectedHead] || [];
            }}
            
            // 重新生成序列显示
            const lineLength = 60;
            const newHtml = generateSequenceLines(viewerId, sequence, weights, lineLength, modalId);
            sequenceDisplay.innerHTML = newHtml;
            
            // 重新应用颜色
            const logCheckbox = document.getElementById(`logScale_${{modalId}}`);
            const colorSelect = document.getElementById(`colorScheme_${{modalId}}`);
            if (logCheckbox && colorSelect) {{
                applyColorsToResidues(modalId, logCheckbox.checked, colorSelect.value);
            }}
        }}
        
        // 复制序列到剪贴板（仅序列，不包含位置信息）
        function copySequenceToClipboard(viewerId) {{
            const sequenceDisplay = document.getElementById(`sequenceDisplay_${{viewerId}}`);
            if (!sequenceDisplay) return;
            
            const sequence = sequenceDisplay.getAttribute('data-sequence');
            if (!sequence) return;
            
            // 创建临时文本区域
            const textArea = document.createElement('textarea');
            textArea.value = sequence;
            document.body.appendChild(textArea);
            textArea.select();
            
            try {{
                document.execCommand('copy');
                // 显示反馈
                const feedback = document.getElementById(`copyFeedback_${{viewerId}}`);
                if (feedback) {{
                    feedback.classList.add('show');
                    setTimeout(() => {{
                        feedback.classList.remove('show');
                    }}, 2000);
                }}
            }} catch (err) {{
                console.error('复制失败:', err);
            }}
            
            document.body.removeChild(textArea);
        }}

        // 颜色方案
        const colorSchemes = {{
            heat: (intensity) => {{
                const red = Math.round(intensity * 255);
                return `rgb(${{red}}, 0, 0)`;
            }},
            viridis: (intensity) => {{
                // Viridis颜色方案：紫蓝绿黄渐变
                if (intensity < 0.25) {{
                    // 紫色到蓝色 (0-0.25)
                    const t = intensity / 0.25;
                    const r = Math.round(68 + t * (55 - 68));
                    const g = Math.round(1 + t * (119 - 1));
                    const b = Math.round(84 + t * (176 - 84));
                    return `rgb(${{r}}, ${{g}}, ${{b}})`;
                }} else if (intensity < 0.5) {{
                    // 蓝色到绿色 (0.25-0.5)
                    const t = (intensity - 0.25) / 0.25;
                    const r = Math.round(55 + t * (53 - 55));
                    const g = Math.round(119 + t * (183 - 119));
                    const b = Math.round(176 + t * (121 - 176));
                    return `rgb(${{r}}, ${{g}}, ${{b}})`;
                }} else if (intensity < 0.75) {{
                    // 绿色到黄色 (0.5-0.75)
                    const t = (intensity - 0.5) / 0.25;
                    const r = Math.round(53 + t * (253 - 53));
                    const g = Math.round(183 + t * (231 - 183));
                    const b = Math.round(121 + t * (37 - 121));
                    return `rgb(${{r}}, ${{g}}, ${{b}})`;
                }} else {{
                    // 黄色 (0.75-1.0)
                    const t = (intensity - 0.75) / 0.25;
                    const r = Math.round(253 + t * (254 - 253));
                    const g = Math.round(231 + t * (232 - 231));
                    const b = Math.round(37 + t * (36 - 37));
                    return `rgb(${{r}}, ${{g}}, ${{b}})`;
                }}
            }},
            plasma: (intensity) => {{
                // Plasma颜色方案：紫粉橙黄渐变
                if (intensity < 0.33) {{
                    // 紫色到粉色 (0-0.33)
                    const t = intensity / 0.33;
                    const r = Math.round(13 + t * (135 - 13));
                    const g = Math.round(8 + t * (46 - 8));
                    const b = Math.round(135 + t * (213 - 135));
                    return `rgb(${{r}}, ${{g}}, ${{b}})`;
                }} else if (intensity < 0.66) {{
                    // 粉色到橙色 (0.33-0.66)
                    const t = (intensity - 0.33) / 0.33;
                    const r = Math.round(135 + t * (221 - 135));
                    const g = Math.round(46 + t * (72 - 46));
                    const b = Math.round(213 + t * (93 - 213));
                    return `rgb(${{r}}, ${{g}}, ${{b}})`;
                }} else {{
                    // 橙色到黄色 (0.66-1.0)
                    const t = (intensity - 0.66) / 0.34;
                    const r = Math.round(221 + t * (240 - 221));
                    const g = Math.round(72 + t * (249 - 72));
                    const b = Math.round(93 + t * (33 - 93));
                    return `rgb(${{r}}, ${{g}}, ${{b}})`;
                }}
            }},
            blue: (intensity) => {{
                const blue = Math.round(intensity * 255);
                return `rgb(0, 0, ${{blue}})`;
            }}
        }};

        // 应用颜色到残基
        function applyColorsToResidues(modalId, useLog = false, scheme = 'heat') {{
            const viewer = document.querySelectorAll(`[data-modal="${{modalId}}"]`);
            viewer.forEach(residue => {{
                let intensity = parseFloat(residue.getAttribute('data-intensity'));
                const weight = parseFloat(residue.getAttribute('data-weight'));
                
                if (useLog && weight > 0) {{
                    // 对数缩放
                    intensity = Math.log(weight + 1) / Math.log(2); // log2(weight + 1)
                    intensity = Math.max(0, Math.min(1, intensity)); // 归一化到[0,1]
                }}
                
                const color = colorSchemes[scheme](intensity);
                // 使用文字颜色而不是背景色
                residue.style.color = color;
                residue.style.backgroundColor = 'transparent';
                residue.style.fontWeight = intensity > 0.5 ? 'bold' : 'normal';
            }});
            
            // 更新图例
            updateLegend(modalId, scheme);
        }}

        // 更新图例
        function updateLegend(modalId, scheme) {{
            const legendGradient = document.getElementById(`legendGradient_${{modalId}}`);
            if (legendGradient) {{
                const steps = 10;
                let gradientStops = [];
                for (let i = 0; i <= steps; i++) {{
                    const intensity = i / steps;
                    const color = colorSchemes[scheme](intensity);
                    gradientStops.push(`${{color}} ${{i * 10}}%`);
                }}
                legendGradient.style.background = `linear-gradient(to right, ${{gradientStops.join(', ')}})`;
                legendGradient.style.height = '20px';
                legendGradient.style.width = '200px';
                legendGradient.style.border = '1px solid #ccc';
            }}
        }}

        // 控制函数
        function toggleLogScale(modalId) {{
            const logCheckbox = document.getElementById(`logScale_${{modalId}}`);
            const colorSelect = document.getElementById(`colorScheme_${{modalId}}`);
            applyColorsToResidues(modalId, logCheckbox.checked, colorSelect.value);
        }}

        function changeColorScheme(modalId) {{
            const logCheckbox = document.getElementById(`logScale_${{modalId}}`);
            const colorSelect = document.getElementById(`colorScheme_${{modalId}}`);
            applyColorsToResidues(modalId, logCheckbox.checked, colorSelect.value);
        }}

        // 在模态框显示后初始化颜色
        function initializeAttentionColors(modalId) {{
            setTimeout(() => {{
                applyColorsToResidues(modalId, false, 'heat');
            }}, 100);
        }}

        // 生成CIS可解释性可视化
        function generateCisVisualization(cisData) {{
            let html = `
                <div class="cis-section">
                    <h4>CIS可解释性分析</h4>
                    <p><strong>查询蛋白对:</strong> ${{cisData.protein1_id}} - ${{cisData.protein2_id}}</p>
            `;
            
            if (cisData.prediction_score !== undefined) {{
                html += `<p><strong>预测分数:</strong> ${{cisData.prediction_score.toFixed(3)}}</p>`;
            }}
            
            if (cisData.prediction_label !== undefined) {{
                html += `<p><strong>预测标签:</strong> ${{cisData.prediction_label === 1 ? '阳性' : '阴性'}}</p>`;
            }}
            
            if (cisData.top_similar_training_samples && cisData.top_similar_training_samples.length > 0) {{
                html += `
                    <h5>最相似的训练样本 (Top ${{cisData.top_similar_training_samples.length}})</h5>
                    <table class="cis-table">
                        <thead>
                            <tr>
                                <th>排名</th>
                                <th>蛋白对</th>
                                <th>相似度分数</th>
                                <th>距离</th>
                                <th>标签</th>
                            </tr>
                        </thead>
                        <tbody>
                `;
                
                cisData.top_similar_training_samples.forEach((sample, index) => {{
                    // 兼容不同的数据结构
                    const protein1 = sample.protein1 || (sample.training_pair && sample.training_pair[0]) || 'N/A';
                    const protein2 = sample.protein2 || (sample.training_pair && sample.training_pair[1]) || 'N/A';
                    const similarity = sample.similarity_score || sample.similarity || 0;
                    const distance = sample.distance || (1 - similarity); // 如果没有distance，用1-similarity估算
                    const label = sample.label !== undefined ? sample.label : (sample.is_positive !== undefined ? (sample.is_positive ? 1 : 0) : -1);
                    
                    // 格式化科学计数法显示
                    const formatNumber = (num) => {{
                        if (num === null || num === undefined || isNaN(num)) return 'N/A';
                        if (num < 0.001 || num > 1000) {{
                            return num.toExponential(3);
                        }} else {{
                            return num.toFixed(4);
                        }}
                    }};
                    
                    // 标签显示
                    let labelHtml = 'N/A';
                    if (label === 1) {{
                        labelHtml = '<span class="positive-label">阳性</span>';
                    }} else if (label === 0) {{
                        labelHtml = '<span class="negative-label">阴性</span>';
                    }}
                    
                    html += `
                        <tr>
                            <td>${{index + 1}}</td>
                            <td>${{protein1}} - ${{protein2}}</td>
                            <td class="similarity-score scientific-notation">${{formatNumber(similarity)}}</td>
                            <td class="scientific-notation">${{formatNumber(distance)}}</td>
                            <td>${{labelHtml}}</td>
                        </tr>
                    `;
                }});
                
                html += `
                        </tbody>
                    </table>
                `;
            }} else {{
                html += '<p>无相似训练样本数据</p>';
            }}
            
            if (cisData.error) {{
                html += `<p style="color: var(--error-color);"><strong>错误:</strong> ${{cisData.error}}</p>`;
            }}
            
            html += '</div>';
            return html;
        }}
        """ 


def main():
    """命令行入口函数"""
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description='生成SEPAL-PPI预测结果HTML报告')
    parser.add_argument('output_dir', help='结果输出目录路径')
    parser.add_argument('--config', '-c', help='配置文件路径')
    parser.add_argument('--verbose', '-v', action='store_true', help='详细输出')
    
    args = parser.parse_args()
    
    # 设置日志
    log_level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    try:
        # 创建HTML生成器
        html_gen = HTMLGenerator(args.output_dir, args.config, logger)
        
        # 从输出目录加载结果数据
        results = html_gen._load_results_from_dir()
        
        if not results:
            logger.error("未找到有效的预测结果数据")
            sys.exit(1)
        
        # 生成HTML报告
        html_path = html_gen.generate_prediction_results_html(results)
        print(f"HTML报告已生成: {html_path}")
        
    except Exception as e:
        logger.error(f"生成HTML报告失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 