"""
Hydra 配置管理器
处理 Hydra 配置加载、变量解析和配置组合
"""

import os
import yaml
import logging
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional, Union

try:
    from omegaconf import OmegaConf, DictConfig, ListConfig
    import hydra
    from hydra.core.hydra_config import HydraConfig
    from hydra.utils import instantiate
    HYDRA_AVAILABLE = True
except ImportError:
    HYDRA_AVAILABLE = False
    # 创建占位符类
    class DictConfig:
        pass
    class ListConfig:
        pass
    class OmegaConf:
        @staticmethod
        def to_container(config, resolve=True):
            return config

logger = logging.getLogger(__name__)


class HydraConfigManager:
    """
    Hydra 配置管理器
    处理配置加载、变量解析和配置组合
    """
    
    def __init__(self, config_dir: str = "config"):
        """
        初始化 Hydra 配置管理器
        
        Args:
            config_dir: 配置文件目录
        """
        # 使用绝对路径，避免后续 Path.relative_to 由于相对/绝对混用导致失败
        self.config_dir = Path(config_dir).resolve()
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
    def load_config(self, config_path: str, overrides: Optional[list] = None) -> DictConfig:
        """
        加载 Hydra 配置
        
        Args:
            config_path: 配置文件路径
            overrides: 配置覆盖列表
            
        Returns:
            DictConfig: 解析后的配置
        """
        try:
            original_input = config_path
            # 去掉可能的前后空白
            config_path = config_path.strip()

            # 允许用户传入:   sepal-ppi/S5/sepal-ppi-nofeature.yaml
            #                sepal-ppi/S5/sepal-ppi-nofeature
            #                S5/sepal-ppi-nofeature.yaml (在 config/sepal-ppi 下递归搜索)
            #                S5/sepal-ppi-nofeature
            #                sepal-ppi-nofeature (如唯一匹配)

            # 补全扩展名
            has_yaml_suffix = config_path.endswith('.yaml') or config_path.endswith('.yml')
            if not has_yaml_suffix:
                search_name = f"{config_path}.yaml"
            else:
                search_name = config_path

            # 构造候选绝对文件路径列表
            candidates: list[Path] = []

            # 1) 直接按相对 self.config_dir 路径
            direct_path = (self.config_dir / search_name).resolve()
            if direct_path.exists():
                candidates.append(direct_path)

            # 2) 如果以 'config/' 开头，去掉前缀再次尝试
            if search_name.startswith('config/'):
                alt = (self.config_dir / search_name[len('config/'):]).resolve()
                if alt.exists() and alt not in candidates:
                    candidates.append(alt)

            # 3) 专门处理主 config 期望所在的 sepal-ppi 子树: 在 config/sepal-ppi/** 下递归搜索文件名匹配
            #    用户需求: 主 config 文件可以在 config/sepal-ppi 下任意子目录
            main_root = self.config_dir / 'sepal-ppi'
            if main_root.exists():
                # 如果用户给了子路径 (包含 '/'), 则尝试直接 under main_root
                # eg: S5/sepal-ppi-nofeature.yaml
                if '/' in search_name and not (self.config_dir / search_name).exists():
                    # 移除可能的前导 'sepal-ppi/' 以免重复
                    rel_part = search_name
                    if rel_part.startswith('sepal-ppi/'):
                        rel_part = rel_part[len('sepal-ppi/'):]
                    candidate = (main_root / rel_part).resolve()
                    if candidate.exists() and candidate not in candidates:
                        candidates.append(candidate)
                # 递归匹配文件名 (当只给了短名或简单子路径失败时)
                if not candidates:
                    file_basename = Path(search_name).name
                    for p in main_root.rglob('*.yaml'):
                        if p.name == file_basename:
                            candidates.append(p.resolve())

            # 去重
            candidates = list(dict.fromkeys(candidates))

            if not candidates:
                raise FileNotFoundError(f"配置文件不存在: {original_input}")

            if len(candidates) > 1:
                # 如果出现多重匹配且用户给的不是一个包含目录的 disambiguated 路径, 报错提示用户指定更精确路径
                if '/' not in original_input and not has_yaml_suffix:
                    cand_list = '\n  - '.join(str(c.relative_to(self.config_dir)) for c in candidates)
                    raise FileExistsError(
                        f"找到多个同名配置文件，请提供更具体的路径 (例如: sepal-ppi/S5/{candidates[0].name}) :\n  - {cand_list}")

            # 采用第一个匹配 (已经确保唯一或用户给了明确路径)
            config_file = candidates[0]

            # 计算 hydra 的 config_name: 需要相对 self.config_dir 的 POSIX 路径且不带扩展名
            relative_path = config_file.relative_to(self.config_dir)
            config_name = relative_path.with_suffix('').as_posix()

            # 子目录主配置时，对 defaults 做内存级前缀补丁（不落盘持久缓存），生成临时文件供 Hydra 读取
            temp_created: Optional[Path] = None
            try:
                if len(relative_path.parts) > 1:  # 位于子目录
                    with open(config_file, 'r', encoding='utf-8') as f:
                        raw_yaml = yaml.safe_load(f)
                    if isinstance(raw_yaml, dict) and isinstance(raw_yaml.get('defaults'), list):
                        changed = False
                        patched_list = []
                        for item in raw_yaml['defaults']:
                            if isinstance(item, dict) and len(item) == 1:
                                (k, v), = item.items()
                                if isinstance(k, str) and not k.startswith(('/', 'hydra/', '_')):
                                    patched_list.append({f'/{k}': v})
                                    changed = True
                                else:
                                    patched_list.append(item)
                            elif isinstance(item, str):
                                if not item.startswith(('/', 'hydra/', '_')):
                                    patched_list.append('/' + item)
                                    changed = True
                                else:
                                    patched_list.append(item)
                            else:
                                patched_list.append(item)
                        if changed:
                            patched_yaml = dict(raw_yaml)
                            patched_yaml['defaults'] = patched_list
                            # 写入临时文件（位于 config_dir 下，保证 Hydra search path 能找到）
                            with tempfile.NamedTemporaryFile('w', suffix='.yaml', prefix='__patched__', delete=False, dir=self.config_dir) as tf:
                                yaml.safe_dump(patched_yaml, tf, allow_unicode=True, sort_keys=False)
                                temp_created = Path(tf.name)
                            config_name = temp_created.relative_to(self.config_dir).with_suffix('').as_posix()
                            logger.debug(f"Created temporary patch file for nested root config: {temp_created}")
            except Exception as patch_e:
                logger.warning(f"Patching defaults for nested root config failed, using original file: {patch_e}")

            # 初始化 hydra 并 compose
            config_dir_abs = self.config_dir.resolve()
            with hydra.initialize_config_dir(version_base=None, config_dir=str(config_dir_abs)):
                cfg = hydra.compose(config_name=config_name, overrides=overrides or [])

            # 加载后删除临时文件
            if temp_created and temp_created.exists():
                try:
                    temp_created.unlink()
                except Exception as rm_e:
                    logger.debug(f"Failed to delete temporary patch file (ignorable): {rm_e}")

            # ---- 扁平化异常结构 ----
            # 某些情况下 (嵌套子目录主配置 + 运行期生成) 可能出现根为 '' 或再包一层 'generated'
            try:
                # 空键包装 {"": {...}}
                if isinstance(cfg, DictConfig) and list(cfg.keys()) == ['']:
                    cfg = cfg['']
                # 单层 generated 包装 {generated: {training:..., model:...}}
                # 过去的 .generated 包装已移除，无需再特殊处理
            except Exception as norm_e:
                logger.debug(f"Config flattening skipped: {norm_e}")

            logger.info(f"Hydra config loaded: {relative_path} (keys: {list(cfg.keys())[:8]})")
            return cfg

        except Exception as e:
            logger.error(f"Failed to load Hydra config: {e}")
            raise
    
    def resolve_config(self, config: DictConfig) -> Dict[str, Any]:
        """
        解析 Hydra 配置，处理变量引用
        
        Args:
            config: Hydra 配置对象
            
        Returns:
            Dict: 解析后的配置字典
        """
        try:
            # 使用 OmegaConf 解析配置
            resolved_config = OmegaConf.to_container(config, resolve=True)
            
            # 处理嵌套的配置引用
            resolved_config = self._resolve_nested_references(resolved_config, config)
            
            logger.debug("Config resolution completed")
            return resolved_config
            
        except Exception as e:
            logger.error(f"Config resolution failed: {e}")
            raise
    
    def _resolve_nested_references(self, config_dict: Any, original_config: DictConfig) -> Any:
        """
        解析嵌套的配置引用
        
        Args:
            config_dict: 配置字典或其他类型
            original_config: 原始 Hydra 配置
            
        Returns:
            Any: 解析后的配置
        """
        if isinstance(config_dict, dict):
            resolved_dict = {}
            for key, value in config_dict.items():
                if isinstance(value, dict):
                    resolved_dict[key] = self._resolve_nested_references(value, original_config)
                elif isinstance(value, list):
                    resolved_dict[key] = [
                        self._resolve_nested_references(item, original_config) if isinstance(item, dict) else item
                        for item in value
                    ]
                else:
                    resolved_dict[key] = value
            return resolved_dict
        return config_dict
    
    def create_model_config(self, config: DictConfig) -> Dict[str, Any]:
        """
        从 Hydra 配置创建模型配置
        
        Args:
            config: Hydra 配置对象
            
        Returns:
            Dict: 模型配置字典
        """
        try:
            # 解析配置
            resolved_config = self.resolve_config(config)
            
            # 提取模型配置
            model_config = resolved_config.get('model', {})
            
            # 如果指定了架构文件，加载架构配置
            if 'model_architecture_file' in model_config:
                architecture_file = model_config['model_architecture_file']
                architecture_config = self._load_architecture_config(architecture_file, model_config)
                model_config['architecture'] = architecture_config
            
            logger.info("Model config created")
            return model_config
            
        except Exception as e:
            logger.error(f"Failed to create model config: {e}")
            raise
    
    def _load_architecture_config(self, architecture_file: str, model_config: Dict) -> Dict[str, Any]:
        """
        加载架构配置文件
        
        Args:
            architecture_file: 架构配置文件路径
            model_config: 模型配置
            
        Returns:
            Dict: 架构配置字典
        """
        try:
            architecture_path = Path(architecture_file)
            if not architecture_path.exists():
                raise FileNotFoundError(f"架构配置文件不存在: {architecture_file}")
            
            # 读取架构配置
            with open(architecture_path, 'r', encoding='utf-8') as f:
                architecture_config = yaml.safe_load(f)
            
            # 解析架构配置中的变量引用
            architecture_config = self._resolve_architecture_references(architecture_config, model_config)
            
            logger.debug(f"Architecture config loaded: {architecture_file}")
            return architecture_config
            
        except Exception as e:
            logger.error(f"Failed to load architecture config: {e}")
            raise
    
    def _resolve_architecture_references(self, architecture_config: Dict, model_config: Dict) -> Any:
        """
        解析架构配置中的变量引用
        
        Args:
            architecture_config: 架构配置
            model_config: 模型配置
            
        Returns:
            Any: 解析后的架构配置
        """
        def resolve_value(value, context):
            if isinstance(value, str) and value.startswith('${') and value.endswith('}'):
                # 提取变量路径
                var_path = value[2:-1]
                return self._get_nested_value(context, var_path)
            elif isinstance(value, dict):
                return {k: resolve_value(v, context) for k, v in value.items()}
            elif isinstance(value, list):
                return [resolve_value(v, context) for v in value]
            else:
                return value
        
        # 创建上下文，包含模型配置
        context = {'model': model_config}
        
        # 解析架构配置
        resolved_config = resolve_value(architecture_config, context)
        
        return resolved_config
    
    def _get_nested_value(self, config: Dict, path: str) -> Any:
        """
        获取嵌套配置值
        
        Args:
            config: 配置字典
            path: 路径字符串，如 'model.embedding_dim'
            
        Returns:
            Any: 配置值
        """
        keys = path.split('.')
        value = config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                raise KeyError(f"Config path not found: {path}")
        
        return value
    
    def save_config(self, config: DictConfig, output_path: str):
        """
        保存配置到文件
        
        Args:
            config: Hydra 配置对象
            output_path: 输出文件路径
        """
        try:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # 转换为字典并保存
            config_dict = OmegaConf.to_container(config, resolve=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True, indent=2)
            
            logger.info(f"Config saved to: {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save config: {e}")
            raise
    
    def validate_config(self, config: DictConfig) -> bool:
        """
        验证配置完整性
        
        Args:
            config: Hydra 配置对象
            
        Returns:
            bool: 配置是否有效
        """
        try:
            resolved_config = self.resolve_config(config)
            
            # 检查必需字段
            required_fields = [
                'training.epochs',
                'training.batch_size',
                'training.learning_rate',
                'model.embedding_dim',
                'data.embedding_file'
            ]
            
            for field in required_fields:
                if not self._get_nested_value(resolved_config, field):
                    logger.error(f"Missing required field: {field}")
                    return False
            
            logger.info("Config validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Config validation failed: {e}")
            return False


def create_hydra_config_manager(config_dir: str = "config") -> HydraConfigManager:
    """
    创建 Hydra 配置管理器实例
    
    Args:
        config_dir: 配置文件目录
        
    Returns:
        HydraConfigManager: 配置管理器实例
    """
    return HydraConfigManager(config_dir) 