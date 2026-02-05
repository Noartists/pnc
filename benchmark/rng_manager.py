"""
RNG 管理器

统一管理随机数生成，保证实验可复现性。

使用 SeedSequence.spawn() 为不同模块创建独立的子 RNG，
避免因调用顺序变化导致的"蝴蝶效应"。

典型用法:
    rng = RNGManager(seed=42)
    scene = make_scene(cfg, rng.rng_env)
    x0 = sample_initial(cfg, rng.rng_init)
    wind = make_wind(cfg, rng.rng_wind)
"""

from numpy.random import SeedSequence, Generator, default_rng
from typing import Optional
from dataclasses import dataclass


@dataclass
class RNGManager:
    """
    统一随机数管理器
    
    属性:
        seed: 主种子
        rng_init: 初始状态采样用 RNG（位置噪声、航向噪声）
        rng_env: 场景生成用 RNG（障碍物随机化、起终点随机化）
        rng_wind: 风场采样用 RNG（预留）
        rng_noise: 传感器/过程噪声用 RNG（预留）
    """
    seed: int
    rng_init: Generator
    rng_env: Generator
    rng_wind: Generator
    rng_noise: Generator
    
    def __init__(self, seed: int):
        """
        初始化 RNG 管理器
        
        参数:
            seed: 主种子（整数）
        """
        self.seed = seed
        
        # 使用 SeedSequence 创建独立的子序列
        ss = SeedSequence(seed)
        child_seeds = ss.spawn(4)
        
        # 为不同模块分配独立的 RNG
        self.rng_init = default_rng(child_seeds[0])   # 初始状态
        self.rng_env = default_rng(child_seeds[1])    # 场景/环境
        self.rng_wind = default_rng(child_seeds[2])   # 风场
        self.rng_noise = default_rng(child_seeds[3])  # 噪声
    
    def get_init_rng(self) -> Generator:
        """获取初始状态采样用 RNG"""
        return self.rng_init
    
    def get_env_rng(self) -> Generator:
        """获取场景生成用 RNG"""
        return self.rng_env
    
    def get_wind_rng(self) -> Generator:
        """获取风场采样用 RNG"""
        return self.rng_wind
    
    def get_noise_rng(self) -> Generator:
        """获取噪声用 RNG"""
        return self.rng_noise
    
    def sample_position_noise(self, 
                               x_range: tuple = (-20, 20),
                               y_range: tuple = (-20, 20),
                               z_range: tuple = (-10, 10)) -> tuple:
        """
        采样初始位置噪声
        
        参数:
            x_range: x方向噪声范围 (min, max)
            y_range: y方向噪声范围 (min, max)
            z_range: z方向噪声范围 (min, max)
        
        返回:
            (dx, dy, dz) 位置噪声元组
        """
        dx = self.rng_init.uniform(*x_range)
        dy = self.rng_init.uniform(*y_range)
        dz = self.rng_init.uniform(*z_range)
        return (dx, dy, dz)
    
    def sample_heading_noise(self, range_rad: float = 0.3) -> float:
        """
        采样初始航向噪声
        
        参数:
            range_rad: 航向噪声范围（弧度），实际范围为 [-range_rad, range_rad]
        
        返回:
            航向噪声（弧度）
        """
        return self.rng_init.uniform(-range_rad, range_rad)
    
    def to_dict(self) -> dict:
        """
        导出为字典（用于保存到 metrics.json）
        
        返回:
            包含 seed 的字典
        """
        return {
            'seed': self.seed
        }
    
    @classmethod
    def from_seed(cls, seed: int) -> 'RNGManager':
        """
        从种子创建 RNG 管理器（工厂方法）
        
        参数:
            seed: 主种子
        
        返回:
            RNGManager 实例
        """
        return cls(seed)


def get_config_hash(configs: dict) -> str:
    """
    计算配置的哈希值
    
    参数:
        configs: 配置字典，例如 {
            'map_config': {...},
            'model_config': {...},
            'planner_config': {...},
            'controller_config': {...},
            'sim_config': {...},
            'disturbance_config': {...}
        }
    
    返回:
        sha256 哈希字符串（前16位）
    """
    import hashlib
    import json
    
    # 将配置序列化为 JSON 字符串（排序键以保证一致性）
    config_str = json.dumps(configs, sort_keys=True, default=str)
    
    # 计算 SHA256 哈希
    hash_obj = hashlib.sha256(config_str.encode('utf-8'))
    
    # 返回前16位（足够唯一标识）
    return hash_obj.hexdigest()[:16]


def get_git_commit() -> Optional[str]:
    """
    获取当前 Git commit hash
    
    返回:
        commit hash（前7位）或 None（如果不在 git 仓库中）
    """
    import subprocess
    import os
    
    try:
        result = subprocess.run(
            ['git', 'rev-parse', '--short', 'HEAD'],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    
    return None
