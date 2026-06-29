# app/core/hardware.py
"""硬件检测与模式管理模块

自动检测最优计算设备（CUDA GPU > OpenVINO 核显 > CPU），
支持两种运行模式：
- pure_cloud：纯云端模式，所有 AI 任务通过 API 完成
- hybrid_acceleration：混合加速模式，本地 GPU 处理预处理/检索任务

云端 API（Embedding/Reranker/LLM）始终不变，
本地 GPU 仅加速 OCR、向量检索等预处理环节。
"""
from __future__ import annotations

import logging
from typing import Any, Literal

logger = logging.getLogger(__name__)

DeviceType = Literal["cuda", "openvino_gpu", "cpu"]
AccelerationMode = Literal["pure_cloud", "hybrid_acceleration"]

__all__ = [
    "CURRENT_DEVICE",
    "DeviceType",
    "AccelerationMode",
    "HardwareManager",
    "hardware_manager",
]


class HardwareManager:
    """硬件检测与模式管理器
    
    自动检测硬件环境，确定加速模式，提供可用优化列表。
    
    加速模式说明：
    - pure_cloud：纯云端模式，所有 AI 任务通过 API 完成
    - hybrid_acceleration：混合加速模式，本地 GPU 加速预处理/检索
    
    本地 GPU 加速的环节（不影响云端 API 调用）：
    - OCR 识别加速（PaddleOCR GPU 版）
    - 向量检索加速（GPU 余弦相似度计算）
    - 可选：本地 Reranker 精排（4GB 显存可运行 bge-reranker-base）
    
    注意：文本清洗/标准化为纯 CPU Python 字符串操作，无 GPU 加速。
    """
    
    # 4GB 显存阈值（字节）
    _MIN_GPU_MEMORY_BYTES: int = 4 * 1024 * 1024 * 1024
    
    def __init__(self) -> None:
        self._device: DeviceType = self._detect_device()
        self._mode: AccelerationMode = self._determine_mode()
        self._gpu_name: str = ""
        self._gpu_memory_gb: float = 0.0
        
        if self._device == "cuda":
            try:
                import torch
                self._gpu_name = torch.cuda.get_device_name(0)
                self._gpu_memory_gb = (
                    torch.cuda.get_device_properties(0).total_memory / (1024**3)
                )
                logger.info(
                    "GPU 信息：%s，显存：%.1f GB",
                    self._gpu_name,
                    self._gpu_memory_gb,
                )
            except Exception:
                pass
        
        logger.info(
            "硬件管理器初始化完成：device=%s, mode=%s",
            self._device,
            self._mode,
        )
    
    @property
    def device(self) -> DeviceType:
        """当前计算设备"""
        return self._device
    
    @property
    def mode(self) -> AccelerationMode:
        """当前加速模式"""
        return self._mode
    
    @property
    def gpu_name(self) -> str:
        """GPU 名称（仅 CUDA 模式有值）"""
        return self._gpu_name
    
    @property
    def gpu_memory_gb(self) -> float:
        """GPU 显存大小（GB，仅 CUDA 模式有值）"""
        return self._gpu_memory_gb
    
    def _detect_device(self) -> DeviceType:
        """检测最优计算设备
        
        检测顺序：
        1. NVIDIA GPU（CUDA）
        2. Intel 核显（OpenVINO GPU）
        3. CPU 回退
        
        Returns:
            设备类型字符串
        """
        # Step 1: 检测 CUDA
        try:
            import torch
            if torch.cuda.is_available():
                device_name = torch.cuda.get_device_name(0)
                logger.info("检测到 NVIDIA GPU：%s，启用 CUDA 加速。", device_name)
                return "cuda"
        except ImportError:
            logger.debug("PyTorch 未安装，跳过 CUDA 检测。")
        except Exception as e:
            logger.warning("CUDA 检测失败：%s", e)
        
        # Step 2: 检测 OpenVINO 核显
        try:
            import openvino as ov
            core = ov.Core()
            if "GPU" in core.available_devices:
                logger.info("检测到 Intel 核显，启用 OpenVINO GPU 加速。")
                return "openvino_gpu"
        except ImportError:
            logger.debug("OpenVINO 未安装，跳过核显检测。")
        except Exception as e:
            logger.warning("OpenVINO 检测失败：%s", e)
        
        # Step 3: 回退到 CPU
        logger.info("未检测到可用 GPU，回退至 CPU 模式。")
        return "cpu"
    
    def _determine_mode(self) -> AccelerationMode:
        """确定加速模式
        
        CUDA GPU 且显存 >= 4GB 时启用混合加速模式，
        否则使用纯云端模式。
        """
        if self._device == "cuda":
            try:
                import torch
                gpu_memory = torch.cuda.get_device_properties(0).total_memory
                if gpu_memory >= self._MIN_GPU_MEMORY_BYTES:
                    logger.info(
                        "GPU 显存 %.1f GB >= 4GB，启用混合加速模式。",
                        gpu_memory / (1024**3),
                    )
                    return "hybrid_acceleration"
                else:
                    logger.info(
                        "GPU 显存 %.1f GB < 4GB，使用纯云端模式。",
                        gpu_memory / (1024**3),
                    )
            except Exception:
                pass
        elif self._device == "openvino_gpu":
            logger.info("OpenVINO 核显可用，启用混合加速模式。")
            return "hybrid_acceleration"
        
        return "pure_cloud"
    
    def get_available_optimizations(self) -> list[str]:
        """获取当前硬件环境下可用的优化功能列表
        
        Returns:
            优化功能标识列表
        """
        optimizations: list[str] = []
        
        if self._device == "cuda" and self._mode == "hybrid_acceleration":
            optimizations.extend([
                "ocr_acceleration",           # OCR 识别加速
                "vector_search_acceleration",  # 向量检索加速
                "local_reranker_acceleration", # 本地精排加速（可选）
            ])
        elif self._device == "openvino_gpu" and self._mode == "hybrid_acceleration":
            optimizations.extend([
                "ocr_acceleration",
                "vector_search_acceleration",
            ])
        
        return optimizations
    
    def get_optimization_config(self) -> dict[str, Any]:
        """获取完整的优化配置信息
        
        Returns:
            包含设备、模式、可用优化的配置字典
        """
        return {
            "device": self._device,
            "mode": self._mode,
            "gpu_name": self._gpu_name,
            "gpu_memory_gb": round(self._gpu_memory_gb, 1),
            "available_optimizations": self.get_available_optimizations(),
        }
    
    def is_hybrid_acceleration(self) -> bool:
        """是否启用混合加速模式"""
        return self._mode == "hybrid_acceleration"
    
    def supports_ocr_acceleration(self) -> bool:
        """是否支持 OCR 加速"""
        return "ocr_acceleration" in self.get_available_optimizations()
    
    def supports_vector_search_acceleration(self) -> bool:
        """是否支持向量检索加速"""
        return "vector_search_acceleration" in self.get_available_optimizations()
    
    def supports_local_reranker(self) -> bool:
        """是否支持本地 Reranker 加速"""
        return "local_reranker_acceleration" in self.get_available_optimizations()


# 全局硬件管理器单例
hardware_manager: HardwareManager = HardwareManager()

# 向后兼容：保留 CURRENT_DEVICE 全局变量
CURRENT_DEVICE: DeviceType = hardware_manager.device
