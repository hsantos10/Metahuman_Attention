"""
Metahuman Attention - Core ML package for attention-based human motion analysis
"""

__version__ = "1.0.0"

try:
    from src.model import (
        ImprovedConvFormer,
        CNN_Network,
        CNN_LSTM_Network,
        LSTM_Network,
        LSTM_Network_with_Attention,
        LightConvFormer,
    )
except ImportError:
    # torch not installed — model classes unavailable but data utils still work
    pass

__all__ = [
    "ImprovedConvFormer",
    "CNN_Network",
    "CNN_LSTM_Network",
    "LSTM_Network",
    "LSTM_Network_with_Attention",
    "LightConvFormer",
]
