"""
Market Research & Strategy Development Module.
Architectural choice: Separation of data fetching, technical analysis, and pattern detection
for maintainability and testability. Uses TA-Lib for production-grade indicators.
"""
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import talib
from dataclasses import dataclass
from enum import Enum
import requests
from requests.exceptions import RequestException, Timeout
import time

from config import get_config, get_firestore_client

logger = logging.getLogger(__name__)


class MarketCondition(Enum):
    """Enum for standardized market condition classification."""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    VOLATILE = "volatile"
    BREAKOUT = "breakout"


@dataclass
class MarketOpportunity:
    """Structured opportunity detection result."""
    symbol: str
    condition: MarketCondition
    confidence_score: float
    indicators: Dict[str, float]
    timestamp: datetime
    timeframe: str
    
    def to_firestore_dict(self) -> dict:
        """Convert to Firestore-compatible dictionary."""
        return {
            "symbol": self.symbol,
            "condition": self.condition.value,
            "confidence_score": self.confidence_score,
            "indicators": self.indicators,
            "timestamp": self.t