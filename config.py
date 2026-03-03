"""
Configuration module with environment validation and Firebase initialization.
Architectural choice: Pydantic for runtime type safety and environment validation.
Critical: Centralized config prevents scattered API keys and ensures single source of truth.
"""
import os
import logging
from typing import Optional
from pydantic import BaseSettings, Field, validator
import firebase_admin
from firebase_admin import credentials, firestore
from firebase_admin.exceptions import FirebaseError

# Configure logging immediately for bootstrapping
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


class TradingConfig(BaseSettings):
    """Validated configuration with environment fallbacks."""
    
    # Firebase Configuration
    FIREBASE_CREDENTIALS_PATH: str = Field(
        default="./credentials/firebase_service_account.json",
        description="Path to Firebase service account JSON"
    )
    FIRESTORE_COLLECTION: str = Field(
        default="trading_strategies",
        description="Firestore collection for strategy storage"
    )
    
    # Trading API Configuration
    EXCHANGE_API_KEY: Optional[str] = Field(
        default=None,
        description="Primary exchange API key (CCXT compatible)"
    )
    EXCHANGE_SECRET: Optional[str] = Field(
        default=None,
        description="Primary exchange API secret"
    )
    ALPHA_VANTAGE_KEY: Optional[str] = Field(
        default=None,
        description="Alpha Vantage API key for market data"
    )
    
    # Trading Parameters
    MAX_POSITION_SIZE_USD: float = Field(
        default=10000.0,
        gt=0,
        description="Maximum position size in USD"
    )
    MAX_DAILY_LOSS_PCT: float = Field(
        default=2.0,
        ge=0,
        le=100,
        description="Maximum daily loss percentage before shutdown"
    )
    RISK_PER_TRADE_PCT: float = Field(
        default=0.5,
        gt=0,
        le=5,
        description="Risk percentage per individual trade"
    )
    
    # Model Configuration
    ML_MODEL_PATH: str = Field(
        default="./models/strategy_predictor.pkl",
        description="Path to persisted ML model"
    )
    BACKTEST_LOOKBACK_DAYS: int = Field(
        default=365,
        gt=30,
        description="Days of historical data for backtesting"
    )
    
    # Execution Settings
    ORDER_TIMEOUT_SECONDS: int = Field(
        default=30,
        gt=0,
        description="Seconds to wait for order execution confirmation"
    )
    MAX_RETRY_ATTEMPTS: int = Field(
        default=3,
        ge=0,
        description="Maximum retry attempts for failed operations"
    )
    
    @validator("FIREBASE_CREDENTIALS_PATH")
    def validate_firebase_path(cls, v):
        """Ensure Firebase credentials exist before initialization."""
        if not os.path.exists(v):
            logger.error(f"Firebase credentials not found at {v}")
            raise FileNotFoundError(f"Firebase credentials missing at {v}")
        return v
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Initialize configuration
try:
    config = TradingConfig()
    logger.info("Configuration loaded successfully")
except Exception as e:
    logger.critical(f"Configuration validation failed: {e}")
    raise


# Firebase initialization with error handling
_firebase_app = None
_firestore_client = None

def initialize_firebase() -> bool:
    """
    Initialize Firebase with singleton pattern and comprehensive error handling.
    Returns: True if successful, False otherwise
    """
    global _firebase_app, _firestore_client
    
    if _firebase_app is not None:
        logger.debug("Firebase already initialized")
        return True
    
    try:
        cred = credentials.Certificate(config.FIREBASE_CREDENTIALS_PATH)
        _firebase_app = firebase_admin.initialize_app(cred)
        _firestore_client = firestore.client()
        
        # Test connection with timeout
        test_doc = _firestore_client.collection("health").document("test")
        test_doc.set({"timestamp": firestore.SERVER_TIMESTAMP}, timeout=5)
        test_doc.delete()
        
        logger.info("Firebase initialized and connection verified")
        return True
        
    except FileNotFoundError as e:
        logger.error(f"Firebase credentials file not found: {e}")
        return False
    except FirebaseError as e:
        logger.error(f"Firebase initialization error: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error during Firebase init: {e}")
        return False


def get_firestore_client():
    """Lazy loader for Firestore client with reinitialization attempts."""
    global _firestore_client
    
    if _firestore_client is None:
        if not initialize_firebase():
            raise ConnectionError("Firebase initialization failed")
    
    return _firestore_client


def get_config() -> TradingConfig:
    """Safe configuration accessor with validation."""
    return config