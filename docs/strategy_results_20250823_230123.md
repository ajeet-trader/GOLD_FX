Strategy Run Results (2025-08-23 23:01:23 IST)
================================================================================

### Strategy: src/strategies/fusion/confidence_sizing.py
### Strategy Name: confidence_sizing

--- Mode: MOCK ---
STDOUT OUTPUT:
============================================================
RUN MODE: MOCK - using simulated OHLCV data
============================================================
============================================================
TESTING OPTIMIZED CONFIDENCE SIZING STRATEGY
============================================================
Running in MOCK mode

1. Testing signal generation:
Confidence Sizing - Analyzing XAUUSDm on M15
   Generated 3 signals (avg confidence: 0.62)
      - BUY at 1930.19 (conf: 0.69)
        Position Size: 0.0205
      - BUY at 1930.19 (conf: 0.62)
        Position Size: 0.0246
      - BUY at 1930.19 (conf: 0.55)
        Position Size: 0.0287
   Generated 3 signals
   - Signal: BUY at 1930.19, Confidence: 0.692, Grade: C
     Reason: N/A
     Position Size: 0.020471770178128856
   - Signal: BUY at 1930.19, Confidence: 0.623, Grade: C
     Reason: confidence_sizing_fusion_2
     Position Size: 0.024566124213754627
   - Signal: BUY at 1930.19, Confidence: 0.553, Grade: C
     Reason: confidence_sizing_fusion_3
     Position Size: 0.028660478249380394

2. Testing analysis method:
   Analysis results keys: dict_keys(['strategy', 'symbol', 'timeframe', 'timestamp', 'position_statistics', 'sizing_parameters', 'memory_optimized', 'max_history', 'recent_performance'])
   Memory Optimized: True

3. Testing performance tracking:
   {'strategy_name': 'ConfidenceSizing', 'total_signals': 0, 'win_rate': 0.0, 'profit_factor': 0.0, 'total_pnl': 0.0, 'grade_distribution': {'A': 0, 'B': 0, 'C': 0, 'D': 0}, 'last_signal': None}

4. Strategy Information:
   Name: Confidence-Based Position Sizing Fusion Strategy
   Version: 2.0.0
   Type: Fusion
   Description: Adjusts position sizes based on signal confidence and market conditions.
   Parameters: {'min_signals': 2, 'min_confidence': 0.55, 'base_position_size': 0.01, 'max_position_size': 0.05, 'confidence_multiplier': 2.0, 'volatility_window': 15, 'volatility_threshold': 0.02, 'volatility_adjustment': 0.5, 'correlation_penalty': 0.25, 'max_correlation': 0.75, 'max_history_records': 300}
   Overall Trading Performance:
     Total Signals Generated: 0
     Win Rate: 0.00%
     Profit Factor: 0.00
   Position Sizing Metrics: {'total_positions': 3, 'avg_position_size': np.float64(0.020471770178128856), 'max_position_size': np.float64(0.020471770178128856), 'min_position_size': np.float64(0.020471770178128856), 'avg_confidence': np.float64(0.6916977572286657), 'position_size_std': np.float64(0.0), 'base_position_size': 0.01, 'max_allowed_size': 0.05}

============================================================
CONFIDENCE SIZING STRATEGY TEST COMPLETED!
============================================================

STDERR OUTPUT:
2025-08-23 23:01:24,557 - ConfidenceSizing - INFO - Fused signal: BUY with confidence 0.692, position size 0.0205
2025-08-23 23:01:24,557 - ConfidenceSizing - INFO - Fused signal: BUY with confidence 0.692, position size 0.0205
2025-08-23 23:01:24,558 - ConfidenceSizing - INFO - Fused signal: BUY with confidence 0.692, position size 0.0205

------------------------------------------------------------

--- Mode: LIVE ---
STDOUT OUTPUT:
============================================================
RUN MODE: LIVE - connecting to MT5 (production)
============================================================
✅ Connected to live MT5
============================================================
TESTING OPTIMIZED CONFIDENCE SIZING STRATEGY
============================================================
Running in LIVE mode

1. Testing signal generation:
Confidence Sizing - Analyzing XAUUSDm on M15
   Generated 3 signals (avg confidence: 0.63)
      - BUY at 3372.30 (conf: 0.70)
        Position Size: 0.0215
      - BUY at 3372.30 (conf: 0.63)
        Position Size: 0.0258
      - BUY at 3372.30 (conf: 0.56)
        Position Size: 0.0301
   Generated 3 signals
   - Signal: BUY at 3372.30, Confidence: 0.703, Grade: B
     Reason: N/A
     Position Size: 0.021475924148947414
   - Signal: BUY at 3372.30, Confidence: 0.633, Grade: B
     Reason: confidence_sizing_fusion_2
     Position Size: 0.025771108978736895
   - Signal: BUY at 3372.30, Confidence: 0.563, Grade: B
     Reason: confidence_sizing_fusion_3
     Position Size: 0.03006629380852638

2. Testing analysis method:
   Analysis results keys: dict_keys(['strategy', 'symbol', 'timeframe', 'timestamp', 'position_statistics', 'sizing_parameters', 'memory_optimized', 'max_history', 'recent_performance'])
   Memory Optimized: True

3. Testing performance tracking:
   {'strategy_name': 'ConfidenceSizing', 'total_signals': 0, 'win_rate': 0.0, 'profit_factor': 0.0, 'total_pnl': 0.0, 'grade_distribution': {'A': 0, 'B': 0, 'C': 0, 'D': 0}, 'last_signal': None}

4. Strategy Information:
   Name: Confidence-Based Position Sizing Fusion Strategy
   Version: 2.0.0
   Type: Fusion
   Description: Adjusts position sizes based on signal confidence and market conditions.
   Parameters: {'min_signals': 2, 'min_confidence': 0.55, 'base_position_size': 0.01, 'max_position_size': 0.05, 'confidence_multiplier': 2.0, 'volatility_window': 15, 'volatility_threshold': 0.02, 'volatility_adjustment': 0.5, 'correlation_penalty': 0.25, 'max_correlation': 0.75, 'max_history_records': 300}
   Overall Trading Performance:
     Total Signals Generated: 0
     Win Rate: 0.00%
     Profit Factor: 0.00
   Position Sizing Metrics: {'total_positions': 3, 'avg_position_size': np.float64(0.021475924148947414), 'max_position_size': np.float64(0.021475924148947414), 'min_position_size': np.float64(0.021475924148947414), 'avg_confidence': np.float64(0.7031474228827466), 'position_size_std': np.float64(0.0), 'base_position_size': 0.01, 'max_allowed_size': 0.05}

============================================================
CONFIDENCE SIZING STRATEGY TEST COMPLETED!
============================================================

STDERR OUTPUT:
2025-08-23 23:01:25,360 - src.core.mt5_manager - INFO - Loaded 227 tradable symbols from CSV
2025-08-23 23:01:25,360 - src.core.mt5_manager - INFO - MT5Manager initialized for symbol: XAUUSD
2025-08-23 23:01:25,360 - src.core.mt5_manager - INFO - Magic number: 123456
2025-08-23 23:01:25,360 - src.core.mt5_manager - INFO - Initializing MT5 with default path
2025-08-23 23:01:25,371 - src.core.mt5_manager - INFO - Attempting to login to account 273949055 on server Exness-MT5Trial6
2025-08-23 23:01:25,371 - src.core.mt5_manager - WARNING - Symbol XAUUSD not found, trying alternative symbols
2025-08-23 23:01:25,371 - src.core.mt5_manager - INFO - Using alternative symbol: XAUUSDm
2025-08-23 23:01:25,371 - src.core.mt5_manager - INFO - ✅ Successfully connected to MT5
2025-08-23 23:01:25,371 - src.core.mt5_manager - INFO -    Account: 273949055
2025-08-23 23:01:25,372 - src.core.mt5_manager - INFO -    Balance: $194.55
2025-08-23 23:01:25,372 - src.core.mt5_manager - INFO -    Server: Exness-MT5Trial6
2025-08-23 23:01:25,372 - src.core.mt5_manager - INFO -    Symbol: XAUUSDm
2025-08-23 23:01:25,374 - src.core.mt5_manager - INFO - Retrieved 80 bars for XAUUSDm M15
2025-08-23 23:01:25,378 - ConfidenceSizing - INFO - Fused signal: BUY with confidence 0.703, position size 0.0215
2025-08-23 23:01:25,378 - ConfidenceSizing - INFO - Fused signal: BUY with confidence 0.703, position size 0.0215
2025-08-23 23:01:25,379 - ConfidenceSizing - INFO - Fused signal: BUY with confidence 0.703, position size 0.0215
2025-08-23 23:01:25,381 - src.core.mt5_manager - INFO - Retrieved 80 bars for XAUUSDm M15

------------------------------------------------------------

================================================================================

### Strategy: src/strategies/fusion/regime_detection.py
### Strategy Name: regime_detection

--- Mode: MOCK ---
STDOUT OUTPUT:
============================================================
RUN MODE: MOCK - using simulated OHLCV data
============================================================
============================================================
TESTING MODIFIED REGIME DETECTION STRATEGY
============================================================
Running in MOCK mode

1. Testing signal generation:
   Generated 0 signals

2. Testing analysis method:
   Analysis results keys: dict_keys(['strategy', 'symbol', 'timeframe', 'timestamp', 'current_regime_detected', 'current_regime_confidence', 'regime_statistics_history', 'detection_parameters', 'total_component_signals_processed', 'memory_optimized', 'max_regime_history', 'max_history_component_signals'])
   Current Regime Detected: high_volatility
   Regime History Length: N/A

3. Testing performance tracking:
   {'strategy_name': 'RegimeDetection', 'total_signals': 0, 'win_rate': 0.0, 'profit_factor': 0.0, 'total_pnl': 0.0, 'grade_distribution': {'A': 0, 'B': 0, 'C': 0, 'D': 0}, 'last_signal': None}

4. Strategy Information:
   Name: Regime Detection Fusion Strategy
   Version: 2.0.0
   Type: Fusion
   Description: Detects market regimes and adapts signal fusion based on current market conditions.
   Parameters: {'min_signals': 2, 'min_confidence': 0.55, 'lookback_period': 30, 'trend_threshold': 0.015, 'volatility_window': 15, 'volatility_threshold': 0.012, 'breakout_threshold': 1.5, 'max_regime_history': 100, 'max_history_component_signals': 400}
   Current Regime State: {'regime_type': 'high_volatility', 'confidence': 1.0, 'history_length': 1}
   Regime Detection Statistics: {'current_regime': 'high_volatility', 'regime_confidence': 1.0, 'regime_history_length': 1, 'regime_distribution': {'high_volatility': 1}, 'regime_performance_by_component_strategy': {}}
   Overall Trading Performance:
     Total Signals Generated: 0
     Win Rate: 0.00%
     Profit Factor: 0.00

============================================================
REGIME DETECTION STRATEGY TEST COMPLETED!
============================================================

STDERR OUTPUT:
2025-08-23 23:01:26,093 - RegimeDetection - INFO - RegimeDetection initialized
2025-08-23 23:01:26,093 - RegimeDetection - INFO - Regime Detection - Analyzing XAUUSDm on M15
2025-08-23 23:01:26,103 - RegimeDetection - INFO - No valid signals generated in high_volatility regime

------------------------------------------------------------

--- Mode: LIVE ---
STDOUT OUTPUT:
============================================================
RUN MODE: LIVE - connecting to MT5 (production)
============================================================
✅ Connected to live MT5
============================================================
TESTING MODIFIED REGIME DETECTION STRATEGY
============================================================
Running in LIVE mode

1. Testing signal generation:
   Generated 0 signals

2. Testing analysis method:
   Analysis results keys: dict_keys(['strategy', 'symbol', 'timeframe', 'timestamp', 'current_regime_detected', 'current_regime_confidence', 'regime_statistics_history', 'detection_parameters', 'total_component_signals_processed', 'memory_optimized', 'max_regime_history', 'max_history_component_signals'])
   Current Regime Detected: high_volatility
   Regime History Length: N/A

3. Testing performance tracking:
   {'strategy_name': 'RegimeDetection', 'total_signals': 0, 'win_rate': 0.0, 'profit_factor': 0.0, 'total_pnl': 0.0, 'grade_distribution': {'A': 0, 'B': 0, 'C': 0, 'D': 0}, 'last_signal': None}

4. Strategy Information:
   Name: Regime Detection Fusion Strategy
   Version: 2.0.0
   Type: Fusion
   Description: Detects market regimes and adapts signal fusion based on current market conditions.
   Parameters: {'min_signals': 2, 'min_confidence': 0.55, 'lookback_period': 30, 'trend_threshold': 0.015, 'volatility_window': 15, 'volatility_threshold': 0.012, 'breakout_threshold': 1.5, 'max_regime_history': 100, 'max_history_component_signals': 400}
   Current Regime State: {'regime_type': 'high_volatility', 'confidence': 1.0, 'history_length': 1}
   Regime Detection Statistics: {'current_regime': 'high_volatility', 'regime_confidence': 1.0, 'regime_history_length': 1, 'regime_distribution': {'high_volatility': 1}, 'regime_performance_by_component_strategy': {}}
   Overall Trading Performance:
     Total Signals Generated: 0
     Win Rate: 0.00%
     Profit Factor: 0.00

============================================================
REGIME DETECTION STRATEGY TEST COMPLETED!
============================================================

STDERR OUTPUT:
2025-08-23 23:01:26,818 - src.core.mt5_manager - INFO - Loaded 227 tradable symbols from CSV
2025-08-23 23:01:26,818 - src.core.mt5_manager - INFO - MT5Manager initialized for symbol: XAUUSD
2025-08-23 23:01:26,818 - src.core.mt5_manager - INFO - Magic number: 123456
2025-08-23 23:01:26,819 - src.core.mt5_manager - INFO - Initializing MT5 with default path
2025-08-23 23:01:26,829 - src.core.mt5_manager - INFO - Attempting to login to account 273949055 on server Exness-MT5Trial6
2025-08-23 23:01:26,829 - src.core.mt5_manager - WARNING - Symbol XAUUSD not found, trying alternative symbols
2025-08-23 23:01:26,829 - src.core.mt5_manager - INFO - Using alternative symbol: XAUUSDm
2025-08-23 23:01:26,829 - src.core.mt5_manager - INFO - ✅ Successfully connected to MT5
2025-08-23 23:01:26,829 - src.core.mt5_manager - INFO -    Account: 273949055
2025-08-23 23:01:26,829 - src.core.mt5_manager - INFO -    Balance: $194.55
2025-08-23 23:01:26,829 - src.core.mt5_manager - INFO -    Server: Exness-MT5Trial6
2025-08-23 23:01:26,829 - src.core.mt5_manager - INFO -    Symbol: XAUUSDm
2025-08-23 23:01:26,829 - RegimeDetection - INFO - RegimeDetection initialized
2025-08-23 23:01:26,830 - RegimeDetection - INFO - Regime Detection - Analyzing XAUUSDm on M15
2025-08-23 23:01:26,833 - src.core.mt5_manager - INFO - Retrieved 90 bars for XAUUSDm M15
2025-08-23 23:01:26,840 - RegimeDetection - INFO - No valid signals generated in high_volatility regime
2025-08-23 23:01:26,841 - src.core.mt5_manager - INFO - Retrieved 90 bars for XAUUSDm M15

------------------------------------------------------------

================================================================================

### Strategy: src/strategies/fusion/weighted_voting.py
### Strategy Name: weighted_voting

--- Mode: MOCK ---
STDOUT OUTPUT:
============================================================
RUN MODE: MOCK - using simulated OHLCV data
============================================================
============================================================
TESTING MODIFIED WEIGHTED VOTING STRATEGY
============================================================
Running in MOCK mode

1. Testing signal generation:
   Generated 3 signals
   - Signal: BUY at 1929.23, Confidence: 0.626, Grade: C
     Reason: N/A
   - Signal: BUY at 1929.23, Confidence: 0.657, Grade: C
     Reason: weighted_voting_fusion_variation_2
   - Signal: BUY at 1929.23, Confidence: 0.688, Grade: C
     Reason: weighted_voting_fusion_variation_3

2. Testing analysis method:
   Analysis results keys: dict_keys(['strategy', 'symbol', 'timeframe', 'timestamp', 'total_fused_signals_recorded', 'active_component_strategies', 'component_strategy_performance_metrics', 'current_component_weights', 'fusion_parameters', 'memory_optimized', 'max_fusion_records_history', 'recent_fused_signal_count', 'average_fused_confidence'])
   Total Fused Signals Recorded: 3
   Active Component Strategies: 5

3. Testing performance tracking:
   {'strategy_name': 'WeightedVoting', 'total_signals': 0, 'win_rate': 0.0, 'profit_factor': 0.0, 'total_pnl': 0.0, 'grade_distribution': {'A': 0, 'B': 0, 'C': 0, 'D': 0}, 'last_signal': None}

4. Strategy Information:
   Name: Weighted Voting Fusion Strategy
   Version: 2.0.0
   Type: Fusion
   Description: Combines signals from multiple strategies using weighted voting, dynamically adjusting weights.
   Parameters: {'min_signals': 2, 'min_confidence': 0.6, 'performance_window': 50, 'initial_weight': 1.0, 'weight_decay': 0.95, 'weight_boost': 1.1, 'min_weight': 0.1, 'max_weight': 3.0, 'max_history_records': 500}
   Overall Trading Performance:
     Total Signals Generated: 0
     Win Rate: 0.00%
     Profit Factor: 0.00
   Component Strategy Insights:
     Active Strategies Count: 5

============================================================
WEIGHTED VOTING STRATEGY TEST COMPLETED!
============================================================

STDERR OUTPUT:
2025-08-23 23:01:27,542 - WeightedVoting - INFO - WeightedVoting initialized
2025-08-23 23:01:27,543 - WeightedVoting - INFO - Weighted Voting - Analyzing XAUUSDm on M15
2025-08-23 23:01:27,544 - WeightedVoting - INFO - Fused signal: BUY with confidence 0.626
2025-08-23 23:01:27,544 - WeightedVoting - INFO - Fused signal: BUY with confidence 0.626
2025-08-23 23:01:27,544 - WeightedVoting - INFO - Fused signal: BUY with confidence 0.626
2025-08-23 23:01:27,544 - WeightedVoting - INFO - Generated 3 signals (avg confidence: 0.66)

------------------------------------------------------------

--- Mode: LIVE ---
STDOUT OUTPUT:
============================================================
RUN MODE: LIVE - connecting to MT5 (production)
============================================================
✅ Connected to live MT5
============================================================
TESTING MODIFIED WEIGHTED VOTING STRATEGY
============================================================
Running in LIVE mode

1. Testing signal generation:
   Generated 3 signals
   - Signal: BUY at 3372.30, Confidence: 0.627, Grade: C
     Reason: N/A
   - Signal: BUY at 3372.30, Confidence: 0.659, Grade: C
     Reason: weighted_voting_fusion_variation_2
   - Signal: BUY at 3372.30, Confidence: 0.690, Grade: C
     Reason: weighted_voting_fusion_variation_3

2. Testing analysis method:
   Analysis results keys: dict_keys(['strategy', 'symbol', 'timeframe', 'timestamp', 'total_fused_signals_recorded', 'active_component_strategies', 'component_strategy_performance_metrics', 'current_component_weights', 'fusion_parameters', 'memory_optimized', 'max_fusion_records_history', 'recent_fused_signal_count', 'average_fused_confidence'])
   Total Fused Signals Recorded: 3
   Active Component Strategies: 5

3. Testing performance tracking:
   {'strategy_name': 'WeightedVoting', 'total_signals': 0, 'win_rate': 0.0, 'profit_factor': 0.0, 'total_pnl': 0.0, 'grade_distribution': {'A': 0, 'B': 0, 'C': 0, 'D': 0}, 'last_signal': None}

4. Strategy Information:
   Name: Weighted Voting Fusion Strategy
   Version: 2.0.0
   Type: Fusion
   Description: Combines signals from multiple strategies using weighted voting, dynamically adjusting weights.
   Parameters: {'min_signals': 2, 'min_confidence': 0.6, 'performance_window': 50, 'initial_weight': 1.0, 'weight_decay': 0.95, 'weight_boost': 1.1, 'min_weight': 0.1, 'max_weight': 3.0, 'max_history_records': 500}
   Overall Trading Performance:
     Total Signals Generated: 0
     Win Rate: 0.00%
     Profit Factor: 0.00
   Component Strategy Insights:
     Active Strategies Count: 5

============================================================
WEIGHTED VOTING STRATEGY TEST COMPLETED!
============================================================

STDERR OUTPUT:
2025-08-23 23:01:28,261 - src.core.mt5_manager - INFO - Loaded 227 tradable symbols from CSV
2025-08-23 23:01:28,261 - src.core.mt5_manager - INFO - MT5Manager initialized for symbol: XAUUSD
2025-08-23 23:01:28,261 - src.core.mt5_manager - INFO - Magic number: 123456
2025-08-23 23:01:28,261 - src.core.mt5_manager - INFO - Initializing MT5 with default path
2025-08-23 23:01:28,271 - src.core.mt5_manager - INFO - Attempting to login to account 273949055 on server Exness-MT5Trial6
2025-08-23 23:01:28,272 - src.core.mt5_manager - WARNING - Symbol XAUUSD not found, trying alternative symbols
2025-08-23 23:01:28,272 - src.core.mt5_manager - INFO - Using alternative symbol: XAUUSDm
2025-08-23 23:01:28,272 - src.core.mt5_manager - INFO - ✅ Successfully connected to MT5
2025-08-23 23:01:28,272 - src.core.mt5_manager - INFO -    Account: 273949055
2025-08-23 23:01:28,272 - src.core.mt5_manager - INFO -    Balance: $194.55
2025-08-23 23:01:28,272 - src.core.mt5_manager - INFO -    Server: Exness-MT5Trial6
2025-08-23 23:01:28,272 - src.core.mt5_manager - INFO -    Symbol: XAUUSDm
2025-08-23 23:01:28,272 - WeightedVoting - INFO - WeightedVoting initialized
2025-08-23 23:01:28,272 - WeightedVoting - INFO - Weighted Voting - Analyzing XAUUSDm on M15
2025-08-23 23:01:28,275 - src.core.mt5_manager - INFO - Retrieved 100 bars for XAUUSDm M15
2025-08-23 23:01:28,276 - WeightedVoting - INFO - Fused signal: BUY with confidence 0.627
2025-08-23 23:01:28,277 - WeightedVoting - INFO - Fused signal: BUY with confidence 0.627
2025-08-23 23:01:28,277 - WeightedVoting - INFO - Fused signal: BUY with confidence 0.627
2025-08-23 23:01:28,277 - WeightedVoting - INFO - Generated 3 signals (avg confidence: 0.66)
2025-08-23 23:01:28,279 - src.core.mt5_manager - INFO - Retrieved 100 bars for XAUUSDm M15

------------------------------------------------------------

================================================================================

### Strategy: src/strategies/fusion/adaptive_ensemble.py
### Strategy Name: adaptive_ensemble

--- Mode: MOCK ---
STDOUT OUTPUT:
============================================================
RUN MODE: MOCK - using simulated OHLCV data
============================================================
============================================================
TESTING MODIFIED ADAPTIVE ENSEMBLE FUSION STRATEGY
============================================================
Running in MOCK mode

1. Testing signal generation:
   Generated 1 signals
   - Signal: BUY at 1929.23, Confidence: 0.711, Grade: B
     Regime: LOW_VOLATILITY
     Component Signals: 8

2. Testing analysis method:
   Analysis results keys: dict_keys(['strategy', 'symbol', 'timeframe', 'analysis_time', 'current_market_regime', 'current_strategy_weights', 'ensemble_performance_summary', 'component_strategy_performance_snapshot'])
   Current Market Regime: LOW_VOLATILITY
   Component Strategy Weights: {'technical_trend': 0.13, 'technical_momentum': 0.116, 'technical_breakout': 0.125, 'smc_structure': 0.14, 'smc_liquidity': 0.12, 'ml_lstm': 0.13, 'ml_ensemble': 0.121, 'volume_profile': 0.118}

3. Testing performance tracking:
   {'strategy_name': 'AdaptiveEnsembleFusionStrategy', 'total_signals': 0, 'win_rate': 0.0, 'profit_factor': 0.0, 'total_pnl': 0.0, 'grade_distribution': {'A': 0, 'B': 0, 'C': 0, 'D': 0}, 'last_signal': None}

4. Strategy Information:
   Name: Adaptive Ensemble Fusion Strategy
   Version: 2.0.0
   Type: Fusion
   Description: Dynamically adjusts strategy weights for optimal signal fusion.
   Parameters: {'lookback_bars': 100, 'performance_window': 30, 'adaptation_rate': 0.1, 'min_signals_for_weight': 5, 'correlation_threshold': 0.7, 'decay_factor': 0.95}
   Overall Trading Performance (from AbstractStrategy):
     Total Signals Generated: 0
     Win Rate: 0.00%
     Profit Factor: 0.00
   Internal Ensemble Metrics:
     Ensemble Internal Performance (Total Signals): 1
     Current Regime Snapshot (Type): LOW_VOLATILITY
     Strategy Weights Snapshot: {'technical_trend': 0.1301811462282112, 'technical_momentum': 0.11638480168978863, 'technical_breakout': 0.12470369286526568, 'smc_structure': 0.13989612493905865, 'smc_liquidity': 0.1196632915526182, 'ml_lstm': 0.12983590839262668, 'ml_ensemble': 0.12084216542911451, 'volume_profile': 0.11849286890331635}

============================================================
ADAPTIVE ENSEMBLE FUSION STRATEGY TEST COMPLETED!
============================================================

STDERR OUTPUT:
2025-08-23 23:01:29,008 - AdaptiveEnsembleFusionStrategy - INFO - Adaptive Ensemble Fusion Strategy initialized
2025-08-23 23:01:29,008 - AdaptiveEnsembleFusionStrategy - INFO - Adaptive Ensemble - Analyzing XAUUSDm on M15
2025-08-23 23:01:29,016 - AdaptiveEnsembleFusionStrategy - INFO - Generated 1 signals (avg confidence: 0.71)

------------------------------------------------------------

--- Mode: LIVE ---
STDOUT OUTPUT:
============================================================
RUN MODE: LIVE - connecting to MT5 (production)
============================================================
✅ Connected to live MT5
============================================================
TESTING MODIFIED ADAPTIVE ENSEMBLE FUSION STRATEGY
============================================================
Running in LIVE mode

1. Testing signal generation:
   Generated 1 signals
   - Signal: BUY at 3372.30, Confidence: 0.696, Grade: C
     Regime: LOW_VOLATILITY
     Component Signals: 8

2. Testing analysis method:
   Analysis results keys: dict_keys(['strategy', 'symbol', 'timeframe', 'analysis_time', 'current_market_regime', 'current_strategy_weights', 'ensemble_performance_summary', 'component_strategy_performance_snapshot'])
   Current Market Regime: LOW_VOLATILITY
   Component Strategy Weights: {'technical_trend': 0.128, 'technical_momentum': 0.117, 'technical_breakout': 0.123, 'smc_structure': 0.129, 'smc_liquidity': 0.134, 'ml_lstm': 0.128, 'ml_ensemble': 0.118, 'volume_profile': 0.123}

3. Testing performance tracking:
   {'strategy_name': 'AdaptiveEnsembleFusionStrategy', 'total_signals': 0, 'win_rate': 0.0, 'profit_factor': 0.0, 'total_pnl': 0.0, 'grade_distribution': {'A': 0, 'B': 0, 'C': 0, 'D': 0}, 'last_signal': None}

4. Strategy Information:
   Name: Adaptive Ensemble Fusion Strategy
   Version: 2.0.0
   Type: Fusion
   Description: Dynamically adjusts strategy weights for optimal signal fusion.
   Parameters: {'lookback_bars': 100, 'performance_window': 30, 'adaptation_rate': 0.1, 'min_signals_for_weight': 5, 'correlation_threshold': 0.7, 'decay_factor': 0.95}
   Overall Trading Performance (from AbstractStrategy):
     Total Signals Generated: 0
     Win Rate: 0.00%
     Profit Factor: 0.00
   Internal Ensemble Metrics:
     Ensemble Internal Performance (Total Signals): 1
     Current Regime Snapshot (Type): LOW_VOLATILITY
     Strategy Weights Snapshot: {'technical_trend': 0.1278791813899442, 'technical_momentum': 0.1167154482894284, 'technical_breakout': 0.12273380638757492, 'smc_structure': 0.12934835960146823, 'smc_liquidity': 0.13381566609090018, 'ml_lstm': 0.12810435174350832, 'ml_ensemble': 0.11790480247014418, 'volume_profile': 0.12349838402703175}

============================================================
ADAPTIVE ENSEMBLE FUSION STRATEGY TEST COMPLETED!
============================================================

STDERR OUTPUT:
2025-08-23 23:01:29,775 - src.core.mt5_manager - INFO - Loaded 227 tradable symbols from CSV
2025-08-23 23:01:29,775 - src.core.mt5_manager - INFO - MT5Manager initialized for symbol: XAUUSD
2025-08-23 23:01:29,775 - src.core.mt5_manager - INFO - Magic number: 123456
2025-08-23 23:01:29,775 - src.core.mt5_manager - INFO - Initializing MT5 with default path
2025-08-23 23:01:29,786 - src.core.mt5_manager - INFO - Attempting to login to account 273949055 on server Exness-MT5Trial6
2025-08-23 23:01:29,787 - src.core.mt5_manager - WARNING - Symbol XAUUSD not found, trying alternative symbols
2025-08-23 23:01:29,787 - src.core.mt5_manager - INFO - Using alternative symbol: XAUUSDm
2025-08-23 23:01:29,787 - src.core.mt5_manager - INFO - ✅ Successfully connected to MT5
2025-08-23 23:01:29,787 - src.core.mt5_manager - INFO -    Account: 273949055
2025-08-23 23:01:29,787 - src.core.mt5_manager - INFO -    Balance: $194.55
2025-08-23 23:01:29,787 - src.core.mt5_manager - INFO -    Server: Exness-MT5Trial6
2025-08-23 23:01:29,787 - src.core.mt5_manager - INFO -    Symbol: XAUUSDm
2025-08-23 23:01:29,787 - AdaptiveEnsembleFusionStrategy - INFO - Adaptive Ensemble Fusion Strategy initialized
2025-08-23 23:01:29,787 - AdaptiveEnsembleFusionStrategy - INFO - Adaptive Ensemble - Analyzing XAUUSDm on M15
2025-08-23 23:01:29,790 - src.core.mt5_manager - INFO - Retrieved 100 bars for XAUUSDm M15
2025-08-23 23:01:29,797 - AdaptiveEnsembleFusionStrategy - INFO - Generated 1 signals (avg confidence: 0.70)
2025-08-23 23:01:29,799 - src.core.mt5_manager - INFO - Retrieved 200 bars for XAUUSDm M15

------------------------------------------------------------

================================================================================

### Strategy: src/strategies/ml/ensemble_nn.py
### Strategy Name: ensemble_nn

--- Mode: MOCK ---
STDOUT OUTPUT:
============================================================
RUN MODE: MOCK - using simulated OHLCV data
============================================================
============================================================
TESTING MODIFIED ENSEMBLE NN STRATEGY
============================================================

1. Testing signal generation:
   Generated 0 signals

2. Testing analysis method:
   Analysis results keys: dict_keys(['strategy', 'symbol', 'timeframe', 'tensorflow_available', 'ensemble_trained', 'ensemble_accuracy', 'predictions_attempted', 'models_count', 'memory_optimized', 'lookback_bars', 'sequence_length', 'latest_training_time'])
   TensorFlow Available: True
   Ensemble Trained: False
   Ensemble Accuracy: 0.00

3. Testing performance tracking:
   {'strategy_name': 'EnsembleNNStrategy', 'total_signals': 0, 'win_rate': 0.0, 'profit_factor': 0.0, 'total_pnl': 0.0, 'grade_distribution': {'A': 0, 'B': 0, 'C': 0, 'D': 0}, 'last_signal': None}

4. Strategy Information:
   Name: Ensemble Neural Network Strategy
   Version: 2.0.0
   Type: Machine Learning
   Description: Combines multiple neural network architectures for robust predictions.
   Parameters: {'lookback_bars': 150, 'sequence_length': 20, 'min_confidence': 0.65, 'ensemble_size': 2, 'epochs': 2, 'batch_size': 8, 'learning_rate': 0.001, 'max_training_samples': 800, 'memory_cleanup_interval': 30}
   ML Specific Metrics:
     - tensorflow_available: True
     - ensemble_trained: False
     - ensemble_accuracy: 0.00
     - predictions_attempted: 1
     - models_count: 2
   Overall Trading Performance:
     Total Signals Generated: 0
     Win Rate: 0.00%
     Profit Factor: 0.00

============================================================
ENSEMBLE NN STRATEGY TEST COMPLETED!
============================================================

STDERR OUTPUT:
2025-08-23 23:01:31.925475: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
2025-08-23 23:01:41.139846: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
2025-08-23 23:01:44.088796: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: SSE3 SSE4.1 SSE4.2 AVX AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
2025-08-23 23:01:46,974 - EnsembleNNStrategy - INFO - Initialized 2 neural network models
2025-08-23 23:01:46,975 - EnsembleNNStrategy - INFO - EnsembleNNStrategy initialized (TensorFlow available: True)
2025-08-23 23:01:46,975 - EnsembleNNStrategy - INFO - Ensemble NN - Analyzing XAUUSDm on M15
2025-08-23 23:01:46,979 - EnsembleNNStrategy - INFO - Training ensemble models...
2025-08-23 23:01:46,979 - EnsembleNNStrategy - INFO - Starting ensemble neural networks training...
2025-08-23 23:01:47,347 - EnsembleNNStrategy - WARNING - Insufficient or invalid training data for ensemble training.
2025-08-23 23:01:47,371 - EnsembleNNStrategy - WARNING - Scaler not fitted yet for feature extraction, returning unscaled features.
2025-08-23 23:01:47,373 - EnsembleNNStrategy - INFO - No valid signals generated

------------------------------------------------------------

--- Mode: LIVE ---
STDOUT OUTPUT:
============================================================
RUN MODE: LIVE - connecting to MT5 (production)
============================================================
============================================================
TESTING MODIFIED ENSEMBLE NN STRATEGY
============================================================

1. Testing signal generation:
   Generated 0 signals

2. Testing analysis method:
   Analysis results keys: dict_keys(['strategy', 'symbol', 'timeframe', 'tensorflow_available', 'ensemble_trained', 'ensemble_accuracy', 'predictions_attempted', 'models_count', 'memory_optimized', 'lookback_bars', 'sequence_length', 'latest_training_time'])
   TensorFlow Available: True
   Ensemble Trained: True
   Ensemble Accuracy: 1.00

3. Testing performance tracking:
   {'strategy_name': 'EnsembleNNStrategy', 'total_signals': 0, 'win_rate': 0.0, 'profit_factor': 0.0, 'total_pnl': 0.0, 'grade_distribution': {'A': 0, 'B': 0, 'C': 0, 'D': 0}, 'last_signal': None}

4. Strategy Information:
   Name: Ensemble Neural Network Strategy
   Version: 2.0.0
   Type: Machine Learning
   Description: Combines multiple neural network architectures for robust predictions.
   Parameters: {'lookback_bars': 150, 'sequence_length': 20, 'min_confidence': 0.65, 'ensemble_size': 2, 'epochs': 2, 'batch_size': 8, 'learning_rate': 0.001, 'max_training_samples': 800, 'memory_cleanup_interval': 30}
   ML Specific Metrics:
     - tensorflow_available: True
     - ensemble_trained: True
     - ensemble_accuracy: 1.00
     - predictions_attempted: 1
     - models_count: 2
   Overall Trading Performance:
     Total Signals Generated: 0
     Win Rate: 0.00%
     Profit Factor: 0.00

============================================================
ENSEMBLE NN STRATEGY TEST COMPLETED!
============================================================

STDERR OUTPUT:
2025-08-23 23:01:52.367738: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
2025-08-23 23:01:56.256058: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
2025-08-23 23:01:56.968626: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: SSE3 SSE4.1 SSE4.2 AVX AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
2025-08-23 23:01:57,107 - EnsembleNNStrategy - INFO - Initialized 2 neural network models
2025-08-23 23:01:57,107 - EnsembleNNStrategy - INFO - EnsembleNNStrategy initialized (TensorFlow available: True)
2025-08-23 23:01:57,108 - EnsembleNNStrategy - INFO - Ensemble NN - Analyzing XAUUSDm on M15
2025-08-23 23:01:57,113 - EnsembleNNStrategy - INFO - Training ensemble models...
2025-08-23 23:01:57,113 - EnsembleNNStrategy - INFO - Starting ensemble neural networks training...
2025-08-23 23:02:14,710 - EnsembleNNStrategy - INFO - Model 1 trained. Accuracy: 1.000
2025-08-23 23:02:27,531 - EnsembleNNStrategy - INFO - Model 2 trained. Accuracy: 1.000
2025-08-23 23:02:27,531 - EnsembleNNStrategy - INFO - Ensemble training completed. Average accuracy: 1.000
2025-08-23 23:02:27,651 - EnsembleNNStrategy - ERROR - Model 2 prediction failed: Exception encountered when calling Sequential.call().

[1mCannot take the length of shape with unknown rank.[0m

Arguments received by Sequential.call():
  • inputs=tf.Tensor(shape=<unknown>, dtype=float32)
  • training=False
  • mask=None
  • kwargs=<class 'inspect._empty'>
Traceback (most recent call last):
  File "J:\Gold_FX\src\strategies\ml\ensemble_nn.py", line 762, in _make_ensemble_prediction
    pred = model.predict(lstm_input, verbose=0)[0]
           ~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^
  File "J:\Gold_FX\venv\Lib\site-packages\keras\src\utils\traceback_utils.py", line 122, in error_handler
    raise e.with_traceback(filtered_tb) from None
  File "J:\Gold_FX\venv\Lib\site-packages\keras\src\utils\traceback_utils.py", line 124, in error_handler
    del filtered_tb
        ^^^^^^^^^^^
ValueError: Exception encountered when calling Sequential.call().

[1mCannot take the length of shape with unknown rank.[0m

Arguments received by Sequential.call():
  • inputs=tf.Tensor(shape=<unknown>, dtype=float32)
  • training=False
  • mask=None
  • kwargs=<class 'inspect._empty'>
2025-08-23 23:02:27,863 - EnsembleNNStrategy - ERROR - Individual model prediction failed: Exception encountered when calling Sequential.call().

[1mCannot take the length of shape with unknown rank.[0m

Arguments received by Sequential.call():
  • inputs=tf.Tensor(shape=<unknown>, dtype=float32)
  • training=False
  • mask=None
  • kwargs=<class 'inspect._empty'>
Traceback (most recent call last):
  File "J:\Gold_FX\src\strategies\ml\ensemble_nn.py", line 948, in _make_multiple_predictions
    pred = model.predict(lstm_input, verbose=0)[0]
           ~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^
  File "J:\Gold_FX\venv\Lib\site-packages\keras\src\utils\traceback_utils.py", line 122, in error_handler
    raise e.with_traceback(filtered_tb) from None
  File "J:\Gold_FX\venv\Lib\site-packages\keras\src\utils\traceback_utils.py", line 124, in error_handler
    del filtered_tb
        ^^^^^^^^^^^
ValueError: Exception encountered when calling Sequential.call().

[1mCannot take the length of shape with unknown rank.[0m

Arguments received by Sequential.call():
  • inputs=tf.Tensor(shape=<unknown>, dtype=float32)
  • training=False
  • mask=None
  • kwargs=<class 'inspect._empty'>
2025-08-23 23:02:27,864 - EnsembleNNStrategy - INFO - No valid signals generated

------------------------------------------------------------

================================================================================

### Strategy: src/strategies/ml/lstm_predictor.py
### Strategy Name: lstm_predictor

--- Mode: MOCK ---
EXECUTION TIMEOUT: Strategy execution timed out after 60 seconds

------------------------------------------------------------

--- Mode: LIVE ---
EXECUTION TIMEOUT: Strategy execution timed out after 60 seconds

------------------------------------------------------------

================================================================================

### Strategy: src/strategies/ml/rl_agent.py
### Strategy Name: rl_agent

--- Mode: MOCK ---
STDOUT OUTPUT:
============================================================
RUN MODE: MOCK - using simulated OHLCV data
============================================================
============================================================
TESTING MODIFIED REINFORCEMENT LEARNING AGENT STRATEGY
============================================================
Running in MOCK mode

1. Testing signal generation:
   Generated 0 signals

2. Testing analysis method:
   Analysis results keys: dict_keys(['strategy', 'symbol', 'timeframe', 'analysis_time', 'current_price', 'tensorflow_available', 'rl_agent_trained', 'total_episodes', 'total_predictions_made', 'current_epsilon', 'current_total_reward', 'memory_usage_percent', 'training_steps_performed', 'last_action_taken', 'lookback_bars_used'])
   TensorFlow Available: True
   RL Agent Trained: False
   Current Epsilon: 0.8
   Memory Usage: 0.0%

3. Testing performance tracking:
   {'strategy_name': 'RLAgentStrategy', 'total_signals': 0, 'win_rate': 0.0, 'profit_factor': 0.0, 'total_pnl': 0.0, 'grade_distribution': {'A': 0, 'B': 0, 'C': 0, 'D': 0}, 'last_signal': None}

4. Strategy Information:
   Name: Reinforcement Learning Agent Strategy
   Version: 2.0.0
   Type: Machine Learning
   Description: Deep Q-Network (DQN) based RL agent for XAUUSD trading.
   Parameters: {'lookback_bars': 80, 'min_confidence': 0.55, 'state_size': 8, 'action_size': 3, 'memory_size': 100, 'batch_size': 8, 'learning_rate': 0.001, 'epsilon': 0.8, 'epsilon_min': 0.05, 'epsilon_decay': 0.99, 'gamma': 0.9, 'update_target_freq': 10, 'memory_cleanup_interval': 5, 'max_training_frequency': 5}
   RL Specific Metrics:
     - tensorflow_available: True
     - rl_agent_trained: False
     - total_reward_accumulated: 0.00
     - total_episodes_completed: 0
     - total_predictions_made: 1
     - current_epsilon_value: 0.80
     - current_memory_usage: 0
     - total_training_steps: 0
   Overall Trading Performance:
     Total Signals Generated: 0
     Win Rate: 0.00%
     Profit Factor: 0.00

============================================================
REINFORCEMENT LEARNING AGENT STRATEGY TEST COMPLETED!
============================================================

STDERR OUTPUT:
2025-08-23 23:04:29.929824: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
2025-08-23 23:04:34.026918: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
2025-08-23 23:04:35.717785: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: SSE3 SSE4.1 SSE4.2 AVX AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
2025-08-23 23:04:35,967 - RLAgentStrategy - INFO - RL networks initialized
2025-08-23 23:04:35,967 - RLAgentStrategy - INFO - RLAgentStrategy initialized (TensorFlow available: True)
2025-08-23 23:04:35,967 - RLAgentStrategy - INFO - RL Agent - Analyzing XAUUSDm on M15
2025-08-23 23:04:35,992 - RLAgentStrategy - INFO - No valid signals generated

------------------------------------------------------------

--- Mode: LIVE ---
STDOUT OUTPUT:
============================================================
RUN MODE: LIVE - connecting to MT5 (production)
============================================================
✅ Connected to live MT5
============================================================
TESTING MODIFIED REINFORCEMENT LEARNING AGENT STRATEGY
============================================================
Running in LIVE mode

1. Testing signal generation:
   Generated 0 signals

2. Testing analysis method:
   Analysis results keys: dict_keys(['strategy', 'symbol', 'timeframe', 'analysis_time', 'current_price', 'tensorflow_available', 'rl_agent_trained', 'total_episodes', 'total_predictions_made', 'current_epsilon', 'current_total_reward', 'memory_usage_percent', 'training_steps_performed', 'last_action_taken', 'lookback_bars_used'])
   TensorFlow Available: True
   RL Agent Trained: False
   Current Epsilon: 0.8
   Memory Usage: 0.0%

3. Testing performance tracking:
   {'strategy_name': 'RLAgentStrategy', 'total_signals': 0, 'win_rate': 0.0, 'profit_factor': 0.0, 'total_pnl': 0.0, 'grade_distribution': {'A': 0, 'B': 0, 'C': 0, 'D': 0}, 'last_signal': None}

4. Strategy Information:
   Name: Reinforcement Learning Agent Strategy
   Version: 2.0.0
   Type: Machine Learning
   Description: Deep Q-Network (DQN) based RL agent for XAUUSD trading.
   Parameters: {'lookback_bars': 80, 'min_confidence': 0.55, 'state_size': 8, 'action_size': 3, 'memory_size': 100, 'batch_size': 8, 'learning_rate': 0.001, 'epsilon': 0.8, 'epsilon_min': 0.05, 'epsilon_decay': 0.99, 'gamma': 0.9, 'update_target_freq': 10, 'memory_cleanup_interval': 5, 'max_training_frequency': 5}
   RL Specific Metrics:
     - tensorflow_available: True
     - rl_agent_trained: False
     - total_reward_accumulated: 0.00
     - total_episodes_completed: 0
     - total_predictions_made: 1
     - current_epsilon_value: 0.80
     - current_memory_usage: 0
     - total_training_steps: 0
   Overall Trading Performance:
     Total Signals Generated: 0
     Win Rate: 0.00%
     Profit Factor: 0.00

============================================================
REINFORCEMENT LEARNING AGENT STRATEGY TEST COMPLETED!
============================================================

STDERR OUTPUT:
2025-08-23 23:04:37.657558: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
2025-08-23 23:04:42.332212: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
2025-08-23 23:04:44,073 - src.core.mt5_manager - INFO - Loaded 227 tradable symbols from CSV
2025-08-23 23:04:44,073 - src.core.mt5_manager - INFO - MT5Manager initialized for symbol: XAUUSD
2025-08-23 23:04:44,073 - src.core.mt5_manager - INFO - Magic number: 123456
2025-08-23 23:04:44,073 - src.core.mt5_manager - INFO - Initializing MT5 with default path
2025-08-23 23:04:44,086 - src.core.mt5_manager - INFO - Attempting to login to account 273949055 on server Exness-MT5Trial6
2025-08-23 23:04:44,086 - src.core.mt5_manager - WARNING - Symbol XAUUSD not found, trying alternative symbols
2025-08-23 23:04:44,087 - src.core.mt5_manager - INFO - Using alternative symbol: XAUUSDm
2025-08-23 23:04:44,087 - src.core.mt5_manager - INFO - ✅ Successfully connected to MT5
2025-08-23 23:04:44,087 - src.core.mt5_manager - INFO -    Account: 273949055
2025-08-23 23:04:44,087 - src.core.mt5_manager - INFO -    Balance: $194.55
2025-08-23 23:04:44,087 - src.core.mt5_manager - INFO -    Server: Exness-MT5Trial6
2025-08-23 23:04:44,087 - src.core.mt5_manager - INFO -    Symbol: XAUUSDm
2025-08-23 23:04:44.091517: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: SSE3 SSE4.1 SSE4.2 AVX AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
2025-08-23 23:04:44,246 - RLAgentStrategy - INFO - RL networks initialized
2025-08-23 23:04:44,246 - RLAgentStrategy - INFO - RLAgentStrategy initialized (TensorFlow available: True)
2025-08-23 23:04:44,246 - RLAgentStrategy - INFO - RL Agent - Analyzing XAUUSDm on M15
2025-08-23 23:04:44,250 - src.core.mt5_manager - INFO - Retrieved 80 bars for XAUUSDm M15
2025-08-23 23:04:44,435 - RLAgentStrategy - INFO - No valid signals generated
2025-08-23 23:04:44,439 - src.core.mt5_manager - INFO - Retrieved 90 bars for XAUUSDm M15

------------------------------------------------------------

================================================================================

### Strategy: src/strategies/ml/xgboost_classifier.py
### Strategy Name: xgboost_classifier

--- Mode: MOCK ---
STDOUT OUTPUT:
============================================================
RUN MODE: MOCK - using simulated OHLCV data
============================================================
============================================================
TESTING MODIFIED XGBOOST CLASSIFIER STRATEGY
============================================================
Running in MOCK mode

1. Testing signal generation:
   Generated 0 signals

2. Testing analysis method:
   Analysis results keys: dict_keys(['strategy', 'symbol', 'timeframe', 'xgboost_available', 'model_trained', 'model_accuracy', 'predictions_attempted', 'memory_optimized', 'lookback_bars', 'max_training_samples', 'latest_training_time'])
   XGBoost Available: True
   Model Trained: True
   Model Accuracy: 0.538

3. Testing performance tracking:
   {'strategy_name': 'XGBoostClassifierStrategy', 'total_signals': 0, 'win_rate': 0.0, 'profit_factor': 0.0, 'total_pnl': 0.0, 'grade_distribution': {'A': 0, 'B': 0, 'C': 0, 'D': 0}, 'last_signal': None}

4. Strategy Information:
   Name: XGBoost Classifier Strategy
   Version: 1.0.0
   Type: Machine Learning
   Description: Advanced XGBoost-based trading strategy for XAUUSD signal generation.
   Parameters:
     - lookback_bars: 120
     - min_confidence: 0.6
     - xgb_params: {'objective': 'multi:softprob', 'num_class': 3, 'max_depth': 3, 'learning_rate': 0.2, 'n_estimators': 10, 'random_state': 42, 'max_leaves': 15, 'tree_method': 'exact'}
     - memory_cleanup_interval: 10
     - max_training_samples: 500
   ML Specific Metrics:
     - xgboost_available: True
     - model_trained: True
     - model_accuracy: 0.538
     - predictions_attempted: 1
   Overall Trading Performance:
     Total Signals Generated: 0
     Win Rate: 0.00%
     Profit Factor: 0.00

============================================================
XGBOOST CLASSIFIER STRATEGY TEST COMPLETED!
============================================================

STDERR OUTPUT:
2025-08-23 23:04:48,210 - XGBoostClassifierStrategy - INFO - XGBoost model initialized
2025-08-23 23:04:48,210 - XGBoostClassifierStrategy - INFO - XGBoostClassifierStrategy initialized (XGBoost available: True)
2025-08-23 23:04:48,210 - XGBoostClassifierStrategy - INFO - XGBoost Classifier - Analyzing XAUUSDm on M15
2025-08-23 23:04:48,214 - XGBoostClassifierStrategy - INFO - Training XGBoost model...
2025-08-23 23:04:48,214 - XGBoostClassifierStrategy - INFO - Training XGBoost model...
2025-08-23 23:04:48,656 - XGBoostClassifierStrategy - INFO - Model trained. Accuracy: 0.538
2025-08-23 23:04:48,670 - XGBoostClassifierStrategy - INFO - No valid signals generated

------------------------------------------------------------

--- Mode: LIVE ---
STDOUT OUTPUT:
============================================================
RUN MODE: LIVE - connecting to MT5 (production)
============================================================
✅ Connected to live MT5
============================================================
TESTING MODIFIED XGBOOST CLASSIFIER STRATEGY
============================================================
Running in LIVE mode

1. Testing signal generation:
   Generated 0 signals

2. Testing analysis method:
   Analysis results keys: dict_keys(['strategy', 'symbol', 'timeframe', 'xgboost_available', 'model_trained', 'model_accuracy', 'predictions_attempted', 'memory_optimized', 'lookback_bars', 'max_training_samples', 'latest_training_time'])
   XGBoost Available: True
   Model Trained: True
   Model Accuracy: 0.923

3. Testing performance tracking:
   {'strategy_name': 'XGBoostClassifierStrategy', 'total_signals': 0, 'win_rate': 0.0, 'profit_factor': 0.0, 'total_pnl': 0.0, 'grade_distribution': {'A': 0, 'B': 0, 'C': 0, 'D': 0}, 'last_signal': None}

4. Strategy Information:
   Name: XGBoost Classifier Strategy
   Version: 1.0.0
   Type: Machine Learning
   Description: Advanced XGBoost-based trading strategy for XAUUSD signal generation.
   Parameters:
     - lookback_bars: 120
     - min_confidence: 0.6
     - xgb_params: {'objective': 'multi:softprob', 'num_class': 3, 'max_depth': 3, 'learning_rate': 0.2, 'n_estimators': 10, 'random_state': 42, 'max_leaves': 15, 'tree_method': 'exact'}
     - memory_cleanup_interval: 10
     - max_training_samples: 500
   ML Specific Metrics:
     - xgboost_available: True
     - model_trained: True
     - model_accuracy: 0.923
     - predictions_attempted: 1
   Overall Trading Performance:
     Total Signals Generated: 0
     Win Rate: 0.00%
     Profit Factor: 0.00

============================================================
XGBOOST CLASSIFIER STRATEGY TEST COMPLETED!
============================================================

STDERR OUTPUT:
2025-08-23 23:04:51,550 - src.core.mt5_manager - INFO - Loaded 227 tradable symbols from CSV
2025-08-23 23:04:51,550 - src.core.mt5_manager - INFO - MT5Manager initialized for symbol: XAUUSD
2025-08-23 23:04:51,550 - src.core.mt5_manager - INFO - Magic number: 123456
2025-08-23 23:04:51,550 - src.core.mt5_manager - INFO - Initializing MT5 with default path
2025-08-23 23:04:51,562 - src.core.mt5_manager - INFO - Attempting to login to account 273949055 on server Exness-MT5Trial6
2025-08-23 23:04:51,562 - src.core.mt5_manager - WARNING - Symbol XAUUSD not found, trying alternative symbols
2025-08-23 23:04:51,562 - src.core.mt5_manager - INFO - Using alternative symbol: XAUUSDm
2025-08-23 23:04:51,562 - src.core.mt5_manager - INFO - ✅ Successfully connected to MT5
2025-08-23 23:04:51,562 - src.core.mt5_manager - INFO -    Account: 273949055
2025-08-23 23:04:51,562 - src.core.mt5_manager - INFO -    Balance: $194.55
2025-08-23 23:04:51,562 - src.core.mt5_manager - INFO -    Server: Exness-MT5Trial6
2025-08-23 23:04:51,562 - src.core.mt5_manager - INFO -    Symbol: XAUUSDm
2025-08-23 23:04:51,563 - XGBoostClassifierStrategy - INFO - XGBoost model initialized
2025-08-23 23:04:51,563 - XGBoostClassifierStrategy - INFO - XGBoostClassifierStrategy initialized (XGBoost available: True)
2025-08-23 23:04:51,563 - XGBoostClassifierStrategy - INFO - XGBoost Classifier - Analyzing XAUUSDm on M15
2025-08-23 23:04:51,566 - src.core.mt5_manager - INFO - Retrieved 120 bars for XAUUSDm M15
2025-08-23 23:04:51,568 - XGBoostClassifierStrategy - INFO - Training XGBoost model...
2025-08-23 23:04:51,568 - XGBoostClassifierStrategy - INFO - Training XGBoost model...
2025-08-23 23:04:51,929 - XGBoostClassifierStrategy - INFO - Model trained. Accuracy: 0.923
2025-08-23 23:04:51,939 - XGBoostClassifierStrategy - INFO - No valid signals generated
2025-08-23 23:04:51,942 - src.core.mt5_manager - INFO - Retrieved 120 bars for XAUUSDm M15

------------------------------------------------------------

================================================================================

### Strategy: src/strategies/smc/liquidity_pools.py
### Strategy Name: liquidity_pools

--- Mode: MOCK ---
STDOUT OUTPUT:
============================================================
RUN MODE: MOCK - using simulated OHLCV data
============================================================
============================================================
TESTING LIQUIDITY POOLS STRATEGY
============================================================
Running in MOCK mode

1. Testing signal generation:
   Generated 141 signals
   - SELL at 1951.26, Confidence: 0.850, Grade: A
     Pool Type: EQUAL_HIGHS
     Setup: sweep_reversal
   - SELL at 1951.26, Confidence: 0.850, Grade: A
     Pool Type: EQUAL_HIGHS
     Setup: sweep_reversal
   - SELL at 1951.26, Confidence: 0.850, Grade: A
     Pool Type: EQUAL_HIGHS
     Setup: sweep_reversal
   - SELL at 1951.26, Confidence: 0.850, Grade: A
     Pool Type: EQUAL_HIGHS
     Setup: sweep_reversal
   - SELL at 1951.26, Confidence: 0.850, Grade: A
     Pool Type: EQUAL_HIGHS
     Setup: sweep_reversal
   - SELL at 1951.26, Confidence: 0.850, Grade: A
     Pool Type: EQUAL_HIGHS
     Setup: sweep_reversal
   - SELL at 1951.26, Confidence: 0.850, Grade: A
     Pool Type: EQUAL_HIGHS
     Setup: sweep_reversal
   - SELL at 1951.26, Confidence: 0.850, Grade: A
     Pool Type: EQUAL_HIGHS
     Setup: sweep_reversal
   - SELL at 1951.26, Confidence: 0.850, Grade: A
     Pool Type: EQUAL_HIGHS
     Setup: sweep_reversal
   - SELL at 1951.26, Confidence: 0.850, Grade: A
     Pool Type: EQUAL_HIGHS
     Setup: sweep_reversal
   - SELL at 1951.26, Confidence: 0.850, Grade: A
     Pool Type: EQUAL_HIGHS
     Setup: sweep_reversal
   - SELL at 1951.26, Confidence: 0.850, Grade: A
     Pool Type: EQUAL_HIGHS
     Setup: sweep_reversal
   - SELL at 1951.26, Confidence: 0.850, Grade: A
     Pool Type: EQUAL_HIGHS
     Setup: sweep_reversal
   - SELL at 1951.26, Confidence: 0.850, Grade: A
     Pool Type: EQUAL_HIGHS
     Setup: sweep_reversal
   - SELL at 1951.26, Confidence: 0.850, Grade: A
     Pool Type: EQUAL_HIGHS
     Setup: sweep_reversal
   - SELL at 1951.26, Confidence: 0.850, Grade: A
     Pool Type: EQUAL_HIGHS
     Setup: sweep_reversal
   - SELL at 1951.26, Confidence: 0.850, Grade: A
     Pool Type: EQUAL_HIGHS
     Setup: sweep_reversal
   - SELL at 1951.26, Confidence: 0.850, Grade: A
     Pool Type: EQUAL_HIGHS
     Setup: sweep_reversal
   - SELL at 1951.26, Confidence: 0.850, Grade: A
     Pool Type: EQUAL_HIGHS
     Setup: sweep_reversal
   - SELL at 1951.26, Confidence: 0.850, Grade: A
     Pool Type: EQUAL_HIGHS
     Setup: sweep_reversal
   - SELL at 1951.26, Confidence: 0.850, Grade: A
     Pool Type: EQUAL_HIGHS
     Setup: sweep_reversal
   - SELL at 1951.26, Confidence: 0.850, Grade: A
     Pool Type: EQUAL_HIGHS
     Setup: sweep_reversal
   - SELL at 1951.26, Confidence: 0.850, Grade: A
     Pool Type: EQUAL_HIGHS
     Setup: sweep_reversal
   - SELL at 1951.26, Confidence: 0.850, Grade: A
     Pool Type: EQUAL_HIGHS
     Setup: sweep_reversal
   - SELL at 1951.26, Confidence: 0.850, Grade: A
     Pool Type: EQUAL_HIGHS
     Setup: sweep_reversal
   - SELL at 1951.26, Confidence: 0.850, Grade: A
     Pool Type: EQUAL_HIGHS
     Setup: sweep_reversal
   - SELL at 1951.26, Confidence: 0.850, Grade: A
     Pool Type: EQUAL_HIGHS
     Setup: sweep_reversal
   - SELL at 1951.26, Confidence: 0.850, Grade: A
     Pool Type: EQUAL_HIGHS
     Setup: sweep_reversal
   - SELL at 1951.26, Confidence: 0.850, Grade: A
     Pool Type: EQUAL_HIGHS
     Setup: sweep_reversal
   - SELL at 1951.26, Confidence: 0.850, Grade: A
     Pool Type: EQUAL_HIGHS
     Setup: sweep_reversal
   - SELL at 1951.26, Confidence: 0.850, Grade: A
     Pool Type: EQUAL_HIGHS
     Setup: sweep_reversal
   - SELL at 1951.26, Confidence: 0.850, Grade: A
     Pool Type: EQUAL_HIGHS
     Setup: sweep_reversal
   - SELL at 1951.26, Confidence: 0.850, Grade: A
     Pool Type: EQUAL_HIGHS
     Setup: sweep_reversal
   - SELL at 1951.26, Confidence: 0.850, Grade: A
     Pool Type: EQUAL_HIGHS
     Setup: sweep_reversal
   - SELL at 1951.26, Confidence: 0.850, Grade: A
     Pool Type: EQUAL_HIGHS
     Setup: sweep_reversal
   - SELL at 1951.26, Confidence: 0.850, Grade: A
     Pool Type: EQUAL_HIGHS
     Setup: sweep_reversal
   - SELL at 1951.26, Confidence: 0.850, Grade: A
     Pool Type: EQUAL_HIGHS
     Setup: sweep_reversal
   - SELL at 1951.26, Confidence: 0.850, Grade: A
     Pool Type: EQUAL_HIGHS
     Setup: sweep_reversal
   - SELL at 1951.26, Confidence: 0.850, Grade: A
     Pool Type: EQUAL_HIGHS
     Setup: sweep_reversal
   - SELL at 1951.26, Confidence: 0.850, Grade: A
     Pool Type: EQUAL_HIGHS
     Setup: sweep_reversal
   - SELL at 1951.26, Confidence: 0.850, Grade: A
     Pool Type: EQUAL_HIGHS
     Setup: sweep_reversal
   - SELL at 1951.26, Confidence: 0.850, Grade: A
     Pool Type: EQUAL_HIGHS
     Setup: sweep_reversal
   - SELL at 1951.26, Confidence: 0.850, Grade: A
     Pool Type: EQUAL_HIGHS
     Setup: sweep_reversal
   - SELL at 1951.26, Confidence: 0.850, Grade: A
     Pool Type: EQUAL_HIGHS
     Setup: sweep_reversal
   - SELL at 1951.26, Confidence: 0.850, Grade: A
     Pool Type: EQUAL_HIGHS
     Setup: sweep_reversal
   - SELL at 1951.26, Confidence: 0.850, Grade: A
     Pool Type: EQUAL_HIGHS
     Setup: sweep_reversal
   - SELL at 1951.26, Confidence: 0.850, Grade: A
     Pool Type: EQUAL_HIGHS
     Setup: sweep_reversal
   - SELL at 1951.26, Confidence: 0.850, Grade: A
     Pool Type: EQUAL_HIGHS
     Setup: sweep_reversal
   - SELL at 1951.26, Confidence: 0.850, Grade: A
     Pool Type: EQUAL_HIGHS
     Setup: sweep_reversal
   - SELL at 1951.26, Confidence: 0.850, Grade: A
     Pool Type: EQUAL_HIGHS
     Setup: sweep_reversal
   - SELL at 1951.26, Confidence: 0.850, Grade: A
     Pool Type: EQUAL_HIGHS
     Setup: sweep_reversal
   - SELL at 1951.26, Confidence: 0.850, Grade: A
     Pool Type: EQUAL_HIGHS
     Setup: sweep_reversal
   - SELL at 1951.26, Confidence: 0.850, Grade: A
     Pool Type: EQUAL_HIGHS
     Setup: sweep_reversal
   - SELL at 1951.26, Confidence: 0.850, Grade: A
     Pool Type: EQUAL_HIGHS
     Setup: sweep_reversal
   - SELL at 1951.26, Confidence: 0.850, Grade: A
     Pool Type: EQUAL_HIGHS
     Setup: sweep_reversal
   - SELL at 1951.26, Confidence: 0.850, Grade: A
     Pool Type: EQUAL_HIGHS
     Setup: sweep_reversal
   - SELL at 1951.26, Confidence: 0.850, Grade: A
     Pool Type: EQUAL_HIGHS
     Setup: sweep_reversal
   - BUY at 1951.26, Confidence: 0.850, Grade: A
     Pool Type: EQUAL_LOWS
     Setup: sweep_reversal
   - BUY at 1951.26, Confidence: 0.850, Grade: A
     Pool Type: EQUAL_LOWS
     Setup: sweep_reversal
   - BUY at 1951.26, Confidence: 0.850, Grade: A
     Pool Type: EQUAL_LOWS
     Setup: sweep_reversal
   - BUY at 1951.26, Confidence: 0.850, Grade: A
     Pool Type: EQUAL_LOWS
     Setup: sweep_reversal
   - BUY at 1951.26, Confidence: 0.850, Grade: A
     Pool Type: EQUAL_LOWS
     Setup: sweep_reversal
   - BUY at 1951.26, Confidence: 0.850, Grade: A
     Pool Type: EQUAL_LOWS
     Setup: sweep_reversal
   - BUY at 1951.26, Confidence: 0.850, Grade: A
     Pool Type: EQUAL_LOWS
     Setup: sweep_reversal
   - BUY at 1951.26, Confidence: 0.850, Grade: A
     Pool Type: EQUAL_LOWS
     Setup: sweep_reversal
   - BUY at 1951.26, Confidence: 0.850, Grade: A
     Pool Type: EQUAL_LOWS
     Setup: sweep_reversal
   - BUY at 1951.26, Confidence: 0.850, Grade: A
     Pool Type: EQUAL_LOWS
     Setup: sweep_reversal
   - BUY at 1951.26, Confidence: 0.850, Grade: A
     Pool Type: EQUAL_LOWS
     Setup: sweep_reversal
   - BUY at 1951.26, Confidence: 0.850, Grade: A
     Pool Type: EQUAL_LOWS
     Setup: sweep_reversal
   - BUY at 1951.26, Confidence: 0.850, Grade: A
     Pool Type: EQUAL_LOWS
     Setup: sweep_reversal
   - BUY at 1951.26, Confidence: 0.850, Grade: A
     Pool Type: EQUAL_LOWS
     Setup: sweep_reversal
   - BUY at 1951.26, Confidence: 0.850, Grade: A
     Pool Type: EQUAL_LOWS
     Setup: sweep_reversal
   - BUY at 1951.26, Confidence: 0.850, Grade: A
     Pool Type: EQUAL_LOWS
     Setup: sweep_reversal
   - BUY at 1951.26, Confidence: 0.850, Grade: A
     Pool Type: EQUAL_LOWS
     Setup: sweep_reversal
   - BUY at 1951.26, Confidence: 0.850, Grade: A
     Pool Type: EQUAL_LOWS
     Setup: sweep_reversal
   - BUY at 1951.26, Confidence: 0.850, Grade: A
     Pool Type: EQUAL_LOWS
     Setup: sweep_reversal
   - BUY at 1951.26, Confidence: 0.850, Grade: A
     Pool Type: EQUAL_LOWS
     Setup: sweep_reversal
   - BUY at 1951.26, Confidence: 0.850, Grade: A
     Pool Type: EQUAL_LOWS
     Setup: sweep_reversal
   - BUY at 1951.26, Confidence: 0.850, Grade: A
     Pool Type: EQUAL_LOWS
     Setup: sweep_reversal
   - BUY at 1951.26, Confidence: 0.850, Grade: A
     Pool Type: EQUAL_LOWS
     Setup: sweep_reversal
   - BUY at 1951.26, Confidence: 0.850, Grade: A
     Pool Type: EQUAL_LOWS
     Setup: sweep_reversal
   - BUY at 1951.26, Confidence: 0.850, Grade: A
     Pool Type: EQUAL_LOWS
     Setup: sweep_reversal
   - BUY at 1951.26, Confidence: 0.850, Grade: A
     Pool Type: EQUAL_LOWS
     Setup: sweep_reversal
   - BUY at 1951.26, Confidence: 0.850, Grade: A
     Pool Type: EQUAL_LOWS
     Setup: sweep_reversal
   - BUY at 1951.26, Confidence: 0.850, Grade: A
     Pool Type: EQUAL_LOWS
     Setup: sweep_reversal
   - BUY at 1951.26, Confidence: 0.850, Grade: A
     Pool Type: EQUAL_LOWS
     Setup: sweep_reversal
   - BUY at 1951.26, Confidence: 0.850, Grade: A
     Pool Type: EQUAL_LOWS
     Setup: sweep_reversal
   - BUY at 1951.26, Confidence: 0.850, Grade: A
     Pool Type: EQUAL_LOWS
     Setup: sweep_reversal
   - BUY at 1951.26, Confidence: 0.850, Grade: A
     Pool Type: EQUAL_LOWS
     Setup: sweep_reversal
   - BUY at 1951.26, Confidence: 0.850, Grade: A
     Pool Type: EQUAL_LOWS
     Setup: sweep_reversal
   - BUY at 1951.26, Confidence: 0.850, Grade: A
     Pool Type: EQUAL_LOWS
     Setup: sweep_reversal
   - BUY at 1951.26, Confidence: 0.850, Grade: A
     Pool Type: EQUAL_LOWS
     Setup: sweep_reversal
   - BUY at 1951.26, Confidence: 0.850, Grade: A
     Pool Type: EQUAL_LOWS
     Setup: sweep_reversal
   - BUY at 1951.26, Confidence: 0.850, Grade: A
     Pool Type: EQUAL_LOWS
     Setup: sweep_reversal
   - BUY at 1951.26, Confidence: 0.850, Grade: A
     Pool Type: EQUAL_LOWS
     Setup: sweep_reversal
   - BUY at 1951.26, Confidence: 0.850, Grade: A
     Pool Type: EQUAL_LOWS
     Setup: sweep_reversal
   - BUY at 1951.26, Confidence: 0.850, Grade: A
     Pool Type: EQUAL_LOWS
     Setup: sweep_reversal
   - BUY at 1951.26, Confidence: 0.850, Grade: A
     Pool Type: EQUAL_LOWS
     Setup: sweep_reversal
   - BUY at 1951.26, Confidence: 0.850, Grade: A
     Pool Type: EQUAL_LOWS
     Setup: sweep_reversal
   - SELL at 1951.26, Confidence: 0.800, Grade: B
     Pool Type: EQUAL_LOWS
     Setup: break_continuation
   - BUY at 1951.26, Confidence: 0.850, Grade: A
     Pool Type: EQUAL_LOWS
     Setup: sweep_reversal
   - BUY at 1951.26, Confidence: 0.850, Grade: A
     Pool Type: EQUAL_LOWS
     Setup: sweep_reversal
   - BUY at 1951.26, Confidence: 0.850, Grade: A
     Pool Type: EQUAL_LOWS
     Setup: sweep_reversal
   - BUY at 1951.26, Confidence: 0.850, Grade: A
     Pool Type: EQUAL_LOWS
     Setup: sweep_reversal
   - BUY at 1951.26, Confidence: 0.850, Grade: A
     Pool Type: EQUAL_LOWS
     Setup: sweep_reversal
   - BUY at 1951.26, Confidence: 0.850, Grade: A
     Pool Type: EQUAL_LOWS
     Setup: sweep_reversal
   - BUY at 1951.26, Confidence: 0.850, Grade: A
     Pool Type: EQUAL_LOWS
     Setup: sweep_reversal
   - BUY at 1951.26, Confidence: 0.850, Grade: A
     Pool Type: EQUAL_LOWS
     Setup: sweep_reversal
   - BUY at 1951.26, Confidence: 0.850, Grade: A
     Pool Type: EQUAL_LOWS
     Setup: sweep_reversal
   - BUY at 1951.26, Confidence: 0.850, Grade: A
     Pool Type: EQUAL_LOWS
     Setup: sweep_reversal
   - SELL at 1951.26, Confidence: 0.800, Grade: B
     Pool Type: EQUAL_LOWS
     Setup: break_continuation
   - BUY at 1951.26, Confidence: 0.850, Grade: A
     Pool Type: EQUAL_LOWS
     Setup: sweep_reversal
   - BUY at 1951.26, Confidence: 0.850, Grade: A
     Pool Type: EQUAL_LOWS
     Setup: sweep_reversal
   - BUY at 1951.26, Confidence: 0.850, Grade: A
     Pool Type: EQUAL_LOWS
     Setup: sweep_reversal
   - BUY at 1951.26, Confidence: 0.850, Grade: A
     Pool Type: EQUAL_LOWS
     Setup: sweep_reversal
   - BUY at 1951.26, Confidence: 0.850, Grade: A
     Pool Type: EQUAL_LOWS
     Setup: sweep_reversal
   - BUY at 1951.26, Confidence: 0.850, Grade: A
     Pool Type: EQUAL_LOWS
     Setup: sweep_reversal
   - BUY at 1951.26, Confidence: 0.850, Grade: A
     Pool Type: EQUAL_LOWS
     Setup: sweep_reversal
   - BUY at 1951.26, Confidence: 0.850, Grade: A
     Pool Type: EQUAL_LOWS
     Setup: sweep_reversal
   - BUY at 1951.26, Confidence: 0.850, Grade: A
     Pool Type: EQUAL_LOWS
     Setup: sweep_reversal
   - BUY at 1951.26, Confidence: 0.850, Grade: A
     Pool Type: EQUAL_LOWS
     Setup: sweep_reversal
   - BUY at 1951.26, Confidence: 0.850, Grade: A
     Pool Type: EQUAL_LOWS
     Setup: sweep_reversal
   - BUY at 1951.26, Confidence: 0.850, Grade: A
     Pool Type: EQUAL_LOWS
     Setup: sweep_reversal
   - BUY at 1951.26, Confidence: 0.850, Grade: A
     Pool Type: EQUAL_LOWS
     Setup: sweep_reversal
   - BUY at 1951.26, Confidence: 0.850, Grade: A
     Pool Type: EQUAL_LOWS
     Setup: sweep_reversal
   - BUY at 1951.26, Confidence: 0.850, Grade: A
     Pool Type: EQUAL_LOWS
     Setup: sweep_reversal
   - BUY at 1951.26, Confidence: 0.850, Grade: A
     Pool Type: EQUAL_LOWS
     Setup: sweep_reversal
   - BUY at 1951.26, Confidence: 0.850, Grade: A
     Pool Type: EQUAL_LOWS
     Setup: sweep_reversal
   - BUY at 1951.26, Confidence: 0.850, Grade: A
     Pool Type: EQUAL_LOWS
     Setup: sweep_reversal
   - BUY at 1951.26, Confidence: 0.850, Grade: A
     Pool Type: EQUAL_LOWS
     Setup: sweep_reversal
   - BUY at 1951.26, Confidence: 0.850, Grade: A
     Pool Type: EQUAL_LOWS
     Setup: sweep_reversal
   - BUY at 1951.26, Confidence: 0.850, Grade: A
     Pool Type: EQUAL_LOWS
     Setup: sweep_reversal
   - BUY at 1951.26, Confidence: 0.850, Grade: A
     Pool Type: EQUAL_LOWS
     Setup: sweep_reversal
   - BUY at 1951.26, Confidence: 0.850, Grade: A
     Pool Type: EQUAL_LOWS
     Setup: sweep_reversal
   - BUY at 1951.26, Confidence: 0.850, Grade: A
     Pool Type: EQUAL_LOWS
     Setup: sweep_reversal
   - BUY at 1951.26, Confidence: 0.850, Grade: A
     Pool Type: EQUAL_LOWS
     Setup: sweep_reversal
   - BUY at 1951.26, Confidence: 0.850, Grade: A
     Pool Type: EQUAL_LOWS
     Setup: sweep_reversal
   - BUY at 1951.26, Confidence: 0.850, Grade: A
     Pool Type: EQUAL_LOWS
     Setup: sweep_reversal
   - BUY at 1951.26, Confidence: 0.850, Grade: A
     Pool Type: EQUAL_LOWS
     Setup: sweep_reversal
   - BUY at 1951.26, Confidence: 0.850, Grade: A
     Pool Type: EQUAL_LOWS
     Setup: sweep_reversal
   - BUY at 1951.26, Confidence: 0.850, Grade: A
     Pool Type: EQUAL_LOWS
     Setup: sweep_reversal

2. Testing analysis method:
   Analysis results keys: ['pools', 'recent_sweeps', 'current_price']
   Detected pools: 636
     - SWING_HIGH: 1963.77 (Strength: 0.01)
     - SWING_LOW: 1935.47 (Strength: 0.01)
     - SWING_HIGH: 1963.54 (Strength: 0.01)

3. Testing performance tracking:
   {'strategy_name': 'LiquidityPoolsStrategy', 'total_signals': 0, 'win_rate': 0.0, 'profit_factor': 0.0, 'total_pnl': 0.0, 'grade_distribution': {'A': 0, 'B': 0, 'C': 0, 'D': 0}, 'last_signal': None}

4. Strategy Information:
   Name: LiquidityPoolsStrategy
   Version: 1.0.0
   Description: Liquidity Pools Strategy identifying swing highs/lows, equal highs/lows, and session levels for sweep-reversal and break-continuation trades
   Type: SMC
   Parameters:
     lookback_bars: 300
     equal_highs_tolerance: 0.12
     approach_buffer: 0.2
     confidence_threshold: 0.65
     cooldown_bars: 3
   Performance Summary:
     Success Rate: 0.00%
     Profit Factor: 0.00

============================================================
LIQUIDITY POOLS STRATEGY TEST COMPLETED!
============================================================

------------------------------------------------------------

--- Mode: LIVE ---
STDOUT OUTPUT:
============================================================
RUN MODE: LIVE - connecting to MT5 (production)
============================================================
✅ Connected to live MT5
============================================================
TESTING LIQUIDITY POOLS STRATEGY
============================================================
Running in LIVE mode

1. Testing signal generation:
   Generated 15 signals
   - SELL at 3372.30, Confidence: 0.850, Grade: A
     Pool Type: EQUAL_HIGHS
     Setup: sweep_reversal
   - SELL at 3372.30, Confidence: 0.850, Grade: A
     Pool Type: EQUAL_HIGHS
     Setup: sweep_reversal
   - BUY at 3372.30, Confidence: 0.800, Grade: B
     Pool Type: EQUAL_HIGHS
     Setup: break_continuation
   - BUY at 3372.30, Confidence: 0.800, Grade: B
     Pool Type: EQUAL_HIGHS
     Setup: break_continuation
   - BUY at 3372.30, Confidence: 0.800, Grade: B
     Pool Type: EQUAL_HIGHS
     Setup: break_continuation
   - SELL at 3372.30, Confidence: 0.850, Grade: A
     Pool Type: EQUAL_HIGHS
     Setup: sweep_reversal
   - BUY at 3372.30, Confidence: 0.767, Grade: B
     Pool Type: EQUAL_HIGHS
     Setup: break_continuation
   - BUY at 3372.30, Confidence: 0.850, Grade: A
     Pool Type: EQUAL_LOWS
     Setup: sweep_reversal
   - BUY at 3372.30, Confidence: 0.850, Grade: A
     Pool Type: EQUAL_LOWS
     Setup: sweep_reversal
   - BUY at 3372.30, Confidence: 0.850, Grade: A
     Pool Type: EQUAL_LOWS
     Setup: sweep_reversal
   - BUY at 3372.30, Confidence: 0.850, Grade: A
     Pool Type: EQUAL_LOWS
     Setup: sweep_reversal
   - BUY at 3372.30, Confidence: 0.850, Grade: A
     Pool Type: EQUAL_LOWS
     Setup: sweep_reversal
   - BUY at 3372.30, Confidence: 0.850, Grade: A
     Pool Type: EQUAL_LOWS
     Setup: sweep_reversal
   - BUY at 3372.30, Confidence: 0.850, Grade: A
     Pool Type: EQUAL_LOWS
     Setup: sweep_reversal
   - BUY at 3372.30, Confidence: 0.850, Grade: A
     Pool Type: EQUAL_LOWS
     Setup: sweep_reversal

2. Testing analysis method:
   Analysis results keys: ['pools', 'recent_sweeps', 'current_price']
   Detected pools: 628
     - SWING_LOW: 3314.85 (Strength: 0.00)
     - SWING_HIGH: 3320.20 (Strength: 0.00)
     - SWING_LOW: 3314.89 (Strength: 0.00)

3. Testing performance tracking:
   {'strategy_name': 'LiquidityPoolsStrategy', 'total_signals': 0, 'win_rate': 0.0, 'profit_factor': 0.0, 'total_pnl': 0.0, 'grade_distribution': {'A': 0, 'B': 0, 'C': 0, 'D': 0}, 'last_signal': None}

4. Strategy Information:
   Name: LiquidityPoolsStrategy
   Version: 1.0.0
   Description: Liquidity Pools Strategy identifying swing highs/lows, equal highs/lows, and session levels for sweep-reversal and break-continuation trades
   Type: SMC
   Parameters:
     lookback_bars: 300
     equal_highs_tolerance: 0.12
     approach_buffer: 0.2
     confidence_threshold: 0.65
     cooldown_bars: 3
   Performance Summary:
     Success Rate: 0.00%
     Profit Factor: 0.00

============================================================
LIQUIDITY POOLS STRATEGY TEST COMPLETED!
============================================================

STDERR OUTPUT:
2025-08-23 23:04:54,456 - src.core.mt5_manager - INFO - Loaded 227 tradable symbols from CSV
2025-08-23 23:04:54,456 - src.core.mt5_manager - INFO - MT5Manager initialized for symbol: XAUUSD
2025-08-23 23:04:54,456 - src.core.mt5_manager - INFO - Magic number: 123456
2025-08-23 23:04:54,456 - src.core.mt5_manager - INFO - Initializing MT5 with default path
2025-08-23 23:04:54,471 - src.core.mt5_manager - INFO - Attempting to login to account 273949055 on server Exness-MT5Trial6
2025-08-23 23:04:54,471 - src.core.mt5_manager - WARNING - Symbol XAUUSD not found, trying alternative symbols
2025-08-23 23:04:54,472 - src.core.mt5_manager - INFO - Using alternative symbol: XAUUSDm
2025-08-23 23:04:54,472 - src.core.mt5_manager - INFO - ✅ Successfully connected to MT5
2025-08-23 23:04:54,472 - src.core.mt5_manager - INFO -    Account: 273949055
2025-08-23 23:04:54,472 - src.core.mt5_manager - INFO -    Balance: $194.55
2025-08-23 23:04:54,472 - src.core.mt5_manager - INFO -    Server: Exness-MT5Trial6
2025-08-23 23:04:54,472 - src.core.mt5_manager - INFO -    Symbol: XAUUSDm
2025-08-23 23:04:54,472 - LiquidityPoolsStrategy - INFO - Liquidity Pools Strategy - Analyzing XAUUSDm on M15
2025-08-23 23:04:54,476 - src.core.mt5_manager - INFO - Retrieved 300 bars for XAUUSDm M15
2025-08-23 23:04:54,660 - LiquidityPoolsStrategy - INFO - Generated 15 signals
2025-08-23 23:04:54,663 - src.core.mt5_manager - INFO - Retrieved 300 bars for XAUUSDm M15

------------------------------------------------------------

================================================================================

### Strategy: src/strategies/smc/manipulation.py
### Strategy Name: manipulation

--- Mode: MOCK ---
STDOUT OUTPUT:
============================================================
RUN MODE: MOCK - using simulated OHLCV data
============================================================
============================================================
TESTING MANIPULATION STRATEGY
============================================================
Running in MOCK mode

1. Testing signal generation:
   Generated 0 signals

2. Testing analysis method:
   Analysis results keys: dict_keys(['manipulation_events', 'recent_levels', 'current_price'])
   Detected manipulation events: 2
     - displacement at 1952.78 (bar 241)
     - displacement at 1948.88 (bar 246)

3. Testing performance tracking:
   {'strategy_name': 'ManipulationStrategy', 'total_signals': 0, 'win_rate': 0.0, 'profit_factor': 0.0, 'total_pnl': 0.0, 'grade_distribution': {'A': 0, 'B': 0, 'C': 0, 'D': 0}, 'last_signal': None}

4. Strategy Information:
   Name: Manipulation Strategy
   Type: SMC
   Description: Detects stop hunts, fakeouts, and displacement with reversion patterns
   Parameters:
     lookback_bars: 250
     wick_ratio_threshold: 1.5
     fakeout_confirm_bars: 2
     confidence_threshold: 0.65
     cooldown_bars: 3
   Performance Summary:
     Success Rate: 0.00%
     Profit Factor: 0.00

============================================================
MANIPULATION STRATEGY TEST COMPLETED!
============================================================

------------------------------------------------------------

--- Mode: LIVE ---
STDOUT OUTPUT:
============================================================
RUN MODE: LIVE - connecting to MT5 (production)
============================================================
✅ Connected to live MT5
============================================================
TESTING MANIPULATION STRATEGY
============================================================
Running in LIVE mode

1. Testing signal generation:
   Generated 0 signals

2. Testing analysis method:
   Analysis results keys: dict_keys(['manipulation_events', 'recent_levels', 'current_price'])
   Detected manipulation events: 0

3. Testing performance tracking:
   {'strategy_name': 'ManipulationStrategy', 'total_signals': 0, 'win_rate': 0.0, 'profit_factor': 0.0, 'total_pnl': 0.0, 'grade_distribution': {'A': 0, 'B': 0, 'C': 0, 'D': 0}, 'last_signal': None}

4. Strategy Information:
   Name: Manipulation Strategy
   Type: SMC
   Description: Detects stop hunts, fakeouts, and displacement with reversion patterns
   Parameters:
     lookback_bars: 250
     wick_ratio_threshold: 1.5
     fakeout_confirm_bars: 2
     confidence_threshold: 0.65
     cooldown_bars: 3
   Performance Summary:
     Success Rate: 0.00%
     Profit Factor: 0.00

============================================================
MANIPULATION STRATEGY TEST COMPLETED!
============================================================

STDERR OUTPUT:
2025-08-23 23:04:57,114 - src.core.mt5_manager - INFO - Loaded 227 tradable symbols from CSV
2025-08-23 23:04:57,114 - src.core.mt5_manager - INFO - MT5Manager initialized for symbol: XAUUSD
2025-08-23 23:04:57,114 - src.core.mt5_manager - INFO - Magic number: 123456
2025-08-23 23:04:57,114 - src.core.mt5_manager - INFO - Initializing MT5 with default path
2025-08-23 23:04:57,126 - src.core.mt5_manager - INFO - Attempting to login to account 273949055 on server Exness-MT5Trial6
2025-08-23 23:04:57,127 - src.core.mt5_manager - WARNING - Symbol XAUUSD not found, trying alternative symbols
2025-08-23 23:04:57,127 - src.core.mt5_manager - INFO - Using alternative symbol: XAUUSDm
2025-08-23 23:04:57,127 - src.core.mt5_manager - INFO - ✅ Successfully connected to MT5
2025-08-23 23:04:57,127 - src.core.mt5_manager - INFO -    Account: 273949055
2025-08-23 23:04:57,127 - src.core.mt5_manager - INFO -    Balance: $194.55
2025-08-23 23:04:57,127 - src.core.mt5_manager - INFO -    Server: Exness-MT5Trial6
2025-08-23 23:04:57,127 - src.core.mt5_manager - INFO -    Symbol: XAUUSDm
2025-08-23 23:04:57,127 - ManipulationStrategy - INFO - Manipulation Strategy initialized
2025-08-23 23:04:57,127 - ManipulationStrategy - INFO - Manipulation Strategy - Analyzing XAUUSDm on M15
2025-08-23 23:04:57,131 - src.core.mt5_manager - INFO - Retrieved 250 bars for XAUUSDm M15
2025-08-23 23:04:57,230 - ManipulationStrategy - INFO - Generated 0 signals
2025-08-23 23:04:57,233 - src.core.mt5_manager - INFO - Retrieved 250 bars for XAUUSDm M15

------------------------------------------------------------

================================================================================

### Strategy: src/strategies/smc/market_structure.py
### Strategy Name: market_structure

--- Mode: MOCK ---
STDOUT OUTPUT:
============================================================
RUN MODE: MOCK - using simulated OHLCV data
============================================================
============================================================
TESTING MARKET STRUCTURE STRATEGY
============================================================
Running in MOCK mode

1. Testing signal generation:
   Generated 0 signals

2. Testing analysis method:
   Analysis keys: ['trend', 'last_bos', 'last_choch', 'recent_swings']
   Current Trend: RANGE

3. Testing performance tracking:
   {'strategy_name': 'MarketStructureStrategy', 'total_signals': 0, 'win_rate': 0.0, 'profit_factor': 0.0, 'total_pnl': 0.0, 'grade_distribution': {'A': 0, 'B': 0, 'C': 0, 'D': 0}, 'last_signal': None}

4. Strategy Information:
   Name: MarketStructureStrategy
   Type: SMC
   Description: Detects market structure via HH/HL/LH/LL patterns, generating signals on BOS and CHoCH events with retest/pullback confirmation.
   Parameters:
     lookback_bars: 200
     swing_window: 5
     retest_window: 3
     confidence_threshold: 0.65
     cooldown_bars: 3
     swing_tolerance: 0.002
   Performance Summary:
     Success Rate: 0.00%
     Profit Factor: 0.00

============================================================
MARKET STRUCTURE STRATEGY TEST COMPLETED!
============================================================

------------------------------------------------------------

--- Mode: LIVE ---
STDOUT OUTPUT:
============================================================
RUN MODE: LIVE - connecting to MT5 (production)
============================================================
✅ Connected to live MT5
============================================================
TESTING MARKET STRUCTURE STRATEGY
============================================================
Running in LIVE mode

1. Testing signal generation:
   Generated 0 signals

2. Testing analysis method:
   Analysis keys: ['trend', 'last_bos', 'last_choch', 'recent_swings']
   Current Trend: RANGE

3. Testing performance tracking:
   {'strategy_name': 'MarketStructureStrategy', 'total_signals': 0, 'win_rate': 0.0, 'profit_factor': 0.0, 'total_pnl': 0.0, 'grade_distribution': {'A': 0, 'B': 0, 'C': 0, 'D': 0}, 'last_signal': None}

4. Strategy Information:
   Name: MarketStructureStrategy
   Type: SMC
   Description: Detects market structure via HH/HL/LH/LL patterns, generating signals on BOS and CHoCH events with retest/pullback confirmation.
   Parameters:
     lookback_bars: 200
     swing_window: 5
     retest_window: 3
     confidence_threshold: 0.65
     cooldown_bars: 3
     swing_tolerance: 0.002
   Performance Summary:
     Success Rate: 0.00%
     Profit Factor: 0.00

============================================================
MARKET STRUCTURE STRATEGY TEST COMPLETED!
============================================================

STDERR OUTPUT:
2025-08-23 23:04:59,438 - src.core.mt5_manager - INFO - Loaded 227 tradable symbols from CSV
2025-08-23 23:04:59,438 - src.core.mt5_manager - INFO - MT5Manager initialized for symbol: XAUUSD
2025-08-23 23:04:59,438 - src.core.mt5_manager - INFO - Magic number: 123456
2025-08-23 23:04:59,438 - src.core.mt5_manager - INFO - Initializing MT5 with default path
2025-08-23 23:04:59,455 - src.core.mt5_manager - INFO - Attempting to login to account 273949055 on server Exness-MT5Trial6
2025-08-23 23:04:59,456 - src.core.mt5_manager - WARNING - Symbol XAUUSD not found, trying alternative symbols
2025-08-23 23:04:59,456 - src.core.mt5_manager - INFO - Using alternative symbol: XAUUSDm
2025-08-23 23:04:59,457 - src.core.mt5_manager - INFO - ✅ Successfully connected to MT5
2025-08-23 23:04:59,457 - src.core.mt5_manager - INFO -    Account: 273949055
2025-08-23 23:04:59,457 - src.core.mt5_manager - INFO -    Balance: $194.55
2025-08-23 23:04:59,457 - src.core.mt5_manager - INFO -    Server: Exness-MT5Trial6
2025-08-23 23:04:59,457 - src.core.mt5_manager - INFO -    Symbol: XAUUSDm
2025-08-23 23:04:59,457 - MarketStructureStrategy - INFO - MarketStructureStrategy initialized with lookback=200, swing_window=5, confidence_threshold=0.65
2025-08-23 23:04:59,463 - src.core.mt5_manager - INFO - Retrieved 200 bars for XAUUSDm M15
2025-08-23 23:04:59,507 - src.core.mt5_manager - INFO - Retrieved 200 bars for XAUUSDm M15

------------------------------------------------------------

================================================================================

### Strategy: src/strategies/smc/order_blocks.py
### Strategy Name: order_blocks

--- Mode: MOCK ---
STDOUT OUTPUT:
============================================================
RUN MODE: MOCK - using simulated OHLCV data
============================================================
============================================================
TESTING MODIFIED ORDER BLOCKS STRATEGY
============================================================
Running in MOCK mode

1. Testing signal generation:
   Generated 1 signals
   - Signal: BUY at 1956.84, Confidence: 1.00, Grade: A
     Type: BULLISH
     Market Structure: N/A

2. Testing analysis method:
   Analysis results keys: dict_keys(['strategy', 'symbol', 'timeframe', 'analysis_time', 'data_points', 'current_market_structure', 'active_order_blocks_count', 'active_fair_value_gaps_count', 'recent_order_blocks', 'recent_fair_value_gaps'])
   Current Market Structure: RANGING
   Active Order Blocks: 0
   Active FVGs: 3

3. Testing performance tracking:
   {'strategy_name': 'OrderBlocksStrategy', 'total_signals': 0, 'win_rate': 0.0, 'profit_factor': 0.0, 'total_pnl': 0.0, 'grade_distribution': {'A': 0, 'B': 0, 'C': 0, 'D': 0}, 'last_signal': None}

4. Strategy Information:
   Name: Order Blocks Strategy
   Version: 2.0.0
   Type: Smart Money Concepts
   Timeframes: H4, H1, M15, M5
   Min Confidence: 0.70
   Performance:
     Success Rate: 0.00%
     Profit Factor: 0.00
   Parameters:
     - swing_length: 10
     - min_ob_strength: 2.0
     - fvg_min_size: 0.5
     - liquidity_sweep_tolerance: 1.2

============================================================
ORDER BLOCKS STRATEGY TEST COMPLETED!
============================================================

------------------------------------------------------------

--- Mode: LIVE ---
STDOUT OUTPUT:
============================================================
RUN MODE: LIVE - connecting to MT5 (production)
============================================================
✅ Connected to live MT5
============================================================
TESTING MODIFIED ORDER BLOCKS STRATEGY
============================================================
Running in LIVE mode

1. Testing signal generation:
   Generated 0 signals

2. Testing analysis method:
   Analysis results keys: dict_keys(['strategy', 'symbol', 'timeframe', 'analysis_time', 'data_points', 'current_market_structure', 'active_order_blocks_count', 'active_fair_value_gaps_count', 'recent_order_blocks', 'recent_fair_value_gaps'])
   Current Market Structure: RANGING
   Active Order Blocks: 0
   Active FVGs: 0

3. Testing performance tracking:
   {'strategy_name': 'OrderBlocksStrategy', 'total_signals': 0, 'win_rate': 0.0, 'profit_factor': 0.0, 'total_pnl': 0.0, 'grade_distribution': {'A': 0, 'B': 0, 'C': 0, 'D': 0}, 'last_signal': None}

4. Strategy Information:
   Name: Order Blocks Strategy
   Version: 2.0.0
   Type: Smart Money Concepts
   Timeframes: H4, H1, M15, M5
   Min Confidence: 0.70
   Performance:
     Success Rate: 0.00%
     Profit Factor: 0.00
   Parameters:
     - swing_length: 10
     - min_ob_strength: 2.0
     - fvg_min_size: 0.5
     - liquidity_sweep_tolerance: 1.2

============================================================
ORDER BLOCKS STRATEGY TEST COMPLETED!
============================================================

STDERR OUTPUT:
2025-08-23 23:05:04,266 - src.core.mt5_manager - INFO - Loaded 227 tradable symbols from CSV
2025-08-23 23:05:04,266 - src.core.mt5_manager - INFO - MT5Manager initialized for symbol: XAUUSD
2025-08-23 23:05:04,267 - src.core.mt5_manager - INFO - Magic number: 123456
2025-08-23 23:05:04,267 - src.core.mt5_manager - INFO - Initializing MT5 with default path
2025-08-23 23:05:04,280 - src.core.mt5_manager - INFO - Attempting to login to account 273949055 on server Exness-MT5Trial6
2025-08-23 23:05:04,280 - src.core.mt5_manager - WARNING - Symbol XAUUSD not found, trying alternative symbols
2025-08-23 23:05:04,281 - src.core.mt5_manager - INFO - Using alternative symbol: XAUUSDm
2025-08-23 23:05:04,281 - src.core.mt5_manager - INFO - ✅ Successfully connected to MT5
2025-08-23 23:05:04,281 - src.core.mt5_manager - INFO -    Account: 273949055
2025-08-23 23:05:04,281 - src.core.mt5_manager - INFO -    Balance: $194.55
2025-08-23 23:05:04,281 - src.core.mt5_manager - INFO -    Server: Exness-MT5Trial6
2025-08-23 23:05:04,281 - src.core.mt5_manager - INFO -    Symbol: XAUUSDm
2025-08-23 23:05:04,284 - src.core.mt5_manager - INFO - Retrieved 500 bars for XAUUSDm M15
2025-08-23 23:05:09,912 - OrderBlocksStrategy - INFO - Order Blocks generated 0 signals from 0 candidates
2025-08-23 23:05:09,914 - src.core.mt5_manager - INFO - Retrieved 200 bars for XAUUSDm M15

------------------------------------------------------------

================================================================================

### Strategy: src/strategies/technical/elliott_wave.py
### Strategy Name: elliott_wave

--- Mode: MOCK ---
STDOUT OUTPUT:
============================================================
RUN MODE: MOCK - using simulated OHLCV data
============================================================
============================================================
TESTING MODIFIED ELLIOTT WAVE STRATEGY
============================================================
Running in MOCK mode

1. Testing signal generation:
   Generated 4 signals
   - Signal 1:
     Type: SELL
     Confidence: 86.20%
     Grade: A
     Price: 74.86
     Stop Loss: 145.29
     Take Profit: 12.15
     Pattern: flat
     Degree: MINOR
     Current Wave: Wave C complete - Awaiting new impulse
   - Signal 2:
     Type: SELL
     Confidence: 80.42%
     Grade: B
     Price: 74.86
     Stop Loss: 101.53
     Take Profit: 41.37
     Pattern: N/A
     Degree: N/A
     Wave Count: 3
   - Signal 3:
     Type: BUY
     Confidence: 64.91%
     Grade: C
     Price: 74.86
     Stop Loss: 61.53
     Take Profit: 111.84
     Pattern: N/A
     Degree: N/A
     Wave Count: 3
   - Signal 4:
     Type: SELL
     Confidence: 60.34%
     Grade: C
     Price: 74.86
     Stop Loss: 140.29
     Take Profit: 25.64
     Pattern: N/A
     Degree: N/A
     Wave Count: 3

2. Testing analysis method:
   Analysis results keys: dict_keys(['strategy', 'symbol', 'timeframe', 'analysis_time', 'swings_detected', 'total_patterns_found', 'valid_patterns_count', 'detailed_patterns'])
   Detected patterns in analysis: 4
   First pattern type: corrective

3. Testing performance tracking:
   {'strategy_name': 'ElliottWaveStrategy', 'total_signals': 0, 'win_rate': 0.0, 'profit_factor': 0.0, 'total_pnl': 0.0, 'grade_distribution': {'A': 0, 'B': 0, 'C': 0, 'D': 0}, 'last_signal': None}

4. Strategy Information:
   Name: Elliott Wave Analysis Strategy
   Version: 2.0.0
   Description: Advanced Elliott Wave pattern recognition with Fibonacci validation
   Type: Technical
   Wave Types Supported: impulse, corrective, diagonal, triangle, flat, zigzag, complex
   Wave Degrees: GRAND_SUPERCYCLE, SUPERCYCLE, CYCLE, PRIMARY, INTERMEDIATE, MINOR, MINUTE, MINUETTE, SUBMINUETTE
   Parameters:
     - min_wave_size: 10
     - lookback_periods: 200
     - min_confidence: 0.6
     - fibonacci_tolerance: 0.1
   Performance Summary:
     Total Signals Generated: 0
     Successful Signals: 0
     Win Rate: 0.00%
     Profit Factor: 0.00

============================================================
ELLIOTT WAVE STRATEGY TEST COMPLETED!
============================================================

STDERR OUTPUT:
INFO:ElliottWaveStrategy:Elliott Wave Strategy initialized with min_wave_size=10, lookback=200
INFO:ElliottWaveStrategy:Found corrective corrective pattern with confidence 0.80
INFO:ElliottWaveStrategy:Found triangle corrective pattern with confidence 0.80
INFO:ElliottWaveStrategy:Found corrective corrective pattern with confidence 0.81
INFO:ElliottWaveStrategy:Found flat corrective pattern with confidence 0.86
INFO:ElliottWaveStrategy:Found corrective corrective pattern with confidence 0.80
INFO:ElliottWaveStrategy:Found triangle corrective pattern with confidence 0.80
INFO:ElliottWaveStrategy:Found corrective corrective pattern with confidence 0.81
INFO:ElliottWaveStrategy:Found flat corrective pattern with confidence 0.86
INFO:ElliottWaveStrategy:Signal confidence 0.58 below threshold 0.6
INFO:ElliottWaveStrategy:Signal confidence 0.58 below threshold 0.6
INFO:ElliottWaveStrategy:Found corrective corrective pattern with confidence 0.80
INFO:ElliottWaveStrategy:Found triangle corrective pattern with confidence 0.80
INFO:ElliottWaveStrategy:Found corrective corrective pattern with confidence 0.81
INFO:ElliottWaveStrategy:Found flat corrective pattern with confidence 0.86

------------------------------------------------------------

--- Mode: LIVE ---
STDOUT OUTPUT:
============================================================
RUN MODE: LIVE - connecting to MT5 (production)
============================================================
✅ Connected to live MT5
============================================================
TESTING MODIFIED ELLIOTT WAVE STRATEGY
============================================================
Running in LIVE mode

1. Testing signal generation:
   Generated 4 signals
   - Signal 1:
     Type: BUY
     Confidence: 83.20%
     Grade: B
     Price: 3372.30
     Stop Loss: 3306.01
     Take Profit: 3382.30
     Pattern: corrective
     Degree: MINOR
     Current Wave: Wave C complete - Awaiting new impulse
   - Signal 2:
     Type: BUY
     Confidence: 83.20%
     Grade: B
     Price: 3372.30
     Stop Loss: 3311.01
     Take Profit: 3380.15
     Pattern: N/A
     Degree: N/A
     Wave Count: 3
   - Signal 3:
     Type: SELL
     Confidence: 74.88%
     Grade: B
     Price: 3372.30
     Stop Loss: 3373.08
     Take Profit: 3343.15
     Pattern: N/A
     Degree: N/A
     Wave Count: 3
   - Signal 4:
     Type: BUY
     Confidence: 62.51%
     Grade: C
     Price: 3372.30
     Stop Loss: 3312.57
     Take Profit: 3385.14
     Pattern: N/A
     Degree: N/A
     Wave Count: 3

2. Testing analysis method:
   Analysis results keys: dict_keys(['strategy', 'symbol', 'timeframe', 'analysis_time', 'swings_detected', 'total_patterns_found', 'valid_patterns_count', 'detailed_patterns'])
   Detected patterns in analysis: 4
   First pattern type: corrective

3. Testing performance tracking:
   {'strategy_name': 'ElliottWaveStrategy', 'total_signals': 0, 'win_rate': 0.0, 'profit_factor': 0.0, 'total_pnl': 0.0, 'grade_distribution': {'A': 0, 'B': 0, 'C': 0, 'D': 0}, 'last_signal': None}

4. Strategy Information:
   Name: Elliott Wave Analysis Strategy
   Version: 2.0.0
   Description: Advanced Elliott Wave pattern recognition with Fibonacci validation
   Type: Technical
   Wave Types Supported: impulse, corrective, diagonal, triangle, flat, zigzag, complex
   Wave Degrees: GRAND_SUPERCYCLE, SUPERCYCLE, CYCLE, PRIMARY, INTERMEDIATE, MINOR, MINUTE, MINUETTE, SUBMINUETTE
   Parameters:
     - min_wave_size: 10
     - lookback_periods: 200
     - min_confidence: 0.6
     - fibonacci_tolerance: 0.1
   Performance Summary:
     Total Signals Generated: 0
     Successful Signals: 0
     Win Rate: 0.00%
     Profit Factor: 0.00

============================================================
ELLIOTT WAVE STRATEGY TEST COMPLETED!
============================================================

STDERR OUTPUT:
INFO:src.core.mt5_manager:Loaded 227 tradable symbols from CSV
INFO:src.core.mt5_manager:MT5Manager initialized for symbol: XAUUSD
INFO:src.core.mt5_manager:Magic number: 123456
INFO:src.core.mt5_manager:Initializing MT5 with default path
INFO:src.core.mt5_manager:Attempting to login to account 273949055 on server Exness-MT5Trial6
WARNING:src.core.mt5_manager:Symbol XAUUSD not found, trying alternative symbols
INFO:src.core.mt5_manager:Using alternative symbol: XAUUSDm
INFO:src.core.mt5_manager:✅ Successfully connected to MT5
INFO:src.core.mt5_manager:   Account: 273949055
INFO:src.core.mt5_manager:   Balance: $194.55
INFO:src.core.mt5_manager:   Server: Exness-MT5Trial6
INFO:src.core.mt5_manager:   Symbol: XAUUSDm
INFO:ElliottWaveStrategy:Elliott Wave Strategy initialized with min_wave_size=10, lookback=200
INFO:src.core.mt5_manager:Retrieved 200 bars for XAUUSDm M15
WARNING:ElliottWaveStrategy:Error checking volume confirmation: cannot do slice indexing on DatetimeIndex with these indexers [120] of type int64
INFO:ElliottWaveStrategy:Found bearish impulse pattern with confidence 0.82
INFO:ElliottWaveStrategy:Found triangle corrective pattern with confidence 0.84
INFO:ElliottWaveStrategy:Found corrective corrective pattern with confidence 0.84
INFO:ElliottWaveStrategy:Found corrective corrective pattern with confidence 0.84
INFO:ElliottWaveStrategy:Found corrective corrective pattern with confidence 0.81
INFO:ElliottWaveStrategy:Found triangle corrective pattern with confidence 0.81
INFO:ElliottWaveStrategy:Found corrective corrective pattern with confidence 0.77
INFO:ElliottWaveStrategy:Found corrective corrective pattern with confidence 0.77
INFO:ElliottWaveStrategy:Found corrective corrective pattern with confidence 0.77
INFO:ElliottWaveStrategy:Found corrective corrective pattern with confidence 0.83
INFO:ElliottWaveStrategy:Found zigzag corrective pattern with confidence 0.83
INFO:ElliottWaveStrategy:Found corrective corrective pattern with confidence 0.78
INFO:ElliottWaveStrategy:Found flat corrective pattern with confidence 0.83
INFO:ElliottWaveStrategy:Found flat corrective pattern with confidence 0.82
INFO:ElliottWaveStrategy:Found corrective corrective pattern with confidence 0.80
INFO:ElliottWaveStrategy:Found zigzag corrective pattern with confidence 0.85
INFO:ElliottWaveStrategy:Found corrective corrective pattern with confidence 0.85
INFO:ElliottWaveStrategy:Found corrective corrective pattern with confidence 0.77
INFO:ElliottWaveStrategy:Found corrective corrective pattern with confidence 0.80
INFO:ElliottWaveStrategy:Found triangle corrective pattern with confidence 0.80
INFO:ElliottWaveStrategy:Found corrective corrective pattern with confidence 0.83
INFO:ElliottWaveStrategy:Found zigzag corrective pattern with confidence 0.83
INFO:ElliottWaveStrategy:Found corrective corrective pattern with confidence 0.77
INFO:ElliottWaveStrategy:Found corrective corrective pattern with confidence 0.77
INFO:ElliottWaveStrategy:Found corrective corrective pattern with confidence 0.82
INFO:ElliottWaveStrategy:Found corrective corrective pattern with confidence 0.82
INFO:ElliottWaveStrategy:Found triangle corrective pattern with confidence 0.84
INFO:ElliottWaveStrategy:Found corrective corrective pattern with confidence 0.85
INFO:ElliottWaveStrategy:Found flat corrective pattern with confidence 0.86
WARNING:ElliottWaveStrategy:Error checking volume confirmation: cannot do slice indexing on DatetimeIndex with these indexers [120] of type int64
INFO:ElliottWaveStrategy:Found bearish impulse pattern with confidence 0.82
INFO:ElliottWaveStrategy:Found triangle corrective pattern with confidence 0.84
INFO:ElliottWaveStrategy:Found corrective corrective pattern with confidence 0.84
INFO:ElliottWaveStrategy:Found corrective corrective pattern with confidence 0.84
INFO:ElliottWaveStrategy:Found corrective corrective pattern with confidence 0.81
INFO:ElliottWaveStrategy:Found triangle corrective pattern with confidence 0.81
INFO:ElliottWaveStrategy:Found corrective corrective pattern with confidence 0.77
INFO:ElliottWaveStrategy:Found corrective corrective pattern with confidence 0.77
INFO:ElliottWaveStrategy:Found corrective corrective pattern with confidence 0.77
INFO:ElliottWaveStrategy:Found corrective corrective pattern with confidence 0.83
INFO:ElliottWaveStrategy:Found zigzag corrective pattern with confidence 0.83
INFO:ElliottWaveStrategy:Found corrective corrective pattern with confidence 0.78
INFO:ElliottWaveStrategy:Found flat corrective pattern with confidence 0.83
INFO:ElliottWaveStrategy:Found flat corrective pattern with confidence 0.82
INFO:ElliottWaveStrategy:Found corrective corrective pattern with confidence 0.80
INFO:ElliottWaveStrategy:Found zigzag corrective pattern with confidence 0.85
INFO:ElliottWaveStrategy:Found corrective corrective pattern with confidence 0.85
INFO:ElliottWaveStrategy:Found corrective corrective pattern with confidence 0.77
INFO:ElliottWaveStrategy:Found corrective corrective pattern with confidence 0.80
INFO:ElliottWaveStrategy:Found triangle corrective pattern with confidence 0.80
INFO:ElliottWaveStrategy:Found corrective corrective pattern with confidence 0.83
INFO:ElliottWaveStrategy:Found zigzag corrective pattern with confidence 0.83
INFO:ElliottWaveStrategy:Found corrective corrective pattern with confidence 0.77
INFO:ElliottWaveStrategy:Found corrective corrective pattern with confidence 0.77
INFO:ElliottWaveStrategy:Found corrective corrective pattern with confidence 0.82
INFO:ElliottWaveStrategy:Found corrective corrective pattern with confidence 0.82
INFO:ElliottWaveStrategy:Found triangle corrective pattern with confidence 0.84
INFO:ElliottWaveStrategy:Found corrective corrective pattern with confidence 0.85
INFO:ElliottWaveStrategy:Found flat corrective pattern with confidence 0.86
INFO:ElliottWaveStrategy:Signal confidence 0.5731321409289594 below threshold 0.6
INFO:ElliottWaveStrategy:Signal confidence 0.58 below threshold 0.6
INFO:ElliottWaveStrategy:Signal confidence 0.58 below threshold 0.6
INFO:ElliottWaveStrategy:Signal confidence 0.58 below threshold 0.6
INFO:ElliottWaveStrategy:Signal confidence 0.58 below threshold 0.6
INFO:src.core.mt5_manager:Retrieved 200 bars for XAUUSDm M15
WARNING:ElliottWaveStrategy:Error checking volume confirmation: cannot do slice indexing on DatetimeIndex with these indexers [120] of type int64
INFO:ElliottWaveStrategy:Found bearish impulse pattern with confidence 0.82
INFO:ElliottWaveStrategy:Found triangle corrective pattern with confidence 0.84
INFO:ElliottWaveStrategy:Found corrective corrective pattern with confidence 0.84
INFO:ElliottWaveStrategy:Found corrective corrective pattern with confidence 0.84
INFO:ElliottWaveStrategy:Found corrective corrective pattern with confidence 0.81
INFO:ElliottWaveStrategy:Found triangle corrective pattern with confidence 0.81
INFO:ElliottWaveStrategy:Found corrective corrective pattern with confidence 0.77
INFO:ElliottWaveStrategy:Found corrective corrective pattern with confidence 0.77
INFO:ElliottWaveStrategy:Found corrective corrective pattern with confidence 0.77
INFO:ElliottWaveStrategy:Found corrective corrective pattern with confidence 0.83
INFO:ElliottWaveStrategy:Found zigzag corrective pattern with confidence 0.83
INFO:ElliottWaveStrategy:Found corrective corrective pattern with confidence 0.78
INFO:ElliottWaveStrategy:Found flat corrective pattern with confidence 0.83
INFO:ElliottWaveStrategy:Found flat corrective pattern with confidence 0.82
INFO:ElliottWaveStrategy:Found corrective corrective pattern with confidence 0.80
INFO:ElliottWaveStrategy:Found zigzag corrective pattern with confidence 0.85
INFO:ElliottWaveStrategy:Found corrective corrective pattern with confidence 0.85
INFO:ElliottWaveStrategy:Found corrective corrective pattern with confidence 0.77
INFO:ElliottWaveStrategy:Found corrective corrective pattern with confidence 0.80
INFO:ElliottWaveStrategy:Found triangle corrective pattern with confidence 0.80
INFO:ElliottWaveStrategy:Found corrective corrective pattern with confidence 0.83
INFO:ElliottWaveStrategy:Found zigzag corrective pattern with confidence 0.83
INFO:ElliottWaveStrategy:Found corrective corrective pattern with confidence 0.77
INFO:ElliottWaveStrategy:Found corrective corrective pattern with confidence 0.77
INFO:ElliottWaveStrategy:Found corrective corrective pattern with confidence 0.82
INFO:ElliottWaveStrategy:Found corrective corrective pattern with confidence 0.82
INFO:ElliottWaveStrategy:Found triangle corrective pattern with confidence 0.84
INFO:ElliottWaveStrategy:Found corrective corrective pattern with confidence 0.85
INFO:ElliottWaveStrategy:Found flat corrective pattern with confidence 0.86

------------------------------------------------------------

================================================================================

### Strategy: src/strategies/technical/fibonacci_advanced.py
### Strategy Name: fibonacci_advanced

--- Mode: MOCK ---
STDOUT OUTPUT:
============================================================
RUN MODE: MOCK - using simulated OHLCV data
============================================================
============================================================
TESTING FIBONACCI ADVANCED STRATEGY
============================================================
Running in MOCK mode

1. Testing signal generation (Multiple runs to simulate daily signals):
   Run 1: Generated 3 signals
     - BUY at 1956.39, Confidence: 0.950, Grade: A
       Reason: support_bounce
       SL: 1951.16, TP: 1963.37
       Risk/Reward: 1.33
     - SELL at 1956.39, Confidence: 0.950, Grade: A
       Reason: resistance_rejection
       SL: 1961.63, TP: 1949.41
       Risk/Reward: 1.33
     - BUY at 1956.39, Confidence: 0.950, Grade: A
       Reason: support_bounce
       SL: 1951.16, TP: 1963.37
       Risk/Reward: 1.33
   Run 2: Generated 3 signals
     - BUY at 1956.39, Confidence: 0.950, Grade: A
       Reason: support_bounce
       SL: 1951.16, TP: 1963.37
       Risk/Reward: 1.33
     - SELL at 1956.39, Confidence: 0.950, Grade: A
       Reason: resistance_rejection
       SL: 1961.63, TP: 1949.41
       Risk/Reward: 1.33
     - BUY at 1956.39, Confidence: 0.950, Grade: A
       Reason: support_bounce
       SL: 1951.16, TP: 1963.37
       Risk/Reward: 1.33
   Run 3: Generated 3 signals
     - BUY at 1956.39, Confidence: 0.950, Grade: A
       Reason: support_bounce
       SL: 1951.16, TP: 1963.37
       Risk/Reward: 1.33
     - SELL at 1956.39, Confidence: 0.950, Grade: A
       Reason: resistance_rejection
       SL: 1961.63, TP: 1949.41
       Risk/Reward: 1.33
     - BUY at 1956.39, Confidence: 0.950, Grade: A
       Reason: support_bounce
       SL: 1951.16, TP: 1963.37
       Risk/Reward: 1.33
   Run 4: Generated 3 signals
     - BUY at 1956.39, Confidence: 0.950, Grade: A
       Reason: support_bounce
       SL: 1951.16, TP: 1963.37
       Risk/Reward: 1.33
     - SELL at 1956.39, Confidence: 0.950, Grade: A
       Reason: resistance_rejection
       SL: 1961.63, TP: 1949.41
       Risk/Reward: 1.33
     - BUY at 1956.39, Confidence: 0.950, Grade: A
       Reason: support_bounce
       SL: 1951.16, TP: 1963.37
       Risk/Reward: 1.33
   Run 5: Generated 3 signals
     - BUY at 1956.39, Confidence: 0.950, Grade: A
       Reason: support_bounce
       SL: 1951.16, TP: 1963.37
       Risk/Reward: 1.33
     - SELL at 1956.39, Confidence: 0.950, Grade: A
       Reason: resistance_rejection
       SL: 1961.63, TP: 1949.41
       Risk/Reward: 1.33
     - BUY at 1956.39, Confidence: 0.950, Grade: A
       Reason: support_bounce
       SL: 1951.16, TP: 1963.37
       Risk/Reward: 1.33
   Run 6: Generated 3 signals
     - BUY at 1956.39, Confidence: 0.950, Grade: A
       Reason: support_bounce
       SL: 1951.16, TP: 1963.37
       Risk/Reward: 1.33
     - SELL at 1956.39, Confidence: 0.950, Grade: A
       Reason: resistance_rejection
       SL: 1961.63, TP: 1949.41
       Risk/Reward: 1.33
     - BUY at 1956.39, Confidence: 0.950, Grade: A
       Reason: support_bounce
       SL: 1951.16, TP: 1963.37
       Risk/Reward: 1.33
   Run 7: Generated 3 signals
     - BUY at 1956.39, Confidence: 0.950, Grade: A
       Reason: support_bounce
       SL: 1951.16, TP: 1963.37
       Risk/Reward: 1.33
     - SELL at 1956.39, Confidence: 0.950, Grade: A
       Reason: resistance_rejection
       SL: 1961.63, TP: 1949.41
       Risk/Reward: 1.33
     - BUY at 1956.39, Confidence: 0.950, Grade: A
       Reason: support_bounce
       SL: 1951.16, TP: 1963.37
       Risk/Reward: 1.33
   Run 8: Generated 3 signals
     - BUY at 1956.39, Confidence: 0.950, Grade: A
       Reason: support_bounce
       SL: 1951.16, TP: 1963.37
       Risk/Reward: 1.33
     - SELL at 1956.39, Confidence: 0.950, Grade: A
       Reason: resistance_rejection
       SL: 1961.63, TP: 1949.41
       Risk/Reward: 1.33
     - BUY at 1956.39, Confidence: 0.950, Grade: A
       Reason: support_bounce
       SL: 1951.16, TP: 1963.37
       Risk/Reward: 1.33

   TOTAL DAILY SIGNALS: 24
   Signal Distribution:
     BUY signals: 16
     SELL signals: 8
     Average confidence: 0.950
     Grade distribution: {'A': 24}

2. Testing analysis method:
   Analysis results keys: ['recent_swings', 'retracement_levels', 'extension_levels', 'clusters', 'current_price', 'trend_direction']
   Recent swings: 5
   Retracement levels: 260
   Extension levels: 156
   Clusters: 5
   Trend direction: sideways

3. Testing performance tracking:
   {'strategy_name': 'FibonacciAdvancedStrategy', 'total_signals': 0, 'win_rate': 0.0, 'profit_factor': 0.0, 'total_pnl': 0.0, 'grade_distribution': {'A': 0, 'B': 0, 'C': 0, 'D': 0}, 'last_signal': None}

4. Strategy Information:
   Name: FibonacciAdvanced
   Type: Technical
   Description: Advanced Fibonacci strategy using retracements, extensions, and clusters
   Parameters:
     lookback_period: 200
     fib_levels: [0.236, 0.382, 0.5, 0.618, 0.786, 1.272, 1.618, 2.0]
     cluster_tolerance: 0.003
     confidence_threshold: 0.65
     multi_timeframe: True
     cooldown_bars: 3
   Performance Summary:
     Success Rate: 0.00%
     Profit Factor: 0.00

============================================================
FIBONACCI ADVANCED STRATEGY TEST COMPLETED!
============================================================

------------------------------------------------------------

--- Mode: LIVE ---
STDOUT OUTPUT:
============================================================
RUN MODE: LIVE - connecting to MT5 (production)
============================================================
✅ Connected to live MT5
============================================================
TESTING FIBONACCI ADVANCED STRATEGY
============================================================
Running in LIVE mode

1. Testing signal generation (Multiple runs to simulate daily signals):
   Run 1: Generated 3 signals
     - BUY at 3372.30, Confidence: 0.950, Grade: A
       Reason: bullish_retracement
       SL: 3370.01, TP: 3375.35
       Risk/Reward: 1.33
     - BUY at 3372.30, Confidence: 0.950, Grade: A
       Reason: bullish_retracement
       SL: 3370.01, TP: 3375.35
       Risk/Reward: 1.33
     - BUY at 3372.30, Confidence: 0.891, Grade: A
       Reason: approaching_resistance
       SL: 3370.01, TP: 3375.35
       Risk/Reward: 1.33
   Run 2: Generated 3 signals
     - BUY at 3372.30, Confidence: 0.950, Grade: A
       Reason: bullish_retracement
       SL: 3370.01, TP: 3375.35
       Risk/Reward: 1.33
     - BUY at 3372.30, Confidence: 0.950, Grade: A
       Reason: bullish_retracement
       SL: 3370.01, TP: 3375.35
       Risk/Reward: 1.33
     - BUY at 3372.30, Confidence: 0.891, Grade: A
       Reason: approaching_resistance
       SL: 3370.01, TP: 3375.35
       Risk/Reward: 1.33
   Run 3: Generated 3 signals
     - BUY at 3372.30, Confidence: 0.950, Grade: A
       Reason: bullish_retracement
       SL: 3370.01, TP: 3375.35
       Risk/Reward: 1.33
     - BUY at 3372.30, Confidence: 0.950, Grade: A
       Reason: bullish_retracement
       SL: 3370.01, TP: 3375.35
       Risk/Reward: 1.33
     - BUY at 3372.30, Confidence: 0.891, Grade: A
       Reason: approaching_resistance
       SL: 3370.01, TP: 3375.35
       Risk/Reward: 1.33
   Run 4: Generated 3 signals
     - BUY at 3372.30, Confidence: 0.950, Grade: A
       Reason: bullish_retracement
       SL: 3370.01, TP: 3375.35
       Risk/Reward: 1.33
     - BUY at 3372.30, Confidence: 0.950, Grade: A
       Reason: bullish_retracement
       SL: 3370.01, TP: 3375.35
       Risk/Reward: 1.33
     - BUY at 3372.30, Confidence: 0.891, Grade: A
       Reason: approaching_resistance
       SL: 3370.01, TP: 3375.35
       Risk/Reward: 1.33
   Run 5: Generated 3 signals
     - BUY at 3372.30, Confidence: 0.950, Grade: A
       Reason: bullish_retracement
       SL: 3370.01, TP: 3375.35
       Risk/Reward: 1.33
     - BUY at 3372.30, Confidence: 0.950, Grade: A
       Reason: bullish_retracement
       SL: 3370.01, TP: 3375.35
       Risk/Reward: 1.33
     - BUY at 3372.30, Confidence: 0.891, Grade: A
       Reason: approaching_resistance
       SL: 3370.01, TP: 3375.35
       Risk/Reward: 1.33
   Run 6: Generated 3 signals
     - BUY at 3372.30, Confidence: 0.950, Grade: A
       Reason: bullish_retracement
       SL: 3370.01, TP: 3375.35
       Risk/Reward: 1.33
     - BUY at 3372.30, Confidence: 0.950, Grade: A
       Reason: bullish_retracement
       SL: 3370.01, TP: 3375.35
       Risk/Reward: 1.33
     - BUY at 3372.30, Confidence: 0.891, Grade: A
       Reason: approaching_resistance
       SL: 3370.01, TP: 3375.35
       Risk/Reward: 1.33
   Run 7: Generated 3 signals
     - BUY at 3372.30, Confidence: 0.950, Grade: A
       Reason: bullish_retracement
       SL: 3370.01, TP: 3375.35
       Risk/Reward: 1.33
     - BUY at 3372.30, Confidence: 0.950, Grade: A
       Reason: bullish_retracement
       SL: 3370.01, TP: 3375.35
       Risk/Reward: 1.33
     - BUY at 3372.30, Confidence: 0.891, Grade: A
       Reason: approaching_resistance
       SL: 3370.01, TP: 3375.35
       Risk/Reward: 1.33
   Run 8: Generated 3 signals
     - BUY at 3372.30, Confidence: 0.950, Grade: A
       Reason: bullish_retracement
       SL: 3370.01, TP: 3375.35
       Risk/Reward: 1.33
     - BUY at 3372.30, Confidence: 0.950, Grade: A
       Reason: bullish_retracement
       SL: 3370.01, TP: 3375.35
       Risk/Reward: 1.33
     - BUY at 3372.30, Confidence: 0.891, Grade: A
       Reason: approaching_resistance
       SL: 3370.01, TP: 3375.35
       Risk/Reward: 1.33

   TOTAL DAILY SIGNALS: 24
   Signal Distribution:
     BUY signals: 24
     SELL signals: 0
     Average confidence: 0.930
     Grade distribution: {'A': 24}

2. Testing analysis method:
   Analysis results keys: ['recent_swings', 'retracement_levels', 'extension_levels', 'clusters', 'current_price', 'trend_direction']
   Recent swings: 5
   Retracement levels: 200
   Extension levels: 120
   Clusters: 5
   Trend direction: up

3. Testing performance tracking:
   {'strategy_name': 'FibonacciAdvancedStrategy', 'total_signals': 0, 'win_rate': 0.0, 'profit_factor': 0.0, 'total_pnl': 0.0, 'grade_distribution': {'A': 0, 'B': 0, 'C': 0, 'D': 0}, 'last_signal': None}

4. Strategy Information:
   Name: FibonacciAdvanced
   Type: Technical
   Description: Advanced Fibonacci strategy using retracements, extensions, and clusters
   Parameters:
     lookback_period: 200
     fib_levels: [0.236, 0.382, 0.5, 0.618, 0.786, 1.272, 1.618, 2.0]
     cluster_tolerance: 0.003
     confidence_threshold: 0.65
     multi_timeframe: True
     cooldown_bars: 3
   Performance Summary:
     Success Rate: 0.00%
     Profit Factor: 0.00

============================================================
FIBONACCI ADVANCED STRATEGY TEST COMPLETED!
============================================================

STDERR OUTPUT:
2025-08-23 23:05:16,894 - src.core.mt5_manager - INFO - Loaded 227 tradable symbols from CSV
2025-08-23 23:05:16,894 - src.core.mt5_manager - INFO - MT5Manager initialized for symbol: XAUUSD
2025-08-23 23:05:16,894 - src.core.mt5_manager - INFO - Magic number: 123456
2025-08-23 23:05:16,894 - src.core.mt5_manager - INFO - Initializing MT5 with default path
2025-08-23 23:05:16,908 - src.core.mt5_manager - INFO - Attempting to login to account 273949055 on server Exness-MT5Trial6
2025-08-23 23:05:16,909 - src.core.mt5_manager - WARNING - Symbol XAUUSD not found, trying alternative symbols
2025-08-23 23:05:16,909 - src.core.mt5_manager - INFO - Using alternative symbol: XAUUSDm
2025-08-23 23:05:16,909 - src.core.mt5_manager - INFO - ✅ Successfully connected to MT5
2025-08-23 23:05:16,909 - src.core.mt5_manager - INFO -    Account: 273949055
2025-08-23 23:05:16,910 - src.core.mt5_manager - INFO -    Balance: $194.55
2025-08-23 23:05:16,910 - src.core.mt5_manager - INFO -    Server: Exness-MT5Trial6
2025-08-23 23:05:16,910 - src.core.mt5_manager - INFO -    Symbol: XAUUSDm
2025-08-23 23:05:16,910 - FibonacciAdvancedStrategy - INFO - Initialized FibonacciAdvancedStrategy with lookback=200, cluster_tolerance=0.3%
2025-08-23 23:05:16,914 - src.core.mt5_manager - INFO - Retrieved 200 bars for XAUUSDm M15
2025-08-23 23:05:16,918 - src.core.mt5_manager - INFO - Retrieved 200 bars for XAUUSDm M15
2025-08-23 23:05:16,924 - src.core.mt5_manager - INFO - Retrieved 200 bars for XAUUSDm H1
2025-08-23 23:05:16,959 - src.core.mt5_manager - INFO - Retrieved 200 bars for XAUUSDm M15
2025-08-23 23:05:16,965 - src.core.mt5_manager - INFO - Retrieved 200 bars for XAUUSDm M15
2025-08-23 23:05:16,969 - src.core.mt5_manager - INFO - Retrieved 200 bars for XAUUSDm H1
2025-08-23 23:05:16,995 - src.core.mt5_manager - INFO - Retrieved 200 bars for XAUUSDm M15
2025-08-23 23:05:16,999 - src.core.mt5_manager - INFO - Retrieved 200 bars for XAUUSDm M15
2025-08-23 23:05:17,003 - src.core.mt5_manager - INFO - Retrieved 200 bars for XAUUSDm H1
2025-08-23 23:05:17,021 - src.core.mt5_manager - INFO - Retrieved 200 bars for XAUUSDm M15
2025-08-23 23:05:17,024 - src.core.mt5_manager - INFO - Retrieved 200 bars for XAUUSDm M15
2025-08-23 23:05:17,028 - src.core.mt5_manager - INFO - Retrieved 200 bars for XAUUSDm H1
2025-08-23 23:05:17,044 - src.core.mt5_manager - INFO - Retrieved 200 bars for XAUUSDm M15
2025-08-23 23:05:17,048 - src.core.mt5_manager - INFO - Retrieved 200 bars for XAUUSDm M15
2025-08-23 23:05:17,051 - src.core.mt5_manager - INFO - Retrieved 200 bars for XAUUSDm H1
2025-08-23 23:05:17,069 - src.core.mt5_manager - INFO - Retrieved 200 bars for XAUUSDm M15
2025-08-23 23:05:17,072 - src.core.mt5_manager - INFO - Retrieved 200 bars for XAUUSDm M15
2025-08-23 23:05:17,075 - src.core.mt5_manager - INFO - Retrieved 200 bars for XAUUSDm H1
2025-08-23 23:05:17,096 - src.core.mt5_manager - INFO - Retrieved 200 bars for XAUUSDm M15
2025-08-23 23:05:17,101 - src.core.mt5_manager - INFO - Retrieved 200 bars for XAUUSDm M15
2025-08-23 23:05:17,105 - src.core.mt5_manager - INFO - Retrieved 200 bars for XAUUSDm H1
2025-08-23 23:05:17,125 - src.core.mt5_manager - INFO - Retrieved 200 bars for XAUUSDm M15
2025-08-23 23:05:17,129 - src.core.mt5_manager - INFO - Retrieved 200 bars for XAUUSDm M15
2025-08-23 23:05:17,133 - src.core.mt5_manager - INFO - Retrieved 200 bars for XAUUSDm H1
2025-08-23 23:05:17,156 - src.core.mt5_manager - INFO - Retrieved 200 bars for XAUUSDm M15

------------------------------------------------------------

================================================================================

### Strategy: src/strategies/technical/gann.py
### Strategy Name: gann

--- Mode: MOCK ---
STDOUT OUTPUT:
============================================================
RUN MODE: MOCK - using simulated OHLCV data
============================================================
============================================================
TESTING GANN STRATEGY
============================================================
Running in MOCK mode

1. Testing signal generation:
   Generated 1 signals
   - SELL at 1960.15, Confidence: 0.850, Grade: B
     Metadata: {'level_type': 'square_of_nine', 'price_level': np.float64(1960.15), 'original_timestamp': datetime.datetime(2025, 8, 23, 23, 5, 18, 77265)}

2. Testing analysis method:
   Analysis results keys: ['recent_swing_high', 'recent_swing_low', 'active_gann_angles', 'gann_price_levels', 'nearest_angle_touch', 'nearest_price_level']
   Swing High: 1953.30
   Swing Low: 1948.38
   Active Angles: 6
   Price Levels: 5

3. Testing performance tracking:
   {'strategy_name': 'GannStrategy', 'total_signals': 0, 'win_rate': 0.0, 'profit_factor': 0.0, 'total_pnl': 0.0, 'grade_distribution': {'A': 0, 'B': 0, 'C': 0, 'D': 0}, 'last_signal': None}

4. Strategy Information:
   Name: Gann Strategy
   Type: Technical
   Description: Uses simplified Gann techniques (1x1, 1x2, 2x1 angles and Square of Nine) to identify turning points in price and time.
   Parameters:
     lookback_period: 150
     gann_angles: [1, 2, 4]
     price_step: 1.0
     time_step: 1
     confidence_threshold: 0.65
     level_tolerance: 0.003
     cooldown_bars: 3
   Performance Summary:
     Success Rate: 0.00%
     Profit Factor: 0.00

============================================================
GANN STRATEGY TEST COMPLETED!
============================================================

------------------------------------------------------------

--- Mode: LIVE ---
STDOUT OUTPUT:
============================================================
RUN MODE: LIVE - connecting to MT5 (production)
============================================================
✅ Connected to live MT5
============================================================
TESTING GANN STRATEGY
============================================================
Running in LIVE mode

1. Testing signal generation:
   Generated 1 signals
   - SELL at 3372.30, Confidence: 0.850, Grade: B
     Metadata: {'level_type': 'square_of_nine', 'price_level': np.float64(3372.3), 'original_timestamp': datetime.datetime(2025, 8, 23, 23, 5, 18, 896113)}

2. Testing analysis method:
   Analysis results keys: ['recent_swing_high', 'recent_swing_low', 'active_gann_angles', 'gann_price_levels', 'nearest_angle_touch', 'nearest_price_level']
   Swing High: 3374.17
   Swing Low: 3328.59
   Active Angles: 6
   Price Levels: 5

3. Testing performance tracking:
   {'strategy_name': 'GannStrategy', 'total_signals': 0, 'win_rate': 0.0, 'profit_factor': 0.0, 'total_pnl': 0.0, 'grade_distribution': {'A': 0, 'B': 0, 'C': 0, 'D': 0}, 'last_signal': None}

4. Strategy Information:
   Name: Gann Strategy
   Type: Technical
   Description: Uses simplified Gann techniques (1x1, 1x2, 2x1 angles and Square of Nine) to identify turning points in price and time.
   Parameters:
     lookback_period: 150
     gann_angles: [1, 2, 4]
     price_step: 1.0
     time_step: 1
     confidence_threshold: 0.65
     level_tolerance: 0.003
     cooldown_bars: 3
   Performance Summary:
     Success Rate: 0.00%
     Profit Factor: 0.00

============================================================
GANN STRATEGY TEST COMPLETED!
============================================================

STDERR OUTPUT:
2025-08-23 23:05:18,868 - src.core.mt5_manager - INFO - Loaded 227 tradable symbols from CSV
2025-08-23 23:05:18,868 - src.core.mt5_manager - INFO - MT5Manager initialized for symbol: XAUUSD
2025-08-23 23:05:18,868 - src.core.mt5_manager - INFO - Magic number: 123456
2025-08-23 23:05:18,868 - src.core.mt5_manager - INFO - Initializing MT5 with default path
2025-08-23 23:05:18,885 - src.core.mt5_manager - INFO - Attempting to login to account 273949055 on server Exness-MT5Trial6
2025-08-23 23:05:18,886 - src.core.mt5_manager - WARNING - Symbol XAUUSD not found, trying alternative symbols
2025-08-23 23:05:18,886 - src.core.mt5_manager - INFO - Using alternative symbol: XAUUSDm
2025-08-23 23:05:18,886 - src.core.mt5_manager - INFO - ✅ Successfully connected to MT5
2025-08-23 23:05:18,886 - src.core.mt5_manager - INFO -    Account: 273949055
2025-08-23 23:05:18,886 - src.core.mt5_manager - INFO -    Balance: $194.55
2025-08-23 23:05:18,887 - src.core.mt5_manager - INFO -    Server: Exness-MT5Trial6
2025-08-23 23:05:18,887 - src.core.mt5_manager - INFO -    Symbol: XAUUSDm
2025-08-23 23:05:18,891 - src.core.mt5_manager - INFO - Retrieved 150 bars for XAUUSDm M15
2025-08-23 23:05:18,899 - src.core.mt5_manager - INFO - Retrieved 150 bars for XAUUSDm M15

------------------------------------------------------------

================================================================================

### Strategy: src/strategies/technical/harmonic.py
### Strategy Name: harmonic

--- Mode: MOCK ---
STDOUT OUTPUT:
============================================================
RUN MODE: MOCK - using simulated OHLCV data
============================================================
============================================================
TESTING MODIFIED HARMONIC STRATEGY
============================================================
Running in MOCK mode

1. Testing signal generation:
   Generated 1 signals
   - BUY at 1971.12, Confidence: 0.730, Grade: B
     Pattern: ABCD
     Pattern Score: 0.9

2. Testing analysis method:
   Analysis results keys: dict_keys(['strategy', 'symbol', 'timeframe', 'analysis_time', 'data_points', 'pivots_detected', 'harmonic_patterns_detected_count', 'patterns'])
   Detected patterns in analysis: 0

3. Testing performance tracking:
   {'strategy_name': 'HarmonicStrategy', 'total_signals': 0, 'win_rate': 0.0, 'profit_factor': 0.0, 'total_pnl': 0.0, 'grade_distribution': {'A': 0, 'B': 0, 'C': 0, 'D': 0}, 'last_signal': None}

4. Strategy Information:
   Name: Harmonic Patterns Strategy
   Version: 2.0.0
   Description: Advanced harmonic pattern recognition with Fibonacci validation
   Type: Technical
   Patterns Supported: GARTLEY, BUTTERFLY, BAT, CRAB, CYPHER, ABCD, THREE_DRIVES
   Minimum Confidence: 0.72
   Fibonacci Tolerance: 0.10
   Minimum Pattern Score: 0.50
   Detected Patterns Count (Last Run): 4
   Performance Summary:
     Success Rate: 0.00%
     Profit Factor: 0.00

============================================================
HARMONIC STRATEGY TEST COMPLETED IN MOCK MODE!
============================================================

------------------------------------------------------------

--- Mode: LIVE ---
STDOUT OUTPUT:
============================================================
RUN MODE: LIVE - connecting to MT5 (production)
============================================================
✅ Connected to live MT5
============================================================
TESTING MODIFIED HARMONIC STRATEGY
============================================================
Running in LIVE mode

1. Testing signal generation:
   Generated 1 signals
   - SELL at 3372.30, Confidence: 0.730, Grade: B
     Pattern: ABCD
     Pattern Score: 0.9

2. Testing analysis method:
   Analysis results keys: dict_keys(['strategy', 'symbol', 'timeframe', 'analysis_time', 'data_points', 'pivots_detected', 'harmonic_patterns_detected_count', 'patterns'])
   Detected patterns in analysis: 0

3. Testing performance tracking:
   {'strategy_name': 'HarmonicStrategy', 'total_signals': 0, 'win_rate': 0.0, 'profit_factor': 0.0, 'total_pnl': 0.0, 'grade_distribution': {'A': 0, 'B': 0, 'C': 0, 'D': 0}, 'last_signal': None}

4. Strategy Information:
   Name: Harmonic Patterns Strategy
   Version: 2.0.0
   Description: Advanced harmonic pattern recognition with Fibonacci validation
   Type: Technical
   Patterns Supported: GARTLEY, BUTTERFLY, BAT, CRAB, CYPHER, ABCD, THREE_DRIVES
   Minimum Confidence: 0.72
   Fibonacci Tolerance: 0.10
   Minimum Pattern Score: 0.50
   Detected Patterns Count (Last Run): 9
   Performance Summary:
     Success Rate: 0.00%
     Profit Factor: 0.00

============================================================
HARMONIC STRATEGY TEST COMPLETED IN LIVE MODE!
============================================================

STDERR OUTPUT:
2025-08-23 23:05:20,804 - src.core.mt5_manager - INFO - Loaded 227 tradable symbols from CSV
2025-08-23 23:05:20,804 - src.core.mt5_manager - INFO - MT5Manager initialized for symbol: XAUUSD
2025-08-23 23:05:20,804 - src.core.mt5_manager - INFO - Magic number: 123456
2025-08-23 23:05:20,805 - src.core.mt5_manager - INFO - Initializing MT5 with default path
2025-08-23 23:05:20,817 - src.core.mt5_manager - INFO - Attempting to login to account 273949055 on server Exness-MT5Trial6
2025-08-23 23:05:20,818 - src.core.mt5_manager - WARNING - Symbol XAUUSD not found, trying alternative symbols
2025-08-23 23:05:20,818 - src.core.mt5_manager - INFO - Using alternative symbol: XAUUSDm
2025-08-23 23:05:20,818 - src.core.mt5_manager - INFO - ✅ Successfully connected to MT5
2025-08-23 23:05:20,818 - src.core.mt5_manager - INFO -    Account: 273949055
2025-08-23 23:05:20,818 - src.core.mt5_manager - INFO -    Balance: $194.55
2025-08-23 23:05:20,818 - src.core.mt5_manager - INFO -    Server: Exness-MT5Trial6
2025-08-23 23:05:20,818 - src.core.mt5_manager - INFO -    Symbol: XAUUSDm
2025-08-23 23:05:20,822 - src.core.mt5_manager - INFO - Retrieved 200 bars for XAUUSDm M15
2025-08-23 23:05:20,836 - HarmonicStrategy - INFO - Harmonic generated 1 signals from 9 patterns
2025-08-23 23:05:20,839 - src.core.mt5_manager - INFO - Retrieved 200 bars for XAUUSDm M15

------------------------------------------------------------

================================================================================

### Strategy: src/strategies/technical/ichimoku.py
### Strategy Name: ichimoku

--- Mode: MOCK ---
STDOUT OUTPUT:
============================================================
RUN MODE: MOCK - using simulated OHLCV data
============================================================
============================================================
TESTING MODIFIED ICHIMOKU STRATEGY
============================================================
Running in MOCK mode

1. Testing signal generation:
   Generated 2 signals
   - BUY at 1933.69, Confidence: 0.70
   - SELL at 1933.69, Confidence: 0.65

2. Testing analysis method:
   Analysis keys: ['strategy', 'symbol', 'timeframe', 'analysis_time', 'data_points', 'current_values', 'market_position', 'signals', 'support_resistance', 'trend_strength', 'recommendation']
   Current Tenkan: 1936.25
   Current Kijun: 1938.34

3. Testing performance tracking:
   {'strategy_name': 'IchimokuStrategy', 'total_signals': 0, 'win_rate': 0.0, 'profit_factor': 0.0, 'total_pnl': 0.0, 'grade_distribution': {'A': 0, 'B': 0, 'C': 0, 'D': 0}, 'last_signal': None}

============================================================
ICHIMOKU STRATEGY TEST COMPLETED IN MOCK MODE!
============================================================

------------------------------------------------------------

--- Mode: LIVE ---
STDOUT OUTPUT:
============================================================
RUN MODE: LIVE - connecting to MT5 (production)
============================================================
✅ Connected to live MT5
============================================================
TESTING MODIFIED ICHIMOKU STRATEGY
============================================================
Running in LIVE mode

1. Testing signal generation:
   Generated 2 signals
   - BUY at 3372.30, Confidence: 0.70
   - SELL at 3372.30, Confidence: 0.65

2. Testing analysis method:
   Analysis keys: ['strategy', 'symbol', 'timeframe', 'analysis_time', 'data_points', 'current_values', 'market_position', 'signals', 'support_resistance', 'trend_strength', 'recommendation']
   Current Tenkan: 3371.37
   Current Kijun: 3372.45

3. Testing performance tracking:
   {'strategy_name': 'IchimokuStrategy', 'total_signals': 0, 'win_rate': 0.0, 'profit_factor': 0.0, 'total_pnl': 0.0, 'grade_distribution': {'A': 0, 'B': 0, 'C': 0, 'D': 0}, 'last_signal': None}

============================================================
ICHIMOKU STRATEGY TEST COMPLETED IN LIVE MODE!
============================================================

STDERR OUTPUT:
2025-08-23 23:05:22,762 - src.core.mt5_manager - INFO - Loaded 227 tradable symbols from CSV
2025-08-23 23:05:22,762 - src.core.mt5_manager - INFO - MT5Manager initialized for symbol: XAUUSD
2025-08-23 23:05:22,762 - src.core.mt5_manager - INFO - Magic number: 123456
2025-08-23 23:05:22,762 - src.core.mt5_manager - INFO - Initializing MT5 with default path
2025-08-23 23:05:22,775 - src.core.mt5_manager - INFO - Attempting to login to account 273949055 on server Exness-MT5Trial6
2025-08-23 23:05:22,775 - src.core.mt5_manager - WARNING - Symbol XAUUSD not found, trying alternative symbols
2025-08-23 23:05:22,775 - src.core.mt5_manager - INFO - Using alternative symbol: XAUUSDm
2025-08-23 23:05:22,776 - src.core.mt5_manager - INFO - ✅ Successfully connected to MT5
2025-08-23 23:05:22,776 - src.core.mt5_manager - INFO -    Account: 273949055
2025-08-23 23:05:22,776 - src.core.mt5_manager - INFO -    Balance: $194.55
2025-08-23 23:05:22,776 - src.core.mt5_manager - INFO -    Server: Exness-MT5Trial6
2025-08-23 23:05:22,776 - src.core.mt5_manager - INFO -    Symbol: XAUUSDm
2025-08-23 23:05:22,779 - src.core.mt5_manager - INFO - Retrieved 200 bars for XAUUSDm M15
2025-08-23 23:05:22,791 - IchimokuStrategy - INFO - Signal confidence 0.6 below threshold 0.65
2025-08-23 23:05:22,791 - IchimokuStrategy - INFO - Signal confidence 0.62 below threshold 0.65
2025-08-23 23:05:22,791 - IchimokuStrategy - INFO - Ichimoku generated 2 valid signals out of 4 total
2025-08-23 23:05:22,793 - src.core.mt5_manager - INFO - Retrieved 200 bars for XAUUSDm M15

------------------------------------------------------------

================================================================================

### Strategy: src/strategies/technical/market_profile.py
### Strategy Name: market_profile

--- Mode: MOCK ---
STDOUT OUTPUT:
============================================================
RUN MODE: MOCK - using simulated OHLCV data
============================================================
============================================================
TESTING MARKET PROFILE STRATEGY
============================================================
Running in MOCK mode

1. Testing signal generation:
   Current price: 3381.82
   POC: 3340.97
   VAH: 3359.26
   VAL: 3340.97
   Generated 5 signals
   - BUY at 3381.82, Confidence: 0.550, Grade: D
     Pattern: vah_breakout_price
   - BUY at 3381.82, Confidence: 0.600, Grade: C
     Pattern: above_poc_bullish
   - SELL at 3381.82, Confidence: 0.550, Grade: D
     Pattern: scalp_momentum
   - SELL at 3381.82, Confidence: 0.570, Grade: D
     Pattern: mean_reversion_poc
   - BUY at 3381.82, Confidence: 0.550, Grade: D
     Pattern: contrarian_momentum

2. Testing analysis method:
   Analysis keys: ['poc', 'vah', 'val', 'ib_high', 'ib_low', 'day_type', 'current_price', 'position_vs_value_area', 'tpo_distribution']
   POC: 3340.97
   VAH: 3359.26
   VAL: 3340.97
   IB High: 3391.40
   IB Low: 3373.70
   Day Type: TREND

3. Testing performance tracking:
   {'strategy_name': 'MarketProfileStrategy', 'total_signals': 0, 'win_rate': 0.0, 'profit_factor': 0.0, 'total_pnl': 0.0, 'grade_distribution': {'A': 0, 'B': 0, 'C': 0, 'D': 0}, 'last_signal': None}

4. Strategy Information:
   Name: Market Profile Strategy
   Type: Technical
   Description: Trades breakouts, rotations, and reversals based on Market Profile analysis
   Parameters: {'lookback_period': 200, 'value_area_pct': 0.7, 'ib_period': 60, 'confidence_threshold': 0.55, 'min_price_distance': 0.08, 'breakout_buffer': 0.0005}
   Performance Summary:
     Success Rate: 0.00%
     Profit Factor: 0.00

============================================================
MARKET PROFILE STRATEGY TEST COMPLETED!
============================================================

------------------------------------------------------------

--- Mode: LIVE ---
STDOUT OUTPUT:
============================================================
RUN MODE: LIVE - connecting to MT5 (production)
============================================================
✅ Connected to live MT5
============================================================
TESTING MARKET PROFILE STRATEGY
============================================================
Running in LIVE mode

1. Testing signal generation:
   Current price: 3372.30
   POC: 3338.86
   VAH: 3347.03
   VAL: 3329.12
   Generated 6 signals
   - BUY at 3372.30, Confidence: 0.650, Grade: C
     Pattern: vah_breakout_momentum
   - BUY at 3372.30, Confidence: 0.600, Grade: C
     Pattern: ib_breakout_buy
   - BUY at 3372.30, Confidence: 0.600, Grade: C
     Pattern: above_poc_bullish
   - BUY at 3372.30, Confidence: 0.550, Grade: D
     Pattern: scalp_momentum
   - SELL at 3372.30, Confidence: 0.570, Grade: D
     Pattern: mean_reversion_poc
   - SELL at 3372.30, Confidence: 0.550, Grade: D
     Pattern: contrarian_momentum

2. Testing analysis method:
   Analysis keys: ['poc', 'vah', 'val', 'ib_high', 'ib_low', 'day_type', 'current_price', 'position_vs_value_area', 'tpo_distribution']
   POC: 3338.86
   VAH: 3347.03
   VAL: 3329.12
   IB High: 3340.22
   IB Low: 3335.06
   Day Type: TREND

3. Testing performance tracking:
   {'strategy_name': 'MarketProfileStrategy', 'total_signals': 0, 'win_rate': 0.0, 'profit_factor': 0.0, 'total_pnl': 0.0, 'grade_distribution': {'A': 0, 'B': 0, 'C': 0, 'D': 0}, 'last_signal': None}

4. Strategy Information:
   Name: Market Profile Strategy
   Type: Technical
   Description: Trades breakouts, rotations, and reversals based on Market Profile analysis
   Parameters: {'lookback_period': 200, 'value_area_pct': 0.7, 'ib_period': 60, 'confidence_threshold': 0.55, 'min_price_distance': 0.08, 'breakout_buffer': 0.0005}
   Performance Summary:
     Success Rate: 0.00%
     Profit Factor: 0.00

============================================================
MARKET PROFILE STRATEGY TEST COMPLETED!
============================================================

STDERR OUTPUT:
2025-08-23 23:05:24,541 - src.core.mt5_manager - INFO - Loaded 227 tradable symbols from CSV
2025-08-23 23:05:24,541 - src.core.mt5_manager - INFO - MT5Manager initialized for symbol: XAUUSD
2025-08-23 23:05:24,541 - src.core.mt5_manager - INFO - Magic number: 123456
2025-08-23 23:05:24,541 - src.core.mt5_manager - INFO - Initializing MT5 with default path
2025-08-23 23:05:24,554 - src.core.mt5_manager - INFO - Attempting to login to account 273949055 on server Exness-MT5Trial6
2025-08-23 23:05:24,555 - src.core.mt5_manager - WARNING - Symbol XAUUSD not found, trying alternative symbols
2025-08-23 23:05:24,555 - src.core.mt5_manager - INFO - Using alternative symbol: XAUUSDm
2025-08-23 23:05:24,555 - src.core.mt5_manager - INFO - ✅ Successfully connected to MT5
2025-08-23 23:05:24,555 - src.core.mt5_manager - INFO -    Account: 273949055
2025-08-23 23:05:24,555 - src.core.mt5_manager - INFO -    Balance: $194.55
2025-08-23 23:05:24,555 - src.core.mt5_manager - INFO -    Server: Exness-MT5Trial6
2025-08-23 23:05:24,555 - src.core.mt5_manager - INFO -    Symbol: XAUUSDm
2025-08-23 23:05:24,555 - MarketProfileStrategy - INFO - Market Profile Strategy initialized
2025-08-23 23:05:24,559 - src.core.mt5_manager - INFO - Retrieved 200 bars for XAUUSDm M15
2025-08-23 23:05:24,571 - src.core.mt5_manager - INFO - Retrieved 200 bars for XAUUSDm M15
2025-08-23 23:05:24,586 - src.core.mt5_manager - INFO - Retrieved 200 bars for XAUUSDm M15

------------------------------------------------------------

================================================================================

### Strategy: src/strategies/technical/momentum_divergence.py
### Strategy Name: momentum_divergence

--- Mode: MOCK ---
STDOUT OUTPUT:
============================================================
RUN MODE: MOCK - using simulated OHLCV data
============================================================
============================================================
TESTING MOMENTUM DIVERGENCE STRATEGY
============================================================
Running in MOCK mode

1. Testing signal generation:
   Generated 0 signals

2. Testing analysis method:
   Analysis results keys: dict_keys(['oscillator', 'divergences_detected', 'recent_rsi_values', 'recent_macd_hist', 'trend_context'])
   Detected divergences: 0

3. Testing performance tracking:
   {'strategy_name': 'MomentumDivergenceStrategy', 'total_signals': 0, 'win_rate': 0.0, 'profit_factor': 0.0, 'total_pnl': 0.0, 'grade_distribution': {'A': 0, 'B': 0, 'C': 0, 'D': 0}, 'last_signal': None}

4. Strategy Information:
   Name: MomentumDivergenceStrategy
   Type: Technical
   Description: Detects regular and hidden divergences using RSI or MACD for trend reversal and continuation setups
   Parameters:
     lookback_period: 200
     oscillator: RSI
     rsi_period: 14
     macd_fast: 12
     macd_slow: 26
     macd_signal: 9
     divergence_tolerance: 2
     confidence_threshold: 0.65
     cooldown_bars: 3
   Performance Summary:
     Success Rate: 0.00%
     Profit Factor: 0.00

============================================================
MOMENTUM DIVERGENCE STRATEGY TEST COMPLETED!
============================================================

------------------------------------------------------------

--- Mode: LIVE ---
STDOUT OUTPUT:
============================================================
RUN MODE: LIVE - connecting to MT5 (production)
============================================================
✅ Connected to live MT5
============================================================
TESTING MOMENTUM DIVERGENCE STRATEGY
============================================================
Running in LIVE mode

1. Testing signal generation:
   Generated 0 signals

2. Testing analysis method:
   Analysis results keys: dict_keys(['oscillator', 'divergences_detected', 'recent_rsi_values', 'recent_macd_hist', 'trend_context'])
   Detected divergences: 0

3. Testing performance tracking:
   {'strategy_name': 'MomentumDivergenceStrategy', 'total_signals': 0, 'win_rate': 0.0, 'profit_factor': 0.0, 'total_pnl': 0.0, 'grade_distribution': {'A': 0, 'B': 0, 'C': 0, 'D': 0}, 'last_signal': None}

4. Strategy Information:
   Name: MomentumDivergenceStrategy
   Type: Technical
   Description: Detects regular and hidden divergences using RSI or MACD for trend reversal and continuation setups
   Parameters:
     lookback_period: 200
     oscillator: RSI
     rsi_period: 14
     macd_fast: 12
     macd_slow: 26
     macd_signal: 9
     divergence_tolerance: 2
     confidence_threshold: 0.65
     cooldown_bars: 3
   Performance Summary:
     Success Rate: 0.00%
     Profit Factor: 0.00

============================================================
MOMENTUM DIVERGENCE STRATEGY TEST COMPLETED!
============================================================

STDERR OUTPUT:
2025-08-23 23:05:26,552 - src.core.mt5_manager - INFO - Loaded 227 tradable symbols from CSV
2025-08-23 23:05:26,552 - src.core.mt5_manager - INFO - MT5Manager initialized for symbol: XAUUSD
2025-08-23 23:05:26,552 - src.core.mt5_manager - INFO - Magic number: 123456
2025-08-23 23:05:26,552 - src.core.mt5_manager - INFO - Initializing MT5 with default path
2025-08-23 23:05:26,565 - src.core.mt5_manager - INFO - Attempting to login to account 273949055 on server Exness-MT5Trial6
2025-08-23 23:05:26,565 - src.core.mt5_manager - WARNING - Symbol XAUUSD not found, trying alternative symbols
2025-08-23 23:05:26,565 - src.core.mt5_manager - INFO - Using alternative symbol: XAUUSDm
2025-08-23 23:05:26,565 - src.core.mt5_manager - INFO - ✅ Successfully connected to MT5
2025-08-23 23:05:26,565 - src.core.mt5_manager - INFO -    Account: 273949055
2025-08-23 23:05:26,566 - src.core.mt5_manager - INFO -    Balance: $194.55
2025-08-23 23:05:26,566 - src.core.mt5_manager - INFO -    Server: Exness-MT5Trial6
2025-08-23 23:05:26,566 - src.core.mt5_manager - INFO -    Symbol: XAUUSDm
2025-08-23 23:05:26,566 - MomentumDivergenceStrategy - INFO - Initialized MomentumDivergenceStrategy with oscillator: RSI
2025-08-23 23:05:26,569 - src.core.mt5_manager - INFO - Retrieved 200 bars for XAUUSDm M15
2025-08-23 23:05:26,620 - MomentumDivergenceStrategy - INFO - Generated 0 signals for XAUUSDm on M15
2025-08-23 23:05:26,623 - src.core.mt5_manager - INFO - Retrieved 200 bars for XAUUSDm M15

------------------------------------------------------------

================================================================================

### Strategy: src/strategies/technical/order_flow.py
### Strategy Name: order_flow

--- Mode: MOCK ---
STDOUT OUTPUT:
============================================================
RUN MODE: MOCK - using simulated OHLCV data
============================================================
============================================================
TESTING ORDER FLOW STRATEGY
============================================================
Running in MOCK mode

1. Testing signal generation:
   Generated 0 signals

2. Testing analysis method:
   Analysis results keys: dict_keys(['last_cdv', 'max_cdv', 'min_cdv', 'recent_imbalances', 'absorption_zones', 'trend_direction'])
   Detected imbalances: 0
   Absorption zones: 1

3. Testing performance tracking:
   {'strategy_name': 'OrderFlowStrategy', 'total_signals': 0, 'win_rate': 0.0, 'profit_factor': 0.0, 'total_pnl': 0.0, 'grade_distribution': {'A': 0, 'B': 0, 'C': 0, 'D': 0}, 'last_signal': None}

4. Strategy Information:
   Name: Order Flow Strategy
   Version: 1.0.0
   Description: Analyzes market order flow using Cumulative Delta Volume, bid/ask imbalances, and absorption events for XAUUSD trading
   Type: Technical
   Parameters:
     lookback_period: 150
     imbalance_threshold: 1.3
     absorption_threshold: 1.5
     confidence_threshold: 0.65
     min_bar_volume: 100
     cooldown_bars: 3
   Performance Summary:
     Success Rate: 0.00%
     Profit Factor: 0.00

============================================================
ORDER FLOW STRATEGY TEST COMPLETED!
============================================================

------------------------------------------------------------

--- Mode: LIVE ---
STDOUT OUTPUT:
============================================================
RUN MODE: LIVE - connecting to MT5 (production)
============================================================
✅ Connected to live MT5
============================================================
TESTING ORDER FLOW STRATEGY
============================================================
Running in LIVE mode

1. Testing signal generation:
   Generated 1 signals
   - BUY at 3372.30, Confidence: 0.950, Grade: A
     CDV: 46840.40
     Imbalance Ratio: 5.25

2. Testing analysis method:
   Analysis results keys: dict_keys(['last_cdv', 'max_cdv', 'min_cdv', 'recent_imbalances', 'absorption_zones', 'trend_direction'])
   Detected imbalances: 0
   Absorption zones: 0

3. Testing performance tracking:
   {'strategy_name': 'OrderFlowStrategy', 'total_signals': 0, 'win_rate': 0.0, 'profit_factor': 0.0, 'total_pnl': 0.0, 'grade_distribution': {'A': 0, 'B': 0, 'C': 0, 'D': 0}, 'last_signal': None}

4. Strategy Information:
   Name: Order Flow Strategy
   Version: 1.0.0
   Description: Analyzes market order flow using Cumulative Delta Volume, bid/ask imbalances, and absorption events for XAUUSD trading
   Type: Technical
   Parameters:
     lookback_period: 150
     imbalance_threshold: 1.3
     absorption_threshold: 1.5
     confidence_threshold: 0.65
     min_bar_volume: 100
     cooldown_bars: 3
   Performance Summary:
     Success Rate: 0.00%
     Profit Factor: 0.00

============================================================
ORDER FLOW STRATEGY TEST COMPLETED!
============================================================

STDERR OUTPUT:
2025-08-23 23:05:28,445 - src.core.mt5_manager - INFO - Loaded 227 tradable symbols from CSV
2025-08-23 23:05:28,446 - src.core.mt5_manager - INFO - MT5Manager initialized for symbol: XAUUSD
2025-08-23 23:05:28,446 - src.core.mt5_manager - INFO - Magic number: 123456
2025-08-23 23:05:28,446 - src.core.mt5_manager - INFO - Initializing MT5 with default path
2025-08-23 23:05:28,458 - src.core.mt5_manager - INFO - Attempting to login to account 273949055 on server Exness-MT5Trial6
2025-08-23 23:05:28,458 - src.core.mt5_manager - WARNING - Symbol XAUUSD not found, trying alternative symbols
2025-08-23 23:05:28,458 - src.core.mt5_manager - INFO - Using alternative symbol: XAUUSDm
2025-08-23 23:05:28,458 - src.core.mt5_manager - INFO - ✅ Successfully connected to MT5
2025-08-23 23:05:28,458 - src.core.mt5_manager - INFO -    Account: 273949055
2025-08-23 23:05:28,458 - src.core.mt5_manager - INFO -    Balance: $194.55
2025-08-23 23:05:28,458 - src.core.mt5_manager - INFO -    Server: Exness-MT5Trial6
2025-08-23 23:05:28,458 - src.core.mt5_manager - INFO -    Symbol: XAUUSDm
2025-08-23 23:05:28,458 - OrderFlowStrategy - INFO - OrderFlowStrategy initialized successfully
2025-08-23 23:05:28,462 - src.core.mt5_manager - INFO - Retrieved 150 bars for XAUUSDm M15
2025-08-23 23:05:28,466 - OrderFlowStrategy - INFO - Generated BUY signal for XAUUSDm at 3372.30, Confidence: 0.95
2025-08-23 23:05:28,468 - src.core.mt5_manager - INFO - Retrieved 150 bars for XAUUSDm M15

------------------------------------------------------------

================================================================================

### Strategy: src/strategies/technical/volume_profile.py
### Strategy Name: volume_profile

--- Mode: MOCK ---
STDOUT OUTPUT:
============================================================
RUN MODE: MOCK - using simulated OHLCV data
============================================================
============================================================
TESTING VOLUME PROFILE STRATEGY
============================================================
Running in MOCK mode

1. Testing signal generation:
   Generated 10 signals
   - SELL at 1933.69, Confidence: 0.748, Grade: B
     Metadata: {'level_type': 'HVN_reversal', 'level': 1924.696513648254, 'original_timestamp': datetime.datetime(2025, 8, 23, 23, 5, 29, 516434)}
   - SELL at 1933.69, Confidence: 0.797, Grade: B
     Metadata: {'level_type': 'HVN_reversal', 'level': 1926.196513648254, 'original_timestamp': datetime.datetime(2025, 8, 23, 23, 5, 29, 516474)}
   - SELL at 1933.69, Confidence: 0.824, Grade: B
     Metadata: {'level_type': 'HVN_reversal', 'level': 1927.196513648254, 'original_timestamp': datetime.datetime(2025, 8, 23, 23, 5, 29, 516493)}
   - SELL at 1933.69, Confidence: 0.797, Grade: B
     Metadata: {'level_type': 'HVN_reversal', 'level': 1927.696513648254, 'original_timestamp': datetime.datetime(2025, 8, 23, 23, 5, 29, 516521)}
   - SELL at 1933.69, Confidence: 0.816, Grade: B
     Metadata: {'level_type': 'HVN_reversal', 'level': 1928.196513648254, 'original_timestamp': datetime.datetime(2025, 8, 23, 23, 5, 29, 516531)}
   - SELL at 1933.69, Confidence: 0.814, Grade: B
     Metadata: {'level_type': 'HVN_reversal', 'level': 1928.696513648254, 'original_timestamp': datetime.datetime(2025, 8, 23, 23, 5, 29, 516542)}
   - SELL at 1933.69, Confidence: 0.904, Grade: A
     Metadata: {'level_type': 'POC_reversal', 'level': 1929.196513648254, 'original_timestamp': datetime.datetime(2025, 8, 23, 23, 5, 29, 516550)}
   - SELL at 1933.69, Confidence: 0.841, Grade: B
     Metadata: {'level_type': 'HVN_reversal', 'level': 1929.696513648254, 'original_timestamp': datetime.datetime(2025, 8, 23, 23, 5, 29, 516562)}
   - SELL at 1933.69, Confidence: 0.829, Grade: B
     Metadata: {'level_type': 'HVN_reversal', 'level': 1930.196513648254, 'original_timestamp': datetime.datetime(2025, 8, 23, 23, 5, 29, 516570)}
   - SELL at 1933.69, Confidence: 0.849, Grade: B
     Metadata: {'level_type': 'HVN_reversal', 'level': 1930.696513648254, 'original_timestamp': datetime.datetime(2025, 8, 23, 23, 5, 29, 516579)}

2. Testing analysis method:
   Analysis keys: ['poc', 'vah', 'val', 'high_volume_nodes', 'low_volume_nodes', 'current_price', 'position_vs_value_area', 'volume_distribution']
   POC: 1929.20
   VAH: 1958.70
   VAL: 1924.70
   HVNs: [1924.696513648254, 1926.196513648254, 1927.196513648254, 1927.696513648254, 1928.196513648254, 1928.696513648254, 1929.196513648254, 1929.696513648254, 1930.196513648254, 1930.696513648254, 1931.196513648254, 1931.696513648254, 1932.196513648254, 1933.196513648254, 1933.696513648254, 1934.196513648254, 1934.696513648254, 1936.196513648254, 1938.196513648254, 1941.696513648254, 1950.696513648254, 1953.696513648254, 1957.196513648254, 1957.696513648254, 1958.696513648254]
   LVNs: [1919.196513648254, 1919.696513648254, 1920.196513648254, 1920.696513648254, 1921.196513648254, 1921.696513648254, 1922.196513648254, 1922.696513648254, 1923.696513648254, 1935.196513648254, 1937.696513648254, 1938.696513648254, 1939.196513648254, 1942.196513648254, 1942.696513648254, 1943.196513648254, 1943.696513648254, 1944.196513648254, 1944.696513648254, 1945.196513648254, 1946.196513648254, 1946.696513648254, 1947.196513648254, 1947.696513648254, 1948.196513648254, 1948.696513648254, 1949.196513648254, 1949.696513648254, 1951.196513648254, 1952.196513648254, 1952.696513648254, 1953.196513648254, 1955.196513648254, 1955.696513648254, 1956.196513648254, 1958.196513648254, 1959.196513648254, 1959.696513648254, 1960.196513648254, 1960.696513648254]
   Position vs VA: within

3. Testing performance tracking:
   {'strategy_name': 'VolumeProfileStrategy', 'total_signals': 0, 'win_rate': 0.0, 'profit_factor': 0.0, 'total_pnl': 0.0, 'grade_distribution': {'A': 0, 'B': 0, 'C': 0, 'D': 0}, 'last_signal': None}

4. Strategy Information:
   Name: Volume Profile
   Version: 1.0.0
   Description: Identifies high/low volume nodes, POC, and value areas for support/resistance and breakout signals.
   Type: Technical
   Parameters:
     lookback_period: 200
     value_area_pct: 0.7
     volume_node_threshold: 1.3
     confidence_threshold: 0.65
     min_price_distance: 0.2
   Performance Summary:
     Success Rate: 0.00%
     Profit Factor: 0.00

============================================================
VOLUME PROFILE STRATEGY TEST COMPLETED!
============================================================

STDERR OUTPUT:
J:\Gold_FX\src\strategies\technical\volume_profile.py:343: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
  volume_dist = data.groupby('bin')['Volume'].sum().to_dict()
J:\Gold_FX\src\strategies\technical\volume_profile.py:343: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
  volume_dist = data.groupby('bin')['Volume'].sum().to_dict()

------------------------------------------------------------

--- Mode: LIVE ---
STDOUT OUTPUT:
============================================================
RUN MODE: LIVE - connecting to MT5 (production)
============================================================
✅ Connected to live MT5
============================================================
TESTING VOLUME PROFILE STRATEGY
============================================================
Running in LIVE mode

1. Testing signal generation:
   Generated 10 signals
   - BUY at 3372.30, Confidence: 0.650, Grade: C
     Metadata: {'level_type': 'LVN_breakout', 'level': 3355.45, 'original_timestamp': datetime.datetime(2025, 8, 23, 23, 5, 30, 468000)}
   - BUY at 3372.30, Confidence: 0.653, Grade: C
     Metadata: {'level_type': 'LVN_breakout', 'level': 3355.95, 'original_timestamp': datetime.datetime(2025, 8, 23, 23, 5, 30, 468071)}
   - BUY at 3372.30, Confidence: 0.656, Grade: C
     Metadata: {'level_type': 'LVN_breakout', 'level': 3356.45, 'original_timestamp': datetime.datetime(2025, 8, 23, 23, 5, 30, 468107)}
   - BUY at 3372.30, Confidence: 0.659, Grade: C
     Metadata: {'level_type': 'LVN_breakout', 'level': 3356.95, 'original_timestamp': datetime.datetime(2025, 8, 23, 23, 5, 30, 468129)}
   - BUY at 3372.30, Confidence: 0.662, Grade: C
     Metadata: {'level_type': 'LVN_breakout', 'level': 3357.45, 'original_timestamp': datetime.datetime(2025, 8, 23, 23, 5, 30, 468148)}
   - BUY at 3372.30, Confidence: 0.665, Grade: C
     Metadata: {'level_type': 'LVN_breakout', 'level': 3357.95, 'original_timestamp': datetime.datetime(2025, 8, 23, 23, 5, 30, 468168)}
   - SELL at 3372.30, Confidence: 0.858, Grade: A
     Metadata: {'level_type': 'HVN_reversal', 'level': 3358.45, 'original_timestamp': datetime.datetime(2025, 8, 23, 23, 5, 30, 468188)}
   - BUY at 3372.30, Confidence: 0.671, Grade: C
     Metadata: {'level_type': 'LVN_breakout', 'level': 3358.95, 'original_timestamp': datetime.datetime(2025, 8, 23, 23, 5, 30, 468205)}
   - BUY at 3372.30, Confidence: 0.674, Grade: C
     Metadata: {'level_type': 'LVN_breakout', 'level': 3359.45, 'original_timestamp': datetime.datetime(2025, 8, 23, 23, 5, 30, 468221)}
   - BUY at 3372.30, Confidence: 0.677, Grade: C
     Metadata: {'level_type': 'LVN_breakout', 'level': 3359.95, 'original_timestamp': datetime.datetime(2025, 8, 23, 23, 5, 30, 468237)}

2. Testing analysis method:
   Analysis keys: ['poc', 'vah', 'val', 'high_volume_nodes', 'low_volume_nodes', 'current_price', 'position_vs_value_area', 'volume_distribution']
   POC: 3338.45
   VAH: 3376.45
   VAL: 3327.95
   HVNs: [3325.95, 3327.95, 3328.45, 3328.95, 3329.95, 3330.45, 3333.45, 3333.95, 3336.95, 3337.45, 3337.95, 3338.45, 3338.95, 3339.45, 3339.95, 3340.95, 3341.45, 3341.95, 3342.45, 3342.95, 3343.45, 3343.95, 3344.45, 3344.95, 3345.45, 3347.45, 3348.45, 3358.45, 3365.95, 3370.45, 3371.45, 3372.95, 3373.45, 3376.45]
   LVNs: [3321.45, 3321.95, 3322.45, 3322.95, 3324.95, 3325.45, 3326.45, 3326.95, 3331.45, 3332.45, 3332.95, 3334.95, 3335.45, 3335.95, 3346.45, 3346.95, 3347.95, 3348.95, 3349.45, 3350.45, 3350.95, 3351.45, 3351.95, 3352.45, 3352.95, 3353.45, 3353.95, 3354.45, 3354.95, 3355.45, 3355.95, 3356.45, 3356.95, 3357.45, 3357.95, 3358.95, 3359.45, 3359.95, 3360.45, 3360.95, 3361.45, 3361.95, 3362.45, 3362.95, 3363.45, 3363.95, 3364.45, 3364.95, 3365.45, 3366.45, 3366.95, 3367.95, 3368.45, 3368.95, 3369.45, 3370.95, 3374.45, 3374.95, 3375.45, 3375.95, 3376.95, 3377.45, 3377.95, 3378.45]
   Position vs VA: within

3. Testing performance tracking:
   {'strategy_name': 'VolumeProfileStrategy', 'total_signals': 0, 'win_rate': 0.0, 'profit_factor': 0.0, 'total_pnl': 0.0, 'grade_distribution': {'A': 0, 'B': 0, 'C': 0, 'D': 0}, 'last_signal': None}

4. Strategy Information:
   Name: Volume Profile
   Version: 1.0.0
   Description: Identifies high/low volume nodes, POC, and value areas for support/resistance and breakout signals.
   Type: Technical
   Parameters:
     lookback_period: 200
     value_area_pct: 0.7
     volume_node_threshold: 1.3
     confidence_threshold: 0.65
     min_price_distance: 0.2
   Performance Summary:
     Success Rate: 0.00%
     Profit Factor: 0.00

============================================================
VOLUME PROFILE STRATEGY TEST COMPLETED!
============================================================

STDERR OUTPUT:
2025-08-23 23:05:30,405 - src.core.mt5_manager - INFO - Loaded 227 tradable symbols from CSV
2025-08-23 23:05:30,405 - src.core.mt5_manager - INFO - MT5Manager initialized for symbol: XAUUSD
2025-08-23 23:05:30,406 - src.core.mt5_manager - INFO - Magic number: 123456
2025-08-23 23:05:30,406 - src.core.mt5_manager - INFO - Initializing MT5 with default path
2025-08-23 23:05:30,418 - src.core.mt5_manager - INFO - Attempting to login to account 273949055 on server Exness-MT5Trial6
2025-08-23 23:05:30,419 - src.core.mt5_manager - WARNING - Symbol XAUUSD not found, trying alternative symbols
2025-08-23 23:05:30,419 - src.core.mt5_manager - INFO - Using alternative symbol: XAUUSDm
2025-08-23 23:05:30,419 - src.core.mt5_manager - INFO - ✅ Successfully connected to MT5
2025-08-23 23:05:30,419 - src.core.mt5_manager - INFO -    Account: 273949055
2025-08-23 23:05:30,419 - src.core.mt5_manager - INFO -    Balance: $194.55
2025-08-23 23:05:30,419 - src.core.mt5_manager - INFO -    Server: Exness-MT5Trial6
2025-08-23 23:05:30,419 - src.core.mt5_manager - INFO -    Symbol: XAUUSDm
2025-08-23 23:05:30,422 - src.core.mt5_manager - INFO - Retrieved 200 bars for XAUUSDm M15
J:\Gold_FX\src\strategies\technical\volume_profile.py:343: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
  volume_dist = data.groupby('bin')['Volume'].sum().to_dict()
2025-08-23 23:05:30,472 - src.core.mt5_manager - INFO - Retrieved 200 bars for XAUUSDm M15
J:\Gold_FX\src\strategies\technical\volume_profile.py:343: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
  volume_dist = data.groupby('bin')['Volume'].sum().to_dict()

------------------------------------------------------------

================================================================================

### Strategy: src/strategies/technical/wyckoff.py
### Strategy Name: wyckoff

--- Mode: MOCK ---
STDOUT OUTPUT:
============================================================
RUN MODE: MOCK - using simulated OHLCV data
============================================================
============================================================
TESTING WYCKOFF STRATEGY
============================================================
Running in MOCK mode

1. Testing signal generation:
   Generated 0 signals

2. Testing analysis method:
   Analysis keys: ['current_phase', 'detected_events', 'range_high', 'range_low', 'trend_context', 'analysis_time']
   Current Phase: DISTRIBUTION
   Detected Events: 1
   Latest Event: MARKUP_SIGNAL

3. Testing performance tracking:
   {'strategy_name': 'WyckoffStrategy', 'total_signals': 0, 'win_rate': 0.0, 'profit_factor': 0.0, 'total_pnl': 0.0, 'grade_distribution': {'A': 0, 'B': 0, 'C': 0, 'D': 0}, 'last_signal': None}

4. Strategy Information:
   Name: Wyckoff Strategy
   Type: Technical
   Description: Implements Wyckoff Method for phase and event detection
   Events Supported: SPRING, UPTHRUST, SIGN_OF_STRENGTH, SIGN_OF_WEAKNESS, ACCUMULATION_SIGNAL, DISTRIBUTION_SIGNAL, MARKUP_SIGNAL, MARKDOWN_SIGNAL, VOLUME_CLIMAX, NO_SUPPLY
   Phases Supported: ACCUMULATION, DISTRIBUTION, RE_ACCUMULATION, RE_DISTRIBUTION
   Lookback Period: 80
   Confidence Threshold: 0.51
   Performance Summary:
     Success Rate: 0.00%
     Profit Factor: 0.00

============================================================
WYCKOFF STRATEGY TEST COMPLETED!
============================================================

------------------------------------------------------------

--- Mode: LIVE ---
STDOUT OUTPUT:
============================================================
RUN MODE: LIVE - connecting to MT5 (production)
============================================================
✅ Connected to live MT5
============================================================
TESTING WYCKOFF STRATEGY
============================================================
Running in LIVE mode

1. Testing signal generation:
   Generated 0 signals

2. Testing analysis method:
   Analysis keys: ['current_phase', 'detected_events', 'range_high', 'range_low', 'trend_context', 'analysis_time']
   Current Phase: DISTRIBUTION
   Detected Events: 1
   Latest Event: MARKUP_SIGNAL

3. Testing performance tracking:
   {'strategy_name': 'WyckoffStrategy', 'total_signals': 0, 'win_rate': 0.0, 'profit_factor': 0.0, 'total_pnl': 0.0, 'grade_distribution': {'A': 0, 'B': 0, 'C': 0, 'D': 0}, 'last_signal': None}

4. Strategy Information:
   Name: Wyckoff Strategy
   Type: Technical
   Description: Implements Wyckoff Method for phase and event detection
   Events Supported: SPRING, UPTHRUST, SIGN_OF_STRENGTH, SIGN_OF_WEAKNESS, ACCUMULATION_SIGNAL, DISTRIBUTION_SIGNAL, MARKUP_SIGNAL, MARKDOWN_SIGNAL, VOLUME_CLIMAX, NO_SUPPLY
   Phases Supported: ACCUMULATION, DISTRIBUTION, RE_ACCUMULATION, RE_DISTRIBUTION
   Lookback Period: 80
   Confidence Threshold: 0.51
   Performance Summary:
     Success Rate: 0.00%
     Profit Factor: 0.00

============================================================
WYCKOFF STRATEGY TEST COMPLETED!
============================================================

STDERR OUTPUT:
2025-08-23 23:05:32,251 - src.core.mt5_manager - INFO - Loaded 227 tradable symbols from CSV
2025-08-23 23:05:32,251 - src.core.mt5_manager - INFO - MT5Manager initialized for symbol: XAUUSD
2025-08-23 23:05:32,251 - src.core.mt5_manager - INFO - Magic number: 123456
2025-08-23 23:05:32,252 - src.core.mt5_manager - INFO - Initializing MT5 with default path
2025-08-23 23:05:32,266 - src.core.mt5_manager - INFO - Attempting to login to account 273949055 on server Exness-MT5Trial6
2025-08-23 23:05:32,267 - src.core.mt5_manager - WARNING - Symbol XAUUSD not found, trying alternative symbols
2025-08-23 23:05:32,267 - src.core.mt5_manager - INFO - Using alternative symbol: XAUUSDm
2025-08-23 23:05:32,267 - src.core.mt5_manager - INFO - ✅ Successfully connected to MT5
2025-08-23 23:05:32,267 - src.core.mt5_manager - INFO -    Account: 273949055
2025-08-23 23:05:32,267 - src.core.mt5_manager - INFO -    Balance: $194.55
2025-08-23 23:05:32,267 - src.core.mt5_manager - INFO -    Server: Exness-MT5Trial6
2025-08-23 23:05:32,267 - src.core.mt5_manager - INFO -    Symbol: XAUUSDm
2025-08-23 23:05:32,267 - WyckoffStrategy - INFO - WyckoffStrategy initialized with lookback=80, confidence_threshold=0.51
2025-08-23 23:05:32,271 - src.core.mt5_manager - INFO - Retrieved 80 bars for XAUUSDm M15
2025-08-23 23:05:32,294 - WyckoffStrategy - INFO - Signal confidence 0.51 below threshold 0.6
2025-08-23 23:05:32,296 - src.core.mt5_manager - INFO - Retrieved 150 bars for XAUUSDm M15

------------------------------------------------------------

================================================================================

