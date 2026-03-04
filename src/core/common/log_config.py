"""
Logging configuration with category-based filtering.

Log categories:
1 = LEARNING - Model training, loss, Q-values, replay buffer
2 = INTENT - Intent creation and updates
3 = SCORING/PLAYBOOK - Q-value scoring, playbook generation
4 = DEVIATION - Deviation detection
5 = OBSERVER - State window generation, feature completeness
6 = REWARD - Reward computation details
7 = KPI - Low-level KPI I/O (message reception, node_id extraction)
8 = COMMANDS - Command execution details
"""

import logging
from typing import Set

# Log categories
LOG_LEARNING = 1
LOG_INTENT = 2
LOG_SCORING = 3
LOG_DEVIATION = 4
LOG_OBSERVER = 5
LOG_REWARD = 6
LOG_KPI = 7
LOG_COMMANDS = 8
LOG_BANDIT = 9

# Category names for display
CATEGORY_NAMES = {
    LOG_LEARNING: "LEARNING",
    LOG_INTENT: "INTENT",
    LOG_SCORING: "SCORING/PLAYBOOK",
    LOG_DEVIATION: "DEVIATION/COMMANDS",
    LOG_OBSERVER: "OBSERVER",
    LOG_REWARD: "REWARD",
    LOG_KPI: "KPI",
    LOG_COMMANDS: "COMMANDS",
    LOG_BANDIT: "BANDIT",
}

# Global log level configuration
_log_levels: Set[int] = set()

def set_log_levels(levels: Set[int]):
    """Set which log categories to enable.
    
    Args:
        levels: Set of category numbers (1-7) to enable
    """
    global _log_levels
    _log_levels = levels
    enabled = [CATEGORY_NAMES.get(l, str(l)) for l in sorted(levels)]
    logging.info(f"[LOG_CONFIG] Enabled log categories: {', '.join(enabled)}")

def parse_log_levels(level_str: str) -> Set[int]:
    """Parse log level string (comma-separated numbers or 'all').
    
    Examples:
        "1,2,3" -> {1, 2, 3}
        "all" -> {1, 2, 3, 4, 5, 6, 7}
        "1" -> {1}
    """
    if not level_str or level_str.lower() == "all":
        return {LOG_LEARNING, LOG_INTENT, LOG_SCORING, LOG_DEVIATION, LOG_OBSERVER, LOG_REWARD, LOG_KPI}
    
    levels = set()
    for part in level_str.split(','):
        part = part.strip()
        try:
            level = int(part)
            if 1 <= level <= 9:
                levels.add(level)
            else:
                logging.warning(f"Invalid log level: {level} (must be 1-9)")
        except ValueError:
            logging.warning(f"Invalid log level format: {part}")
    
    return levels

def should_log(category: int) -> bool:
    """Check if a log category should be logged.
    
    Args:
        category: Log category number (1-7)
    
    Returns:
        True if category is enabled, False otherwise
    """
    return category in _log_levels

def log_if_enabled(category: int, logger, level, message, *args, **kwargs):
    """Log a message only if the category is enabled.
    
    Args:
        category: Log category number (1-7)
        logger: Logger instance
        level: Log level (logging.INFO, logging.DEBUG, etc.)
        message: Log message
        *args, **kwargs: Additional arguments for logger
    """
    if should_log(category):
        logger.log(level, message, *args, **kwargs)

