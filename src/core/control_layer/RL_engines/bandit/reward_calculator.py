import numpy as np
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)


class SLORewardCalculator:
    """Minimal SLO-based reward calculator.

    Maps the LLM's declarative OTM (one objective, N constraints) with the
    RL agent's scalar reward signal using two terms:

        reward = improvement(objective) - weight * sum(violation²(constraints))

    The improvement term is a relative change that rewards moving the objective
    metric in the desired direction.  The violation term is a quadratic penalty
    that activates only when a constraint threshold is breached, penalising
    large breaches disproportionately harder than marginal ones.
    """

    def __init__(self, reward_clip: float = 20.0):
        self.reward_clip = reward_clip

    def calculate_reward(
        self,
        prev_metrics: Dict[str, float],
        curr_metrics: Dict[str, float],
        objective_metric: str,
        maximize: bool = True,
        constraints: List[Dict] = None,
        violation_weight: float = 5.0,
    ) -> float:
        """Minimal reward: objective improvement minus quadratic constraint violation.

        Args:
            prev_metrics: KPI values from the previous observation.
            curr_metrics: KPI values from the current observation.
            objective_metric: The KPI to optimise (e.g. ``UE_DRB_UEThpDl_UEID``).
            maximize: ``True`` to maximise the objective, ``False`` to minimise.
            constraints: List of constraint dicts, each with:
                ``"metric"``    - KPI name,
                ``"operator"``  - ``"le"``/``"lt"`` (upper bound) or ``"ge"``/``"gt"`` (lower bound),
                ``"threshold"`` - limit value.
            violation_weight: Multiplier applied to the total squared violation.

        Returns:
            Clipped scalar reward in ``[-reward_clip, reward_clip]``.
        """
        if constraints is None:
            constraints = []

        prev_obj = prev_metrics.get(objective_metric)
        curr_obj = curr_metrics.get(objective_metric)

        # Bail out if objective metrics are missing
        if prev_obj is None or curr_obj is None:
            return 0.0

        # Reward the Objective — relative improvement
        if maximize:
            improvement = (curr_obj - prev_obj) / max(abs(prev_obj), 1e-6)
        else:
            improvement = (prev_obj - curr_obj) / max(abs(prev_obj), 1e-6)

        # Penalize ALL violated constraints — quadratic penalty
        # Squaring makes small overshoots near-negligible while large
        # breaches receive disproportionately heavy penalties.
        total_violation = 0.0
        for con in constraints:
            curr_con = curr_metrics.get(con["metric"])
            if curr_con is None:
                continue

            threshold = con["threshold"]
            op = con["operator"]

            if op in ("le", "lt") and curr_con > threshold:
                rel = (curr_con - threshold) / max(abs(threshold), 1e-6)
                total_violation += rel ** 2
            elif op in ("ge", "gt") and curr_con < threshold:
                rel = (threshold - curr_con) / max(abs(threshold), 1e-6)
                total_violation += rel ** 2

        total_violation *= violation_weight

        # Final Reward: Improvement minus Penalty
        reward = improvement - total_violation
        return float(np.clip(reward, -self.reward_clip, self.reward_clip))
