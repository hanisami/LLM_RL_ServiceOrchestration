from __future__ import annotations

import random
from typing import Dict, List, Tuple


class QLearningAgent:
    def __init__(
        self,
        n_actions: int,
        alpha: float = 0.15,
        gamma: float = 0.95,
        eps_start: float = 0.30,
        eps_end: float = 0.05,
        eps_decay: float = 0.995,
        bias_beta: float = 2.5,
        seed: int = 0,
    ):
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        # Scales any external action-bias/prior when selecting actions.
        # This is intentionally modest because Q-values are on the order of a few hundred
        # (given per-step rewards ~[-20,0] and gamma ~0.95).
        self.bias_beta = float(bias_beta)
        self.rng = random.Random(seed)
        self.Q: Dict[Tuple, List[float]] = {}

    def _ensure(self, s: Tuple) -> None:
        if s not in self.Q:
            self.Q[s] = [0.0 for _ in range(self.n_actions)]

    def act(self, s: Tuple) -> int:
        """Epsilon-greedy action for TRAINING."""
        self._ensure(s)
        if self.rng.random() < self.eps:
            return self.rng.randrange(self.n_actions)
        return self.act_greedy(s)

    def act_greedy(self, s: Tuple) -> int:
        """Greedy action for EVALUATION."""
        self._ensure(s)
        qvals = self.Q[s]
        return int(max(range(self.n_actions), key=lambda a: qvals[a]))

    def act_biased(self, s: Tuple, bias: List[float]) -> int:
        """Epsilon-greedy action with an external bias/prior (TRAINING)."""
        self._ensure(s)
        if len(bias) != self.n_actions:
            raise ValueError(f"bias length {len(bias)} != n_actions {self.n_actions}")
        if self.rng.random() < self.eps:
            return self.rng.randrange(self.n_actions)
        return self.act_greedy_biased(s, bias)

    def act_greedy_biased(self, s: Tuple, bias: List[float]) -> int:
        """Greedy action with an external bias/prior (EVALUATION)."""
        self._ensure(s)
        if len(bias) != self.n_actions:
            raise ValueError(f"bias length {len(bias)} != n_actions {self.n_actions}")
        qvals = self.Q[s]

        # Break ties deterministically via max() order, but bias should usually avoid ties.
        def score(a: int) -> float:
            return float(qvals[a]) + self.bias_beta * float(bias[a])

        return int(max(range(self.n_actions), key=score))

    def update(self, s: Tuple, a: int, r: float, s2: Tuple, done: bool) -> None:
        self._ensure(s)
        self._ensure(s2)
        qsa = self.Q[s][a]
        target = r if done else (r + self.gamma * max(self.Q[s2]))
        self.Q[s][a] = qsa + self.alpha * (target - qsa)

    def end_episode(self) -> None:
        self.eps = max(self.eps_end, self.eps * self.eps_decay)

    def set_eps(self, eps: float) -> None:
        self.eps = float(eps)
