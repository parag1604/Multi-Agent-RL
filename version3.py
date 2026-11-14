"""
version3 - a normal-form game: regret matching in rock-paper-scissors.

A detour from the cooperative tasks to the lecture's game-theory part. Two
players repeatedly play the zero-sum game with payoff matrix R (row player
gets R[a1, a2], column player gets -R[a1, a2]). Each player keeps, for every
action a, its cumulative regret

    Reg_t(a) = sum_{k <= t} ( u_k(a) - u_k(a_k) ),

how much better it would have done by always playing a against what the
opponent actually played. Regret matching (Hart & Mas-Colell, 2000) plays
each action with probability proportional to its positive regret,

    pi_{t+1}(a) = max(Reg_t(a), 0) / sum_b max(Reg_t(b), 0)    (uniform if all <= 0).

Here both players use expected utilities (they update with the full vector
u_k = R pi_k^{opp}), which makes the run deterministic. In a two-player
zero-sum game the AVERAGE strategies converge to a Nash equilibrium (for RPS:
(1/3, 1/3, 1/3)), while the current strategies keep cycling. We measure
the distance to equilibrium by the exploitability of the average strategies,

    expl = max_a (R pibar_2)_a - min_b (pibar_1^T R)_b,

the sum of what each player could gain by best-responding to the other
(0 exactly at a Nash equilibrium).

Lecture: https://parag1604.github.io/notes/Reinforcement-Learning/Multi-Agent-Reinforcement-Learning
(sections "Two-player zero-sum games and the minimax theorem" and
"Learning an equilibrium by playing: fictitious play").

Run:  python version3.py
"""
import json
import os

import numpy as np

R = np.array([[0, -1, 1],          # rows: player 1 plays rock, paper, scissors
              [1, 0, -1],
              [-1, 1, 0]], dtype=float)


def regret_matching_policy(regret):
    pos = np.maximum(regret, 0.0)
    return pos / pos.sum() if pos.sum() > 0 else np.full(len(regret), 1 / len(regret))


def exploitability(p1, p2):
    return (R @ p2).max() - (p1 @ R).min()


def main(T=10_000):
    reg1, reg2 = np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])   # start off-equilibrium
    sum1, sum2 = np.zeros(3), np.zeros(3)
    log = {'current1': [], 'avg1': [], 'expl': []}
    for t in range(1, T + 1):
        pi1, pi2 = regret_matching_policy(reg1), regret_matching_policy(reg2)
        u1 = R @ pi2                 # player 1's expected payoff of each action
        u2 = -(pi1 @ R)              # player 2's expected payoff of each action
        reg1 += u1 - pi1 @ u1
        reg2 += u2 - pi2 @ u2
        sum1 += pi1
        sum2 += pi2
        avg1, avg2 = sum1 / t, sum2 / t
        log['current1'].append(pi1.tolist())
        log['avg1'].append(avg1.tolist())
        log['expl'].append(exploitability(avg1, avg2))
    print('average strategy of player 1:', np.round(avg1, 4))
    print('average strategy of player 2:', np.round(avg2, 4))
    for t in (10, 100, 1000, T):
        print(f'exploitability after {t:6d} rounds: {log["expl"][t - 1]:.4f}')
    os.makedirs('runs', exist_ok=True)
    with open('runs/v3_rps.json', 'w') as f:
        json.dump(log, f)


if __name__ == '__main__':
    main()
