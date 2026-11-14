"""
version1 - CTDE: value decomposition networks (VDN) on simple_spread.

Same agents, same networks, same DQN machinery and hyperparameters as
version0; only the loss changes. Instead of a separate target per agent,
VDN (Sunehag et al., 2018) fits the TEAM value with a sum of the agents'
utilities,

    Q_tot(s, a) = sum_i Q_i(o_i, a_i),
    y = r_t + gamma * sum_i max_a' Q_i^target(o_i', a'),   loss = (Q_tot - y)^2.

The team reward is explained once, by the team. Each Q_i is only asked for
its share, and the gradient of the single error reaches every agent, so an
agent is no longer blamed or praised for what the others did (credit
assignment). Training is centralized: the loss needs all agents'
observations and actions at once. Execution is decentralized: because a sum
is maximized by maximizing each term (the IGM property), each agent simply
acts greedily on its own Q_i(o_i, .). The target also costs n separate
maximizations instead of a search over all joint actions.

Lecture: https://parag1604.github.io/notes/Reinforcement-Learning/Multi-Agent-Reinforcement-Learning
(section "CTDE with value factorization: VDN and QMIX").

Run:  python version1.py --neighbours 1     (partial observability)
      python version1.py                    (every agent sees everything)
"""
from version0 import main


def vdn_loss(q, q_next, r, done, gamma):
    '''VDN: one TD error for Q_tot = sum_i Q_i, with target sum_i max Q_i^target.'''
    q_tot, q_tot_next = sum(q), sum(q_next)
    return (q_tot - (r + gamma * (1 - done) * q_tot_next)).pow(2).mean()


if __name__ == '__main__':
    main(loss_fn=vdn_loss, default_log='runs/v1_vdn.json')
