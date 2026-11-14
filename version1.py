"""
version1 - CTDE: a centralized critic (MAPPO-style) on simple_spread.

New compared to version0: ONE critic V(s) for the whole team, fed with the
global state s (here: the concatenation of all agents' observations) instead
of a local critic V_i(o_i) per agent. Since all agents share the team reward,
the same advantage estimate A_t (GAE on the centralized critic) is used by
every actor.

Execution is still decentralized: each actor pi_i(a_i | o_i) is exactly as
in version0 and sees only its own observation. The critic is needed only
during training, where the simulator gives us the global state. From agent
i's point of view, its local critic in version0 had to predict a return that
depends on what the other agents see and do; V(s) sees all of it, so its
targets are less noisy and less non-stationary.

Lecture: https://parag1604.github.io/notes/Reinforcement-Learning/Multi-Agent-Reinforcement-Learning
(section "CTDE with actor-critic: MADDPG, COMA and MAPPO").

Run:  python version1.py
      python version1.py --iters 500           (a quick check)
"""
import argparse
import time

import torch
from mpe2 import simple_spread_v3

from common import EnvBatch, gae, mlp, ppo_update, rollout, save_log, set_seed, value_update


def make_env():
    return simple_spread_v3.parallel_env(N=3, max_cycles=25, continuous_actions=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--iters', type=int, default=1000, help='PPO iterations')
    p.add_argument('--n-envs', type=int, default=64, help='episodes per iteration')
    p.add_argument('--lr', type=float, default=7e-4)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--log', type=str, default='runs/v1_mappo.json')
    args = p.parse_args()
    set_seed(args.seed)

    envs = EnvBatch(make_env, args.n_envs, args.seed)
    pi = {a: mlp(envs.obs_dim(a), envs.n_actions(a)) for a in envs.agents}
    state_dim = sum(envs.obs_dim(a) for a in envs.agents)
    V = mlp(state_dim, 1)                                            # centralized critic V(s)
    opt_pi = {a: torch.optim.Adam(pi[a].parameters(), lr=args.lr) for a in envs.agents}
    opt_V = torch.optim.Adam(V.parameters(), lr=args.lr)

    log, steps, t0 = {'steps': [], 'returns': [], 'args': vars(args)}, 0, time.time()
    for it in range(args.iters):
        batch = rollout(envs, pi)
        steps += batch['rew'].numel()
        with torch.no_grad():
            values = V(batch['state']).squeeze(-1)
        adv, target = gae(batch['rew'], values)     # one advantage for the whole team
        for a in envs.agents:                       # decentralized actors
            ppo_update(pi[a], opt_pi[a], batch[a]['obs'], batch[a]['act'], batch[a]['logp'], adv)
        value_update(V, opt_V, batch['state'], target)
        ret = batch['rew'].sum(0).mean().item()     # team return of this batch of episodes
        log['steps'].append(steps)
        log['returns'].append(ret)
        if it % 100 == 0 or it == args.iters - 1:
            print(f'iter {it:5d}  steps {steps:8d}  team return {ret:8.2f}  '
                  f'({time.time() - t0:.0f}s)')
    log['wall_time'] = time.time() - t0
    save_log(args.log, log)


if __name__ == '__main__':
    main()
