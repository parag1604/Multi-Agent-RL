"""
version0 - DTDE: independent PPO learners on simple_spread.

Three agents must spread out to cover three landmarks. Every agent gets the
same team reward r_t (here the mean of the environment's per-agent rewards:
minus the sum over landmarks of the distance to the closest agent, plus a
penalty for collisions).

Decentralized training and execution: agent i has its OWN actor
pi_i(a_i | o_i) and its OWN critic V_i(o_i), both fed only with its own
observation o_i, and is trained with ordinary single-agent PPO as if the
other agents were part of the environment. Nothing is shared between agents,
not even parameters.

Lecture: https://parag1604.github.io/notes/Reinforcement-Learning/Multi-Agent-Reinforcement-Learning
(section "Decentralized training: independent learners").

Run:  python version0.py                       (3 agents, 1000 PPO iterations)
      python version0.py --iters 500           (a quick check)
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
    p.add_argument('--log', type=str, default='runs/v0_ippo.json')
    args = p.parse_args()
    set_seed(args.seed)

    envs = EnvBatch(make_env, args.n_envs, args.seed)
    pi = {a: mlp(envs.obs_dim(a), envs.n_actions(a)) for a in envs.agents}
    V = {a: mlp(envs.obs_dim(a), 1) for a in envs.agents}          # local critics V_i(o_i)
    opt_pi = {a: torch.optim.Adam(pi[a].parameters(), lr=args.lr) for a in envs.agents}
    opt_V = {a: torch.optim.Adam(V[a].parameters(), lr=args.lr) for a in envs.agents}

    log, steps, t0 = {'steps': [], 'returns': [], 'args': vars(args)}, 0, time.time()
    for it in range(args.iters):
        batch = rollout(envs, pi)
        steps += batch['rew'].numel()
        for a in envs.agents:                       # each agent learns on its own
            o = batch[a]['obs']
            with torch.no_grad():
                values = V[a](o).squeeze(-1)
            adv, target = gae(batch['rew'], values)
            ppo_update(pi[a], opt_pi[a], o, batch[a]['act'], batch[a]['logp'], adv)
            value_update(V[a], opt_V[a], o, target)
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
