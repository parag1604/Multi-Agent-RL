"""
version0 - DTDE: independent Q-learning (IQL) on simple_spread.

Three agents must spread out to cover three landmarks. Every agent gets the
same team reward r_t (here the mean of the environment's per-agent rewards:
minus the sum over landmarks of the distance to the closest agent, plus a
penalty for collisions). With --neighbours 1 the task is partially
observable: each agent sees its own position and velocity, but only its
nearest other agent and its nearest landmark.

Decentralized training and execution: every agent i has its own deep
Q-network Q_i(o_i, a_i) and treats the other agents as part of the
environment. Each one is trained with the ordinary DQN target on the TEAM
reward,

    y_i = r_t + gamma * max_a' Q_i^target(o_i', a'),     loss = sum_i (Q_i(o_i, a_i) - y_i)^2,

so each Q_i has to explain the whole team reward from its own observation,
although most of that reward depends on what the other agents see and do.

The DQN parts: a replay buffer of joint transitions, epsilon-greedy
exploration (epsilon from 1 to 0.05 over the first 300k steps), a target
network copied every 500 updates, one gradient step per environment step
of the 16 parallel copies.

Lecture: https://parag1604.github.io/notes/Reinforcement-Learning/Multi-Agent-Reinforcement-Learning
(section "Decentralized training: independent learners").

Run:  python version0.py --neighbours 1     (partial observability)
      python version0.py                    (every agent sees everything)
"""
import argparse
import time

import numpy as np
import torch
from mpe2 import simple_spread_v3

from common import EnvBatch, mlp, save_log, set_seed


def td_loss(q, q_next, r, done, gamma):
    '''
    IQL: q[i] = Q_i(o_i, a_i) and q_next[i] = max_a' Q_i^target(o_i', a') for
    each agent i; every agent regresses on its own target built from the
    team reward.
    '''
    return sum((qi - (r + gamma * (1 - done) * qni)).pow(2).mean()
               for qi, qni in zip(q, q_next))


def make_env(n_agents, neighbours):
    '''simple_spread; with neighbours = k each agent sees only the k nearest agents and landmarks.'''
    return lambda: simple_spread_v3.parallel_env(
        N=n_agents, max_cycles=25, continuous_actions=False,
        num_agent_neighbors=neighbours, num_landmark_neighbors=neighbours)


def main(loss_fn=td_loss, default_log='runs/v0_iql.json'):
    p = argparse.ArgumentParser()
    p.add_argument('--steps', type=int, default=1_000_000, help='environment steps')
    p.add_argument('--n-agents', type=int, default=3)
    p.add_argument('--neighbours', type=int, default=None,
                   help='observe only the k nearest agents and landmarks')
    p.add_argument('--n-envs', type=int, default=16)
    p.add_argument('--lr', type=float, default=5e-4)
    p.add_argument('--gamma', type=float, default=0.99)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--buffer-size', type=int, default=100_000)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--log', type=str, default=None)
    args = p.parse_args()
    args.log = args.log or default_log.replace('.json', f'_k{args.neighbours or "all"}.json')
    set_seed(args.seed)

    envs = EnvBatch(make_env(args.n_agents, args.neighbours), args.n_envs, args.seed)
    A, E, cap = envs.agents, args.n_envs, args.buffer_size
    Q = {a: mlp(envs.obs_dim(a), envs.n_actions(a)) for a in A}
    Q_target = {a: mlp(envs.obs_dim(a), envs.n_actions(a)) for a in A}
    for a in A:
        Q_target[a].load_state_dict(Q[a].state_dict())
    params = [w for a in A for w in Q[a].parameters()]
    opt = torch.optim.Adam(params, lr=args.lr)

    # replay buffer of joint transitions (o_i, a_i, o_i' for every agent; one team reward)
    buf = {'o': {a: np.zeros((cap, envs.obs_dim(a)), np.float32) for a in A},
           'o2': {a: np.zeros((cap, envs.obs_dim(a)), np.float32) for a in A},
           'a': {a: np.zeros(cap, np.int64) for a in A},
           'r': np.zeros(cap, np.float32), 'd': np.zeros(cap, np.float32)}
    ptr = size = steps = updates = 0
    log, t0 = {'steps': [], 'returns': [], 'args': vars(args)}, time.time()

    while steps < args.steps:
        obs, done, ret = envs.reset(), False, np.zeros(E)
        while not done:
            eps = max(0.05, 1 - steps / 300_000)
            act = {}
            with torch.no_grad():
                for a in A:
                    greedy = Q[a](torch.as_tensor(obs[a])).argmax(-1).numpy()
                    random = np.random.randint(envs.n_actions(a), size=E)
                    act[a] = np.where(np.random.rand(E) < eps, random, greedy)
            obs2, r, done = envs.step(act)
            idx = (ptr + np.arange(E)) % cap
            for a in A:
                buf['o'][a][idx], buf['o2'][a][idx], buf['a'][a][idx] = obs[a], obs2[a], act[a]
            buf['r'][idx], buf['d'][idx] = r, float(done)    # fixed horizon: last step terminal
            ptr, size, steps = (ptr + E) % cap, min(cap, size + E), steps + E
            ret += r
            obs = obs2

            if size >= 5000:                                  # one gradient step
                j = np.random.randint(size, size=args.batch_size)
                q, q_next = [], []
                for a in A:
                    o, o2 = torch.as_tensor(buf['o'][a][j]), torch.as_tensor(buf['o2'][a][j])
                    q.append(Q[a](o).gather(1, torch.as_tensor(buf['a'][a][j])[:, None]).squeeze(1))
                    with torch.no_grad():
                        q_next.append(Q_target[a](o2).max(1).values)
                loss = loss_fn(q, q_next, torch.as_tensor(buf['r'][j]),
                               torch.as_tensor(buf['d'][j]), args.gamma)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 10)
                opt.step()
                updates += 1
                if updates % 500 == 0:
                    for a in A:
                        Q_target[a].load_state_dict(Q[a].state_dict())

        log['steps'].append(steps)
        log['returns'].append(float(ret.mean()))      # team return, epsilon-greedy episodes
        if len(log['steps']) % 250 == 0:
            print(f'steps {steps:8d}  team return {np.mean(log["returns"][-250:]):8.2f}  '
                  f'eps {eps:.2f}  ({time.time() - t0:.0f}s)')
    log['wall_time'] = time.time() - t0
    save_log(args.log, log)


if __name__ == '__main__':
    main()
