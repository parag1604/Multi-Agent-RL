"""
Pieces shared by the versions: networks, a batch of PettingZoo environments
stepped together, and (for version2) the PPO machinery: rollout, GAE and the
PPO actor and critic updates.

Notation follows the lecture
(https://parag1604.github.io/notes/Reinforcement-Learning/Multi-Agent-Reinforcement-Learning):
agent i acts on its own observation o_i, and all agents receive the same
team reward r_t.
"""
import json
import os
import random

import numpy as np
import torch
import torch.nn as nn


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)          # small networks: one thread is fastest


def mlp(input_dim, output_dim, hidden_dim=64):
    return nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Tanh(),
                         nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
                         nn.Linear(hidden_dim, output_dim))


class EnvBatch:
    '''
    E copies of a PettingZoo parallel environment, stepped together. All MPE
    episodes have the same length (max_cycles), so the copies stay in sync:
    one call to rollout() plays exactly one episode in every copy.
    '''
    def __init__(self, make_env, n_envs, seed):
        self.envs = [make_env() for _ in range(n_envs)]
        self.agents = list(self.envs[0].possible_agents)
        self.seed = seed
        self.n_resets = 0

    def obs_dim(self, agent):
        return self.envs[0].observation_space(agent).shape[0]

    def n_actions(self, agent):
        return self.envs[0].action_space(agent).n

    def reset(self):
        obs = [env.reset(seed=self.seed + 10_000 * self.n_resets + e)[0]
               for e, env in enumerate(self.envs)]
        self.n_resets += 1
        return {a: np.stack([o[a] for o in obs]) for a in self.agents}

    def step(self, actions):
        '''actions: {agent: (E,) int array}. Returns obs, team reward (E,), done.'''
        obs, team_r, done = [], [], False
        for e, env in enumerate(self.envs):
            o, r, term, trunc, _ = env.step({a: int(actions[a][e]) for a in self.agents})
            obs.append(o)
            team_r.append(np.mean([r[a] for a in self.agents]))
            done = all(term.values()) or all(trunc.values())
        return ({a: np.stack([o[a] for o in obs]) for a in self.agents},
                np.array(team_r, dtype=np.float32), done)


def rollout(envs, policies, obs_filter=None):
    '''
    Play one episode in every copy with the current policies.
    obs_filter(agent, obs) can change what an agent sees (version2 uses it to
    switch the communication channel off).
    Returns a dict of tensors with a leading time axis (T, E, ...).
    '''
    obs = envs.reset()
    buf = {a: {'obs': [], 'act': [], 'logp': []} for a in envs.agents}
    buf['rew'] = []
    done = False
    while not done:
        if obs_filter is not None:
            obs = {a: obs_filter(a, o) for a, o in obs.items()}
        actions = {}
        with torch.no_grad():
            for a in envs.agents:
                o = torch.as_tensor(obs[a], dtype=torch.float32)
                dist = torch.distributions.Categorical(logits=policies[a](o))
                act = dist.sample()
                buf[a]['obs'].append(o)
                buf[a]['act'].append(act)
                buf[a]['logp'].append(dist.log_prob(act))
                actions[a] = act.numpy()
        obs, r, done = envs.step(actions)
        buf['rew'].append(r)
    out = {a: {k: torch.stack(v) for k, v in buf[a].items()} for a in envs.agents}
    out['rew'] = torch.as_tensor(np.stack(buf['rew']))
    return out


def gae(rew, values, gamma=0.99, lam=0.95):
    '''
    GAE(lambda) advantages for (T, E) rewards and values. The episode has a
    fixed length and the observations do not contain the time, so the last
    step is treated as terminal (V = 0 after it).
    '''
    T = rew.shape[0]
    adv = torch.zeros_like(rew)
    last = torch.zeros_like(rew[0])
    for t in reversed(range(T)):
        v_next = values[t + 1] if t + 1 < T else torch.zeros_like(rew[0])
        delta = rew[t] + gamma * v_next - values[t]
        last = delta + gamma * lam * last
        adv[t] = last
    return adv, adv + values          # advantages and lambda-return targets


def ppo_update(policy, opt, obs, act, logp_old, adv, epochs=10, n_minibatch=2,
               clip=0.2, ent_coef=0.01):
    '''The clipped PPO surrogate for one agent's decentralized actor.'''
    obs, act, logp_old = obs.flatten(0, 1), act.flatten(0, 1), logp_old.flatten(0, 1)
    adv = adv.flatten(0, 1)
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    n = obs.shape[0]
    for _ in range(epochs):
        for idx in torch.randperm(n).chunk(n_minibatch):
            dist = torch.distributions.Categorical(logits=policy(obs[idx]))
            ratio = (dist.log_prob(act[idx]) - logp_old[idx]).exp()
            surr = torch.min(ratio * adv[idx],
                             ratio.clamp(1 - clip, 1 + clip) * adv[idx])
            loss = -surr.mean() - ent_coef * dist.entropy().mean()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            opt.step()


def value_update(critic, opt, inputs, targets, epochs=10, n_minibatch=2):
    '''Regress V on the lambda-return targets.'''
    inputs, targets = inputs.flatten(0, 1), targets.flatten(0, 1)
    n = inputs.shape[0]
    for _ in range(epochs):
        for idx in torch.randperm(n).chunk(n_minibatch):
            loss = (critic(inputs[idx]).squeeze(-1) - targets[idx]).pow(2).mean()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(critic.parameters(), 0.5)
            opt.step()


def save_log(path, log):
    if path:
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'w') as f:
            json.dump(log, f)
