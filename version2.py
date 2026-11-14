"""
version2 - DTDE with learned communication: simple_speaker_listener.

A task where talking is necessary. There are three coloured landmarks. The
listener can move but does not know which landmark is the goal; the speaker
knows the goal's colour but cannot move. Both get the same team reward,
minus the squared distance from the listener to the goal. The speaker's
only influence is a message m in R^3 that the listener sees in its
observation (o_listener = velocity, landmark positions, message).

Each agent still has its own actor and its own local critic, trained with
PPO as in version0. What is new is how the speaker learns what to say.
Three settings (--channel):

  dial  (default) Differentiable inter-agent learning (Foerster et al., 2016).
        The speaker's network outputs z = g(goal colour) in R^3, and during
        training the message goes through DIAL's discretize/regularize unit
            m = sigmoid(z + sigma * eps),   eps ~ N(0, I),
        and straight into the listener's network. The noise makes only
        messages with large |z| reliable, which pushes each entry towards a
        clean 0/1 bit. The listener's PPO loss is
        then a differentiable function of the speaker's parameters, so its
        gradient flows back THROUGH THE MESSAGE into the speaker: the speaker
        is told directly how a change in its message would have changed the
        listener's choice. (m is a deterministic function of the speaker's
        parameters given the noise, so this is just the policy gradient of the
        listener's policy pi_L(a | o, m(theta_speaker)) with respect to the
        speaker's parameters.) At execution the message is discretized to
        bits, m = 1[z > 0], as in DIAL. Training is centralized only in
        that one gradient crosses from the listener to the speaker.
  rial  The message is the speaker's discrete environment action, chosen by
        its own PPO policy from the team reward alone (RIAL-style). No
        gradient crosses between the agents.
  off   The channel is switched off: the listener's message entries are zero.
        Without the message the best the listener can do is to hedge between
        the landmarks.

Lecture: https://parag1604.github.io/notes/Reinforcement-Learning/Multi-Agent-Reinforcement-Learning
(section "Learning to communicate").

Run:  python version2.py                    (DIAL)
      python version2.py --channel rial
      python version2.py --channel off
"""
import argparse
import time

import numpy as np
import torch
import torch.nn as nn
from mpe2 import simple_speaker_listener_v4

from common import EnvBatch, gae, mlp, ppo_update, rollout, save_log, set_seed, value_update

MSG = slice(8, 11)       # where the message sits in the listener's observation
S, L = 'speaker_0', 'listener_0'


def make_env():
    return simple_speaker_listener_v4.parallel_env(max_cycles=25, continuous_actions=False)


def speak(speaker, o_s):
    '''z = g(goal colour); the colours (0.65 or 0.15 per channel) are rescaled to 0/1.'''
    return speaker((o_s - 0.15) / 0.5)


def dial_rollout(envs, speaker, listener, sigma, hard=False):
    '''
    One episode per environment copy. The speaker's environment action is
    ignored by us; the listener's message entries are overwritten with the
    DIAL message m (noisy and soft during training, hard bits if hard).
    '''
    obs = envs.reset()
    buf = {k: [] for k in ('o_s', 'eps', 'o_l', 'o_lm', 'act', 'logp', 'rew')}
    done = False
    while not done:
        o_s = torch.as_tensor(obs[S], dtype=torch.float32)
        o_l = torch.as_tensor(obs[L], dtype=torch.float32)
        eps = torch.randn(o_s.shape[0], 3)
        with torch.no_grad():
            z = speak(speaker, o_s)
            m = (z > 0).float() if hard else torch.sigmoid(z + sigma * eps)
            o_lm = o_l.clone()
            o_lm[:, MSG] = m
            dist = torch.distributions.Categorical(logits=listener(o_lm))
            act = dist.sample()
        for k, v in zip(buf, (o_s, eps, o_l, o_lm, act, dist.log_prob(act))):
            buf[k].append(v)
        obs, r, done = envs.step({S: np.zeros(len(act), dtype=int), L: act.numpy()})
        buf['rew'].append(torch.as_tensor(r))
    return {k: torch.stack(v) for k, v in buf.items()}


def dial_update(speaker, listener, opt, b, adv, sigma, epochs=10, n_minibatch=2,
                clip=0.2, ent_coef=0.01):
    '''PPO on the listener; the message is recomputed so gradients reach the speaker.'''
    flat = {k: v.flatten(0, 1) for k, v in b.items()}
    adv = adv.flatten(0, 1)
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    n = adv.shape[0]
    for _ in range(epochs):
        for idx in torch.randperm(n).chunk(n_minibatch):
            m = torch.sigmoid(speak(speaker, flat['o_s'][idx]) + sigma * flat['eps'][idx])
            o_l = torch.cat([flat['o_l'][idx][:, :MSG.start], m], -1)   # message has a gradient
            dist = torch.distributions.Categorical(logits=listener(o_l))
            ratio = (dist.log_prob(flat['act'][idx]) - flat['logp'][idx]).exp()
            surr = torch.min(ratio * adv[idx], ratio.clamp(1 - clip, 1 + clip) * adv[idx])
            loss = -surr.mean() - ent_coef * dist.entropy().mean()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(list(speaker.parameters()) + list(listener.parameters()), 0.5)
            opt.step()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--channel', choices=['dial', 'rial', 'off'], default='dial')
    p.add_argument('--iters', type=int, default=2000, help='PPO iterations')
    p.add_argument('--n-envs', type=int, default=64, help='episodes per iteration')
    p.add_argument('--lr', type=float, default=7e-4)
    p.add_argument('--sigma', type=float, default=2.0, help='DRU noise std (dial)')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--log', type=str, default=None)
    args = p.parse_args()
    args.log = args.log or f'runs/v2_{args.channel}.json'
    set_seed(args.seed)

    envs = EnvBatch(make_env, args.n_envs, args.seed)
    pi = {a: mlp(envs.obs_dim(a), envs.n_actions(a)) for a in envs.agents}
    V = {a: mlp(envs.obs_dim(a), 1) for a in envs.agents}
    opt_pi = {a: torch.optim.Adam(pi[a].parameters(), lr=args.lr) for a in envs.agents}
    opt_V = {a: torch.optim.Adam(V[a].parameters(), lr=args.lr) for a in envs.agents}
    if args.channel == 'dial':     # one optimizer: the listener's loss trains both networks
        opt_dial = torch.optim.Adam(list(pi[S].parameters()) + list(pi[L].parameters()),
                                    lr=args.lr)

    def obs_filter(agent, o):      # used by 'off' (and harmless for 'rial')
        if args.channel == 'off' and agent == L:
            o = o.copy()
            o[:, MSG] = 0.0
        return o

    log, steps, t0 = {'steps': [], 'returns': [], 'args': vars(args)}, 0, time.time()
    for it in range(args.iters):
        if args.channel == 'dial':
            b = dial_rollout(envs, pi[S], pi[L], args.sigma)
            o_v = b['o_lm']        # the listener's critic sees what it saw (message detached)
            with torch.no_grad():
                values = V[L](o_v).squeeze(-1)
            adv, target = gae(b['rew'], values)
            dial_update(pi[S], pi[L], opt_dial, b, adv, args.sigma)
            value_update(V[L], opt_V[L], o_v, target)
            rew = b['rew']
        else:
            batch = rollout(envs, pi, obs_filter)
            for a in envs.agents:
                o = batch[a]['obs']
                with torch.no_grad():
                    values = V[a](o).squeeze(-1)
                adv, target = gae(batch['rew'], values)
                ppo_update(pi[a], opt_pi[a], o, batch[a]['act'], batch[a]['logp'], adv)
                value_update(V[a], opt_V[a], o, target)
            rew = batch['rew']
        steps += rew.numel()
        ret = rew.sum(0).mean().item()
        log['steps'].append(steps)
        log['returns'].append(ret)
        if it % 100 == 0 or it == args.iters - 1:
            print(f'iter {it:5d}  steps {steps:8d}  team return {ret:8.2f}  '
                  f'({time.time() - t0:.0f}s)')

    # What does the speaker say for each goal colour?
    colours = torch.tensor([[0.65, 0.15, 0.15], [0.15, 0.65, 0.15], [0.15, 0.15, 0.65]])
    with torch.no_grad():
        if args.channel == 'dial':
            code = (speak(pi[S], colours) > 0).int().numpy()
            print('speaker: message bits for the red, green, blue goal:', code.tolist())
            log['speaker_code'] = code.tolist()
        else:
            probs = torch.softmax(pi[S](colours), -1).numpy()
            print('speaker: P(symbol | goal colour), rows red, green, blue:')
            print(np.round(probs, 2))
            log['speaker_probs'] = probs.tolist()
    if args.channel == 'dial':     # execution with discrete messages
        with torch.no_grad():
            r = [dial_rollout(envs, pi[S], pi[L], args.sigma, hard=True)['rew'].sum(0).mean().item()
                 for _ in range(20)]
        log['eval_hard_messages'] = float(np.mean(r))
        print(f'team return with discretized messages (320 episodes): {np.mean(r):.2f}')
    log['wall_time'] = time.time() - t0
    save_log(args.log, log)


if __name__ == '__main__':
    main()
