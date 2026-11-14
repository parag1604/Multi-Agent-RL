"""
Draw the figures in static/ from the logs in runs/ and print the numbers
quoted in the README.

Each point of a learning curve is the mean team return of the training
episodes of one iteration (64 episodes of one PPO iteration in version2, 16
epsilon-greedy episodes in version0/1), i.e. of the policies being trained;
the thick line is a moving average over 5% of the iterations. "Final" is the mean
over the last 10% of the iterations.

Run:  python plot_results.py
"""
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

COLORS = ['#0072B2', '#D55E00', '#009E73']


def load(name):
    with open(f'runs/{name}.json') as f:
        return json.load(f)


def smooth(x, w):
    x = np.asarray(x)
    c = np.cumsum(np.insert(x, 0, 0.0))
    out = np.empty(len(x))
    for i in range(len(x)):
        lo = max(0, i - w + 1)
        out[i] = (c[i + 1] - c[lo]) / (i + 1 - lo)
    return out


def final(log):
    r = np.asarray(log['returns'])
    return r[-max(1, len(r) // 10):].mean()


def curves(runs, labels, title, path, ax=None):
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(6.4, 4))
    for k, (log, lab) in enumerate(zip(runs, labels)):
        s, r = np.asarray(log['steps']) / 1e6, np.asarray(log['returns'])
        w = max(1, len(r) // 20)                 # moving average over 5% of the run
        ax.plot(s, r, color=COLORS[k], alpha=0.15, lw=0.8)
        ax.plot(s, smooth(r, w), color=COLORS[k], lw=2, label=f'{lab} (final {final(log):.1f})')
        extra = (f', discretized messages {log["eval_hard_messages"]:.2f}'
                 if 'eval_hard_messages' in log else '')
        print(f'{title} | {lab}: final team return {final(log):.2f}{extra}, '
              f'{log["wall_time"] / 60:.1f} min')
    ax.set_xlabel('environment steps (millions)')
    ax.set_ylabel('team return per episode')
    ax.set_title(title)
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(alpha=0.3)
    if own:
        ax.figure.tight_layout()
        ax.figure.savefig(path, dpi=150)


def spread(path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    labels = ['v0: independent Q-learning (DTDE)', 'v1: VDN (CTDE)']
    curves([load('v0_iql_k1'), load('v1_vdn_k1')], labels,
           'simple_spread, each agent sees 1 agent + 1 landmark', None, axes[0])
    curves([load('v0_iql_kall'), load('v1_vdn_kall')], labels,
           'simple_spread, each agent sees everything', None, axes[1])
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def rps(path):
    log = load('v3_rps')
    cur, avg, expl = (np.asarray(log[k]) for k in ('current1', 'avg1', 'expl'))
    t = np.arange(1, len(expl) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    ax = axes[0]
    for k, name in enumerate(['rock', 'paper', 'scissors']):
        ax.plot(t[:300], cur[:300, k], color=COLORS[k], alpha=0.35, lw=1)
        ax.plot(t[:300], avg[:300, k], color=COLORS[k], lw=2, label=name)
    ax.axhline(1 / 3, color='gray', ls='--', lw=1)
    ax.set_xlabel('round')
    ax.set_ylabel('probability (player 1)')
    ax.set_title('current (thin) vs average (thick) strategy')
    ax.legend(fontsize=8)
    ax = axes[1]
    ax.loglog(t, expl, color=COLORS[0], lw=2, label='regret matching')
    ax.loglog(t, expl[9] * np.sqrt(10 / t), color='gray', ls='--', lw=1,
              label=r'$\propto 1/\sqrt{t}$')
    ax.set_xlabel('round')
    ax.set_ylabel('exploitability of the averages')
    ax.set_title('distance to the Nash equilibrium')
    ax.legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f'RPS: average strategy {np.round(avg[-1], 3)}, exploitability '
          f'{expl[99]:.3f} (100 rounds), {expl[999]:.3f} (1000), {expl[-1]:.4f} ({len(t)})')


if __name__ == '__main__':
    spread('static/spread_dtde_vs_ctde.png')
    curves([load('v2_dial'), load('v2_rial'), load('v2_off')],
           ['v2: DIAL (gradient through the message)', 'v2: RIAL-style (message as an action)',
            'v2: channel switched off'],
           'simple_speaker_listener', 'static/speaker_listener_comm.png')
    rps('static/rps_regret_matching.png')
