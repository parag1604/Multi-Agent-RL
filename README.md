# Multi-Agent-RL

A short tutorial on cooperative multi-agent reinforcement learning with the simple particle environments (MPE) of PettingZoo. One small example per setting from the lecture: independent Q-learners (DTDE, version0), centralized training with decentralized execution through value factorization (VDN, CTDE, version1), and agents that learn to communicate (version2). version3 is a detour to the game-theory part: regret matching in rock–paper–scissors.

---

## Theory

Please note that the theory corresponding to the codes is hosted on this [page](https://parag1604.github.io/notes/Reinforcement-Learning/Multi-Agent-Reinforcement-Learning). The code follows its notation: agent $i$ acts on its own observation $o_i$, with a utility $Q_i(o_i, a_i)$ (versions 0 and 1) or a policy $\pi_i(a_i \mid o_i)$ (version2), and all agents share the team reward $r_t$. The single-agent algorithms underneath are DQN, from the [Deep-Q-Networks](https://github.com/parag1604/Deep-Q-Networks) repository, and PPO, from the [Policy-Gradients](https://github.com/parag1604/Policy-Gradients) repository.

## Python Environment Setup

Python >= 3.10, then:

```bash
pip install -r requirements.txt
```

The MPE tasks used to live in `pettingzoo.mpe`; they are now the separate package `mpe2` (Farama), with the same parallel API (`simple_spread_v3`, `simple_speaker_listener_v4`).

## Running

`common.py` holds the shared pieces: a batch of environment copies stepped together and, for version2, the PPO rollout, GAE($\lambda$) and update. All episodes have 25 steps, and the last step is treated as terminal. Every script has `--help`, saves its learning curve to `runs/`, and prints progress; `python plot_results.py` draws the figures in `static/` and prints the numbers quoted below. The runs were on the CPU of a laptop, several at once, and took 11 to 20 minutes each (rough wall-clock times).

- **versions 0 and 1 (DQN):** 16 environment copies, 1M environment steps, a replay buffer of 100k joint transitions, one gradient step (batch 256, Adam, learning rate $5\times10^{-4}$) per step of the 16 copies, $\gamma = 0.99$, $\varepsilon$-greedy with $\varepsilon$ from 1 to 0.05 over the first 300k steps, target networks copied every 500 updates, two hidden layers of 64 tanh units. The two versions share everything except the loss.
- **version2 (PPO):** each iteration plays one episode in each of 64 copies (1600 environment steps), then runs 10 epochs of PPO with 2 minibatches (Adam, learning rate $7\times10^{-4}$, $\gamma = 0.99$, $\lambda = 0.95$, clip 0.2, entropy bonus 0.01, same network sizes).

The team return is the sum of the 25 team rewards of an episode; the curves show the mean over the training episodes of each iteration (the policies as they are being trained, so including the $\varepsilon = 0.05$ exploration in versions 0 and 1), with a moving average over 5% of the run. "Final" is the mean over the last 10% of the run.

All results at a glance:

| Task | Version | Environment steps | Final team return |
|---|---|---|---|
| simple_spread (3 agents), each sees its nearest agent and landmark | v0: independent Q-learning (DTDE) | 1M | −24.8 |
| | v1: VDN (CTDE) | 1M | −14.9 |
| simple_spread (3 agents), each sees everything | v0: independent Q-learning (DTDE) | 1M | −22.8 |
| | v1: VDN (CTDE) | 1M | −20.0 |
| simple_speaker_listener | v2: DIAL (gradient through the message) | 3.2M | −7.5 (−7.6 with discrete messages) |
| | v2: RIAL-style (message as an action) | 3.2M | −15.8 |
| | v2: channel switched off | 3.2M | −15.8 |
| rock–paper–scissors | v3: regret matching | 10 000 rounds | exploitability 0.020 |

For scale: the untrained, near-uniform policies score about −27 on simple_spread and about −42 on simple_speaker_listener.

### version0: independent Q-learning (DTDE) on simple_spread

```bash
python version0.py --neighbours 1     # each agent sees only its nearest agent and landmark
python version0.py                    # each agent sees everything
```

Three agents must spread out to cover three landmarks. Each agent's environment reward is minus the sum, over landmarks, of the distance to the closest agent, plus a penalty for collisions; the team reward $r_t$ is their mean, and every agent is trained on it. With `--neighbours 1` the task is partially observable: an agent sees its own position and velocity, but only its nearest other agent and its nearest landmark.

Every agent $i$ has its own Q-network $Q_i(o_i, a_i)$ and runs DQN as if the other agents were part of the environment, each with its own target on the team reward:

$$
\mathcal{L} = \sum_i \Big( Q_i(o_i, a_i) - r_t - \gamma \max_{a'} \bar Q_i(o_i', a') \Big)^2 ,
$$

where $\bar Q_i$ is agent $i$'s target network. Nothing is shared between the agents, not even parameters.

### version1: value decomposition networks (VDN, CTDE) on simple_spread

```bash
python version1.py --neighbours 1
python version1.py
```

Same networks, same DQN machinery, same hyperparameters; only the loss changes. VDN fits the team value with the sum of the agents' utilities, and there is one TD error for the team:

$$
Q_{\mathrm{tot}}(s, \mathbf{a}) = \sum_i Q_i(o_i, a_i), \qquad
\mathcal{L} = \Big( \sum_i Q_i(o_i, a_i) - r_t - \gamma \sum_i \max_{a'} \bar Q_i(o_i', a') \Big)^2 .
$$

Training is centralized, since the loss needs every agent's observation and action at once. Execution is decentralized: a sum is maximized by maximizing each term, so each agent acts greedily on its own $Q_i(o_i, \cdot)$.

![Independent Q-learning vs VDN on simple_spread](static/spread_dtde_vs_ctde.png "simple_spread")

Result: with partial observability VDN ends at −14.9 against −24.8 for the independent learners, and it is ahead for the whole run; when every agent sees everything the gap is smaller (−20.0 vs −22.8). The reason is credit assignment. An independent learner must explain the whole team reward from its own view, although most of that reward depends on agents it cannot see and on policies that keep changing; its targets are noisy and non-stationary, and as exploration decays its estimates get worse before they get better (the dip around 0.2–0.4M steps). VDN explains the team reward once, as a sum, so each $Q_i$ only has to learn its own share. The less each agent sees, the more this matters. (VDN also does better here with the small local view than with the full observation, which has more inputs to learn from in the same budget.)

### version2: learning to communicate on simple_speaker_listener

```bash
python version2.py                    # DIAL
python version2.py --channel rial     # message as the speaker's action
python version2.py --channel off      # channel switched off
```

A task where talking is necessary. Of three coloured landmarks, the listener must reach the goal, but only the speaker knows which one it is, and the speaker cannot move. Both get the team reward, minus the squared distance from the listener to the goal. The speaker's only influence is a 3-dimensional message that the listener sees in its observation. Each agent is an independent PPO learner with its own actor and local critic; the settings differ in how the speaker learns what to say:

- **DIAL** (Foerster et al., 2016). The speaker's network outputs $z = g(\text{goal colour}) \in \mathbb{R}^3$, and during training the message is DIAL's noisy "discretize/regularize unit", $m = \sigma(z + \sigma_{\text{DRU}}\,\epsilon)$ with $\epsilon \sim \mathcal{N}(0, I)$ and $\sigma_{\text{DRU}} = 2$, passed straight into the listener's network. The listener's PPO loss is then differentiable in the speaker's parameters, and its gradient flows **through the message** into the speaker. At execution the message is three bits, $m = \mathbb{1}[z > 0]$.
- **RIAL-style.** The message is one of three symbols, chosen by the speaker's own PPO policy from the team reward alone; no gradient crosses between the agents.
- **Off.** The listener's message entries are set to zero; the listener can only hedge between the landmarks.

![Communication on vs off on simple_speaker_listener](static/speaker_listener_comm.png "simple_speaker_listener")

Result: with DIAL the speaker settles on a different code for each goal colour (bits 011 for red, 001 for green, 110 for blue), and the team return rises from the no-communication level of about −16 to −7.5 (−7.6 with discrete bits at execution, over 320 episodes). The protocol takes time to appear: DIAL stays at the no-communication level for the first 1.5M steps, then climbs. With the message as an ordinary action (RIAL-style) no protocol emerged in 3.2M steps: the speaker's symbol probabilities were only weakly tied to the goal colour (0.66, 0.58 and 0.49 on the "diagonal") and the listener gained nothing from the channel, ending exactly where the switched-off channel ends (−15.8). This is the gap DIAL was designed to close: through the message, the speaker gets a direct gradient on how its message changed the listener's choice, while a RIAL speaker must discover the same thing from a team reward that, as long as the listener ignores it, does not depend on what it says.

Two things that did not work first and were changed: a Gumbel-softmax relaxation of a single one-hot symbol made the speaker collapse to the same symbol for every colour (once the relaxed symbol saturates, the gradient through it vanishes), which the independent noisy bits of DIAL's unit avoid; and with 16 episodes per iteration even a listener given the correct goal for free did not learn to use it in 0.8M steps, hence 64 episodes per iteration and 10 PPO epochs.

### version3: regret matching in rock–paper–scissors

```bash
python version3.py
```

A detour to the lecture's game-theory part. Two players repeatedly play rock–paper–scissors. Each keeps, for every action $a$, its cumulative regret $\mathrm{Reg}_t(a) = \sum_{k \le t} \big(u_k(a) - u_k(a_k)\big)$, how much better always playing $a$ would have done, and plays each action with probability proportional to its positive regret (Hart & Mas-Colell, 2000):

$$
\pi_{t+1}(a) = \frac{\max(\mathrm{Reg}_t(a), 0)}{\sum_b \max(\mathrm{Reg}_t(b), 0)} .
$$

Both players start off equilibrium, and the run uses expected payoffs, so it is deterministic. The current strategies keep cycling between pure actions, but the average strategies converge to the mixed Nash equilibrium $(1/3, 1/3, 1/3)$: after $10^4$ rounds player 1's average is $(0.337, 0.336, 0.327)$, and the exploitability of the averages (what both players together could gain by best-responding, 0 exactly at a Nash equilibrium) falls from 0.21 after 100 rounds to 0.064 after 1000 and 0.020 after $10^4$, roughly like $1/\sqrt{t}$. The lecture shows the same behaviour for fictitious play.

![Regret matching in rock-paper-scissors](static/rps_regret_matching.png "Regret matching")

## References

- J. Foerster, Y. Assael, N. de Freitas and S. Whiteson. *Learning to Communicate with Deep Multi-Agent Reinforcement Learning.* NeurIPS 2016. [arXiv:1605.06676](https://arxiv.org/abs/1605.06676)
- P. Sunehag, G. Lever, A. Gruslys, W. M. Czarnecki, V. Zambaldi, M. Jaderberg, M. Lanctot, N. Sonnerat, J. Z. Leibo, K. Tuyls and T. Graepel. *Value-Decomposition Networks For Cooperative Multi-Agent Learning.* AAMAS 2018. [arXiv:1706.05296](https://arxiv.org/abs/1706.05296)
- R. Lowe, Y. Wu, A. Tamar, J. Harb, P. Abbeel and I. Mordatch. *Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments.* NeurIPS 2017 (the particle environments). [arXiv:1706.02275](https://arxiv.org/abs/1706.02275)
- S. Hart and A. Mas-Colell. *A Simple Adaptive Procedure Leading to Correlated Equilibrium.* Econometrica 68(5), 2000. [doi:10.1111/1468-0262.00153](https://doi.org/10.1111/1468-0262.00153)
