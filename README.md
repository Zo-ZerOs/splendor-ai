# Splendor AI

A complete implementation of the 4-player base-game Splendor, with a headless
rules engine, a reinforcement-learning environment, PPO self-play training,
baseline agents, a terminal interface, and a read-only Pygame viewer.

## Installation

Requires Python 3.11+ (developed and tested on 3.12).

```bash
pip install -r requirements.txt
```

Dependencies: `torch`, `gymnasium`, `pygame`, `numpy`, `PyYAML`, `pytest`,
`tensorboard` (optional -- see [Known limitations](#known-limitations)).

## Expected data locations

All game content lives in three folders that this project only ever reads,
never writes:

```
cards/cards.json     + cards/tier1/*.png, cards/tier2/*.png, cards/tier3/*.png
coins/coins.json      + coins/images/*.png
nobles/nobles.json    + nobles/images/*.png
```

### Detected schemas

**Card** (`cards/cards.json`, a JSON array):

```json
{
  "id": "t1_001",
  "tier": 1,
  "bonus": "black",
  "points": 0,
  "cost": {"white": 0, "blue": 0, "green": 1, "red": 3, "black": 1},
  "image": "tier1/t1_001.png"
}
```

**Noble** (`nobles/nobles.json`):

```json
{
  "id": "noble_001",
  "points": 3,
  "requirements": {"white": 3, "blue": 3, "green": 0, "red": 0, "black": 3},
  "image": "images/noble_001.png"
}
```

**Coin** (`coins/coins.json`):

```json
{"color": "white", "is_joker": false, "image": "images/white.png"}
```

Data as loaded: 90 cards (40/30/20 across tiers 1/2/3), 10 nobles, 6 coin
colors (5 normal + gold). Validated clean -- no missing fields, invalid
tiers/colors, negative values, duplicate IDs, or broken image paths.

### How paths are resolved

Every `image` field is resolved **relative to its own JSON file's directory**
(e.g. `cards/cards.json`'s `"tier1/t1_001.png"` -> `cards/tier1/t1_001.png`).
This is handled once in `splendor_ai/data/loader.py`; nothing else touches the
raw JSON.

### Data validation

```bash
python -m splendor_ai.data.validator
```

Read-only. Reports per-tier card counts, noble/coin counts, duplicate IDs,
invalid tiers/colors, missing fields, negative values, broken image paths,
duplicate image references, unreferenced image files, and unknown JSON
fields. Exits non-zero only on a fatal (gameplay-breaking) issue; missing
images are reported as warnings, never fatal (the UI falls back to
placeholders).

## Running tests

```bash
python -m pytest
```

~175 tests covering data validation, every rule (setup, token picking, buying
with bonus discounts and colored-tokens-before-gold, reserving, the
token-return and noble-choice sub-decisions, final-round/tie-break logic),
illegal-action safety, serialization, the RL action-space encode/decode
round-trip, hidden-information protection in observations, action masking,
self-play credit assignment (the trickiest correctness point in the whole
project -- see [Reward design](#reward-design)), checkpoint save/load, a full
PPO training smoke run, and 1000+-game randomized stress tests of the engine
and both baseline agents.

## Architecture

```
splendor_ai/
├── data/      loader.py, validator.py, schemas.py       -- read-only data layer
├── game/      state.py, player.py, actions.py, rules.py, engine.py, simulate.py
├── env/       splendor_env.py, observation.py, action_space.py, action_mask.py
├── agents/    random_agent.py, heuristic_agent.py, rl_agent.py, network.py, base.py
├── training/  train.py, self_play.py, evaluate.py, select_difficulty.py, checkpoint.py
├── ui/        pygame_view.py, layout.py, asset_loader.py
└── cli/       human_input.py, play.py, render.py
```

`splendor_ai/game/` is the single source of truth for legality and state
transitions. It has no dependency on Gymnasium or Pygame -- the RL
environment and both UIs are built on top of it, not the other way around.

## How to play (terminal + read-only viewer)

```bash
python -m splendor_ai.cli.play                              # 3 random-agent opponents
python -m splendor_ai.cli.play --difficulty easy              # all 3 AI seats = Easy
python -m splendor_ai.cli.play --difficulty medium           # all 3 AI seats = Medium
python -m splendor_ai.cli.play --difficulty expert           # all 3 AI seats = Expert
python -m splendor_ai.cli.play --difficulty medium --mixed 2=expert   # mixed: seat 3 is Expert, others Medium
python -m splendor_ai.cli.play --human-seat 2 --seed 123
python -m splendor_ai.cli.play --no-render                   # terminal only, no Pygame window
python -m splendor_ai.cli.play --checkpoint checkpoints/expert/best.pt   # a specific checkpoint file
```

`--difficulty {easy,medium,expert}` resolves to `checkpoints/<difficulty>/best.pt`
(produced by `select_difficulty.py`, see below) and applies to every AI seat
by default; `--mixed SEAT=DIFFICULTY` (repeatable, 0-indexed seat) overrides
individual seats for a mixed-difficulty game. `--checkpoint` bypasses both and
points every AI seat at one specific file. If nothing is given, AI seats fall
back to `RandomAgent`.

Tiers are ordered weakest to strongest and shift down as training improves:
"expert" is always the current best checkpoint from the most recent training
run, and when a stronger one is selected, today's expert becomes tomorrow's
medium (and today's medium becomes tomorrow's easy) rather than being
overwritten by a weaker checkpoint.

### Terminal syntax

```
P white blue green     take 3 different-colored tokens
P red red               take 2 tokens of the same color (bank must have had >=4)
P white [<color>]       take 1 of each remaining color (only legal when the
                         bank has fewer than 3 normal colors left)
B T1-1                  buy the visible card in tier-1 slot 1
B R2                    buy your 2nd reserved card
R T2-3                  reserve the visible card in tier-2 slot 3
R D3                    reserve the face-down top card of tier 3
RETURN white gold       return tokens (only when prompted, over the 10-token limit)
NOBLE 2   (or N 2)      pick noble #2 from an offered list (only when prompted)
PASS                    only ever offered when no other action is legal
help / state / legal / history / quit
```

Commands are case-insensitive. Board *positions* are used (not card IDs)
because a visible slot holds a different card every game.

**Single action system, single legality check.** Every command parses into
exactly the same `PickTokensAction`/`BuyCardAction`/.../`PassAction`
dataclasses the AI's Discrete(149) action indices decode to
(`splendor_ai/cli/human_input.py`), and every one of them -- human or AI --
is validated and applied through the identical
`splendor_ai.game.engine.apply_action`. The CLI never independently decides
legality: even syntactically well-formed commands (like a 2-different-color
pick, which is only legal when the bank has fewer than 3 colors left) are
handed to the engine and rejected there if illegal. `tests/test_human_ai_action_parity.py`
proves the two paths produce byte-identical state transitions across many
random games.

An illegal action prints exactly why, never mutates the state, and never
consumes the turn -- `splendor_ai/cli/explain.py` turns the engine's
rejection into a specific reason (insufficient bank tokens, can't afford a
card, already at the reservation limit, a pending token-return/noble-choice
must be resolved first, etc.) by re-reading the same state the engine just
checked, never by re-deciding legality itself. `legal` lists every currently
legal command in this exact syntax.

Token-return and noble-choice are required sub-actions handled the same way:
picking tokens that pushes you over 10 is a fully legal main action, but the
turn isn't over until you `RETURN` down to exactly 10; qualifying for
multiple nobles at once requires an explicit `NOBLE <n>`. Both reject
unrelated commands (a `P`/`B`/`R` while a return is pending, e.g.) with a
message telling you what to complete first, and AI seats resolve the exact
same sub-decisions through the exact same action types.

The Pygame window is strictly read-only (nobles, all 12 visible cards with
remaining deck counts, bank, all 4 player panels, your own reserved cards
face-up and opponents' face-down, a recent-action log, a top-corner banner for
FINAL ROUND or a pending token-return/noble-choice, and a full final-ranking
overlay on game over). It renders in a background thread so it stays
responsive while the terminal blocks on `input()`; closing the window prints
a one-time terminal notice and the game keeps running in the terminal.
Training never launches it.

## Action representation

The engine's native action type is a small set of dataclasses
(`splendor_ai/game/actions.py`): `PickTokensAction`, `BuyCardAction`,
`ReserveCardAction`, `ReturnTokensAction`, `SelectNobleAction`, `PassAction`.
The RL environment encodes these into a **fixed, documented
`Discrete(149)`** index space (`splendor_ai/env/action_space.py`), built
entirely from token-color combinations and board *positions* -- never card or
noble identities, since slot contents change every game:

```
[  0- 29]  PICK_TOKENS    30 = 5 singles + 5 same-color pairs + 10 distinct pairs + 10 distinct triples
[ 30- 41]  BUY_VISIBLE    12 = 3 tiers x 4 slots
[ 42- 44]  BUY_RESERVED    3 = reserved slot 0/1/2
[ 45- 56]  RESERVE_VISIBLE 12 = 3 tiers x 4 slots
[ 57- 59]  RESERVE_BLIND   3 = one per tier
[ 60    ]  PASS            1 = only ever legal with zero other options
[ 61-143]  RETURN_TOKENS  83 = every multiset of size 1..3 over the 6 colors (incl. gold)
[144-148]  SELECT_NOBLE    5 = index into the *currently pending* eligible-noble list
```

`encode_action`/`decode_action` are pure, state-independent functions except
for `SELECT_NOBLE` (which needs the current `pending_eligible_nobles` list to
resolve a slot to an actual noble ID).

## Observation representation

A ~294-float vector, always encoded from the **current player's own point of
view** (`splendor_ai/env/observation.py`): own tokens/bonuses/prestige/
purchased-card-count/up-to-3-reserved-cards (by property: tier, bonus,
points, cost -- never by ID), each opponent's tokens/bonuses/prestige/
purchased-card-count/reserved-**count**, bank tokens, all 12 visible cards
(by property), remaining-deck-count per tier, up to 5 nobles, a one-hot of
which absolute seat is "me", the current round number, and a final-round
flag. Returned alongside a `action_mask` array in a `gymnasium.spaces.Dict`.

**Hidden-information protection** (tested in `tests/test_observation.py`):
opponents' reserved-card *identities* never appear (only a count); deck
*order* never appears (only a remaining count). Changing which card an
opponent has reserved, or shuffling a deck's internal order, provably does
not change the observation vector.

## Legal-action masking

`splendor_ai/env/action_mask.py` computes a length-149 0/1 array directly
from `splendor_ai.game.rules.legal_actions()` -- the same function the
terminal CLI uses for its `legal` command and the same one every baseline
agent queries. There is exactly one source of truth for legality in the
whole project.

## Reinforcement learning

### Environment design

Splendor is 4-player and turn-based; `SplendorEnv` (a standard
`gymnasium.Env`) always acts for "whichever seat is currently to move"
(`info["seat"]`). Since training uses **one shared policy across all 4 seats**
(parameter sharing), there's no cross-seat credit-assignment problem in the
usual multi-agent sense -- but there IS a subtlety that took a real bug (caught
during Phase 5 testing, see `tests/test_self_play.py`) to get right:

A single game's turns interleave chronologically across all 4 seats. GAE
(Generalized Advantage Estimation) must never bootstrap a seat's value
estimate off the very next buffer entry, because that next entry usually
belongs to a **different seat's turn** -- its value has nothing to do with the
acting seat's continuation. `splendor_ai/training/self_play.py` keeps four
separate per-seat trajectory buffers per environment so each one only ever
contains that seat's own chronological decisions. It also retroactively
patches the terminal reward (`info["terminal_scores"]`, populated by the env
on the step that ends the game) onto the other three seats' *last* stored
transition, since only the mover's own step naturally receives it.

### Reward design (`configs/training.yaml` -> `reward:`)

- Terminal: `win: 1.0`, `loss: -1.0`, `tie: 0.0` (a genuine multi-winner tie,
  per the official tie-break rule).
- Small per-step shaping, deliberately far below the terminal magnitude so it
  can never dominate winning: `prestige_point_coef: 0.02` per prestige point
  gained, `bonus_gained_coef: 0.01` per permanent bonus gained.

All values are configurable.

### Training

```bash
python -m splendor_ai.training.train --config configs/training.yaml
python -m splendor_ai.training.train --resume checkpoints/latest.pt
python -m splendor_ai.training.train --updates 200 --num-envs 4 --device cpu
```

PPO with legal-action masking (illegal logits are set to -inf before the
categorical distribution, both for sampling during rollout collection and for
the log-prob/entropy used in the PPO loss). `--device cuda` is supported if
available; this project was developed and evaluated CPU-only (see
[CPU/GPU requirements](#expected-cpu-and-gpu-training-requirements)).

Self-play optionally mixes in **frozen historical-checkpoint opponents** for
a random subset of seats each update round (`self_play.historical_*` in
`configs/training.yaml`) -- their transitions are never used for the
gradient update (nothing to learn from a policy that isn't being updated),
only the seats using the live policy are recorded.

Logged per update, to **TensorBoard and CSV** (`runs/<run_name>/`): episode
count, average game length (turns per player), win rate by seat position,
average final prestige, policy loss, value loss, entropy, and (every
`eval_interval_updates`) win rate vs. random and vs. heuristic over
`eval_games` fresh games.

### Resuming

`--resume checkpoints/latest.pt` restores the network, optimizer, update
counter, total-games/total-steps counters, and the historical-opponent pool,
so training continues exactly where it left off (own test:
`tests/test_train_smoke.py::test_resume_continues_from_saved_update`).

## Baseline agents

- **Random** (`agents/random_agent.py`): uniform over `legal_actions()`.
- **Heuristic** (`agents/heuristic_agent.py`): greedily buys the best
  affordable card (weighted by points, how many other visible cards want that
  bonus color, and noble progress), reserves a strong unaffordable card when
  far short, otherwise picks tokens targeted at whatever card it's building
  toward, and returns tokens/chooses nobles sensibly. Confirmed meaningfully
  stronger than random (`tests/test_evaluate.py`, win rate > 50% seated
  1-vs-3 against random, i.e. more than 4x its fair share).

### Measured baselines (1000 games each, `evaluation_results/`)

| | avg turns/player | avg final prestige | avg purchased cards |
|---|---|---|---|
| Random vs random (all 4 seats) | 35.4 | 10.6 | 13.5 |
| Heuristic vs heuristic (all 4 seats) | 28.9 | 11.8 | 14.2 |

Reproduce with:
```bash
python -m splendor_ai.training.evaluate --baseline --opponent random --games 1000
python -m splendor_ai.training.evaluate --baseline --opponent heuristic --games 1000
```

## Evaluating agents

```bash
python -m splendor_ai.training.evaluate --checkpoint checkpoints/expert/best.pt --opponent random --games 1000
python -m splendor_ai.training.evaluate --checkpoint checkpoints/expert/best.pt --opponent heuristic --games 1000
python -m splendor_ai.training.evaluate --checkpoint checkpoints/medium/best.pt --opponent checkpoint --opponent-checkpoint checkpoints/expert/best.pt --games 1000
```

The agent under test occupies one seat, rotated across all 4 seats over the
requested games; reports win/tie/loss rate, win rate by seat, average
prestige, average purchased cards, and average game length -- **never** just
game length, and never a claim of quality from a single small batch.

## Easy / Medium / Expert difficulty selection

```bash
python -m splendor_ai.training.select_difficulty --run-dir checkpoints/<run_name> --games 1000
```

Scans every `update_*.pt` checkpoint saved by a training run and picks one
checkpoint per tier named in `configs/difficulty.yaml`'s `tiers` list
(`[easy, medium, expert]`, weakest to strongest), purely from measured win
rates (see that file for the exact thresholds, which you can and should
revisit once you have real training data):

- **Expert** (the top tier) = among checkpoints clearing a minimum
  win-rate-vs-random floor, the one with the best composite score (weighted
  win rate vs. random and vs. heuristic -- heuristic weighted higher as the
  harder benchmark), then double-checked with a second, independently seeded
  evaluation batch and flagged in the report if the score drifts.
- **Every tier below Expert** (Medium, then Easy) = the *earliest*
  (least-trained) checkpoint that clears its own win-rate floors (vs. random
  *and* vs. heuristic -- a self-play policy typically crushes random almost
  immediately, so a random-only floor would let "earliest qualifying"
  degenerate to the very first checkpoint ever evaluated) *and* loses the
  majority of games in a direct matchup against the tier immediately above it
  (already selected). If nothing satisfies both, the tool says so explicitly
  in its output rather than silently picking something that doesn't meet the
  bar.

As training keeps improving, promote a new checkpoint to Expert and let the
old tiers shift down (today's Expert becomes tomorrow's Medium, today's
Medium becomes tomorrow's Easy) rather than overwriting a tier with something
weaker.

Selected checkpoints are copied to `checkpoints/<tier>/best.pt` for each tier
(`checkpoints/easy/best.pt`, `checkpoints/medium/best.pt`,
`checkpoints/expert/best.pt`), each with a `selection_report.json` recording
every number the decision was based on. Add `--dry-run` to see the selection
without copying anything.

## Expected CPU and GPU training requirements

Measured on this machine (16-core CPU, no CUDA available): one PPO update
(16 parallel envs x 2048 steps ≈ 380 self-play games, one gradient pass) took
**≈61 seconds**. `collect_rollout` runs the game engine (pure Python) in a
single process, so throughput scales with CPU single-thread speed more than
core count; only the network forward/backward pass benefits from more cores
via BLAS threading. A GPU mainly helps once `num_parallel_envs` /
`rollout_steps` are large enough that the network forward/backward pass
dominates over environment stepping -- with the default sizes here the game
engine itself is the bottleneck, so `--device cuda` is unlikely to help much
without also increasing batch size substantially.

## How to measure whether the AI is actually improving

**Finishing a training run proves nothing by itself.** Only trust:
1. `eval/win_rate_vs_random` and `eval/win_rate_vs_heuristic` in TensorBoard/
   CSV, trending up over many updates (not a single reading).
2. `python -m splendor_ai.training.evaluate` over >=1000 games against random,
   heuristic, and earlier checkpoints -- a checkpoint tournament, not one
   number.
3. Checkpoint-vs-checkpoint matchups (`--opponent checkpoint
   --opponent-checkpoint <earlier checkpoint>`) to confirm later checkpoints
   actually beat earlier ones, not just random/heuristic.
4. Win rate broken down **by seat position** -- a large imbalance (like
   heuristic's seat-0-vs-seat-3 spread in the measured baseline above) can
   indicate positional bias rather than genuine skill.

Never conclude "the AI is well trained" from training having completed, from
a small number of games, or from average game length alone (a fast loss is
not good performance -- this project deliberately evaluates win rate,
prestige, purchased-card count, and seat-position performance together, per
project policy).

## Known limitations

- **TensorBoard may not import in some environments.** This dev environment
  has a broken protobuf/tensorflow/tensorboard version combination; training
  detects the failed import and falls back to CSV-only logging automatically
  (a printed `NOTE:` says so). If you hit this, `pip install --upgrade
  tensorboard protobuf` in a clean virtual environment typically resolves it;
  CSV logs (`runs/<run_name>/training_log.csv`) always work regardless.
- **Self-play parallelism is single-process.** `num_parallel_envs` batches the
  network's forward/backward pass across environments, but each environment's
  `step()` (pure Python game logic) runs sequentially in one process/thread.
  A subprocess- or thread-pool-based vector environment would let game logic
  itself parallelize across cores; not implemented here.
- **A structural deadlock is possible under the official rules** (extremely
  rare: all 4 players simultaneously hit the 3-reserved-card limit while the
  bank has run dry of normal-color tokens and nobody's holdings match any
  visible card -- found via 1000-game heuristic-self-play stress testing,
  ~0.5% of games). The engine detects a full round of consecutive passes
  (state literally cannot change further) and ends the game immediately,
  scored by current prestige/tie-break rather than looping forever. This is
  an engine-level addition beyond the literal official rulebook, needed for
  any headless simulation to terminate.
- **Actual "Expert" strength depends entirely on how long you train.** The
  PPO/self-play pipeline is complete and tested end-to-end (including a real
  short training run), but genuine convergence for a game like Splendor
  typically needs far more self-play games than a short session allows on
  CPU. Use `--resume` to keep training, then re-run
  `select_difficulty.py` -- do not assume the shipped checkpoints (if any) are
  strong without checking their evaluation numbers first.
- `RLAgent` defaults to `deterministic=True` (greedy argmax over the masked
  policy) for both evaluation and play, so behavior is reproducible and
  doesn't waste strength on unforced randomness; there is no separate
  "difficulty knob" beyond which checkpoint is loaded.
