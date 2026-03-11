# COMPLETE RISK AND POSITIONING MODEL
## Kalman Filter + Ornstein-Uhlenbeck Mean Reversion System
### Phase 4 — Signal Architecture, Position Management, and Adaptive Weight Engine

---

## CONTEXT AND PURPOSE

This document describes the complete risk and positioning model for a Kalman Filter + Ornstein-Uhlenbeck (OU) mean reversion trading strategy. The system is designed to trade intraday mean reversion on equity instruments (primary: AAPL at 1-minute bars) by detecting statistical deviations from an estimated equilibrium and entering positions when conditions are mathematically valid.

The model described here is NOT a simple decision tree. It is a three-layer continuous signal architecture that eliminates all discretionary judgment from trade management. Every decision — entry, sizing, partial exit, full exit, doing nothing — traces back to a mathematical quantity produced by the pipeline.

The pipeline that feeds this model (already built in Phases 1-3) produces every bar:
- x̂_k : Kalman-filtered price estimate (VB-AKF)
- P_k : Kalman error covariance (filter uncertainty)
- φ : OU mean-reversion coefficient
- μ : OU long-run mean
- σ_OU : OU residual standard deviation
- HL : half-life in bars = ln(2) / ln(1/φ)
- IMM_Regime_Score : p₁,k = probability of mean-reverting regime (0 to 1)
- Gate status : OPEN if ADF p-value < 0.05 AND Hurst < 0.45
- ADF p-value and Hurst exponent (raw values)

Everything below operates on these inputs. Nothing else is required.

---

## THE CORE DESIGN PHILOSOPHY

### Why Not a Decision Tree

A decision tree checks conditions sequentially and fires the first one that triggers. This creates three fatal problems:

**Shadowing:** When multiple conditions fire simultaneously, only the first is seen. All others are invisible. You lose diagnostic information and cannot distinguish between different types of failures.

**No aggregation:** Four simultaneous "reduce position" signals should produce a stronger response than one. A tree treats them identically — the first one fires and the rest are ignored.

**No continuity:** A tree produces discrete jumps — you are at 100% position, then suddenly at 50%, then suddenly at 0%. Real risk does not work in discrete jumps. A position that is 70% dangerous should produce a 70% response, not wait for a hard threshold.

### The Solution: Three Continuous Layers

Layer 1 produces a Signal Score — a single number from -1 to +1 encoding all entry conditions simultaneously.

Layer 2 produces a Target Position — a single number from -1 to +1 encoding desired exposure, incorporating signal strength, sizing preference, regime quality, time decay, and composite risk simultaneously.

Layer 3 is the Execution layer — it translates the difference between current and target position into actual orders, with portfolio-level safety checks.

Every condition that used to be a binary check in a decision tree is now a continuous factor in one of these layers. Nothing is checked sequentially. Everything is evaluated simultaneously.

---

## LAYER 1 — THE SIGNAL SCORE

### Purpose

Produces a single number Signal_Score ∈ [-1, +1] that encodes:
- Direction of the trade (positive = bullish, negative = bearish)
- Strength of the signal (magnitude)
- All validity conditions simultaneously (gate, filter, regime, cost)

A Signal_Score of 0 means no signal. +0.8 means a strong bullish mean-reversion opportunity. -0.6 means a moderate bearish opportunity.

### Computation

**Step 1 — Raw directional signal from z-score:**

    Raw_Signal = -tanh(z_t / z_scale)

where z_scale = 2.0 (tunable). The tanh function is chosen deliberately:
- It naturally bounds output to (-1, +1)
- At z = 1.5 → raw signal ≈ 0.64 (meaningful entry)
- At z = 2.5 → raw signal ≈ 0.92 (strong signal)
- At z = 3.5 → raw signal ≈ 0.98 (saturated — no additional benefit to sizing)
- The sign flip (-tanh) is because a negative z-score (price below mean) → positive signal (buy)

This replaces the fixed entry threshold of z = 1.5. Instead of a hard cutoff, the signal grows continuously. Low z-scores produce weak signals (small positions). High z-scores produce strong signals (larger positions). No cliff edge.

**Step 2 — Multiply by validity factors:**

    Signal_Score = Raw_Signal × Gate_Factor × Filter_Factor × Regime_Factor × Cost_Factor

Each factor is a continuous number from 0 to 1:

    Gate_Factor    = 1.0 if gate OPEN, 0.0 if gate CLOSED
                     (Hard binary — the only binary in Layer 1)

    Filter_Factor  = max(0, 1 - P_k / P_max)
                     (Goes to 0 as Kalman uncertainty approaches its limit)
                     (At P_k = 0 → factor = 1.0, full trust)
                     (At P_k = P_max → factor = 0.0, no signal)

    Regime_Factor  = IMM_Regime_Score (p₁,k directly)
                     (Already 0 to 1 — mean-reverting probability IS the factor)
                     (At p₁ = 0.8 → regime is strong, full signal passes through)
                     (At p₁ = 0.2 → trending regime, signal is muted to 20%)

    Cost_Factor    = max(0, (expected_gross - round_trip_costs) / expected_gross)
                     (Goes to 0 when expected edge disappears after costs)
                     (Prevents entering when z-score deviation is too small to cover costs)

**Key property of this structure:**

When ANY factor is 0, Signal_Score = 0. The gate closing kills the signal completely. The filter hitting its uncertainty limit kills the signal completely. A fully trending regime (p₁ = 0) kills the signal completely. No condition can dominate another — they all multiply.

When ALL factors are healthy (near 1.0), Signal_Score ≈ Raw_Signal. The signal passes through undiluted.

### Hard Exits That Bypass Layer 1

Two conditions force Signal_Score = 0 immediately, overriding all computation:

1. |z_t| > 3.5 → Hard z-score stop. Extreme deviation that should not occur under genuine OU dynamics. Model has failed. Set Signal_Score = 0. Exit everything.

2. Gate closes while in a trade → The statistical foundation of the position no longer holds. Set Signal_Score = 0. Exit everything.

These are the ONLY binary checks in the entire system. Everything else is continuous.

---

## LAYER 2 — THE TARGET POSITION

### Purpose

Translates Signal_Score into a desired position size as a fraction of maximum allowed position. Target_Position ∈ [-1, +1].

Positive = long. Negative = short. 0 = flat. 0.5 = half-sized long. -1.0 = full short.

The execution layer will move current position toward this target every bar.

### Two Sizing Modes (Your Switch 2)

You configure one of these modes before the session starts. The mode determines how Signal_Score magnitude maps to risk budget.

**Mode A — Fixed Risk:**

    Risk_Pct = your_fixed_percentage (e.g., 1.5%)

Every trade risks the same percentage of account at the hard stop. Predictable, auditable.

**Mode B — Dynamic Risk (z-score scaled):**

    Risk_Pct = min_risk + clip((|z_t| - 1.5) / (2.5 - 1.5), 0, 1) × (max_risk - min_risk)

You set a range [min_risk, max_risk], e.g., [0.75%, 2.0%]. Risk scales continuously within that range based on z-score magnitude. Stronger signal → larger position. Weaker signal → smaller position. Hard cap at max_risk regardless of z-score.

### The Size Scalar

From Risk_Pct, compute the number of shares using the volatility-normalized formula:

    Max_Shares = (Account × Risk_Pct × IMM_Regime_Score) / (|z_t| × σ_OU × Price)

The denominator (|z_t| × σ_OU × Price) is the dollar move per share from entry to the hard stop zone. Dividing the risk budget by this gives the share count where a worst-case move costs exactly Risk_Pct of account.

IMM_Regime_Score appears again in the numerator — double application is intentional. Regime quality affects both the signal (Layer 1) and the sizing (Layer 2). A deteriorating regime mutes the signal AND reduces the size.

### Time Decay — Fixed At Entry

This solves the perverse HL-extension problem.

When you enter a trade, lock in HL_entry = current half-life at that exact bar. Do not update it. Use it to compute:

    Time_Decay = max(0, 1 - (trade_age_bars / (3 × HL_entry))²)

This is a quadratic decay from 1.0 at entry to 0.0 at 3 × HL_entry bars.

Properties:
- At trade age = 0 → Time_Decay = 1.0 (full size)
- At trade age = 1 × HL_entry → Time_Decay = 0.89 (position slightly reduced)
- At trade age = 2 × HL_entry → Time_Decay = 0.56 (position meaningfully reduced)
- At trade age = 3 × HL_entry → Time_Decay = 0.0 (position fully exited)

The critical property: Time_Decay uses HL_ENTRY, not current HL. If φ jumps mid-trade and HL extends from 4 bars to 23 bars, the time decay clock does NOT reset. The trade that was entered on a 4-bar HL premise still expires on the original schedule. This prevents φ instability from indefinitely extending losing trades.

### The Composite Risk Score

Every bar, for any open position, compute:

    Risk_Score = w1·Z_Risk + w2·Regime_Risk + w3·Filter_Risk + w4·Time_Risk + w5·HL_Jump_Risk

where the five factors are:

    Z_Risk       = clip((|z_t| - 2.0) / 1.5,  0, 1)
                   (0 when |z| ≤ 2.0, grows to 1 at |z| = 3.5)
                   (Measures how far price has moved against you)

    Regime_Risk  = clip(1 - IMM_Score / 0.3,   0, 1)
                   (0 when regime is healthy, grows as IMM falls below 0.3)
                   (Measures regime deterioration)

    Filter_Risk  = clip(P_k / P_max,            0, 1)
                   (0 when filter is converged, grows to 1 at P_max)
                   (Measures Kalman filter uncertainty)

    Time_Risk    = clip((trade_age / (2 × HL_entry))², 0, 1)
                   (0 at entry, grows quadratically to 1 at 2 × HL_entry)
                   (Uses HL_entry — locked at entry, never updates)
                   (Measures how far into expected reversion window we are)

    HL_Jump_Risk = clip((HL_current - HL_entry) / HL_entry / 2, 0, 1)
                   (0 when HL unchanged, grows when current HL extends beyond entry HL)
                   (Measures OU dynamics weakening mid-trade)

Risk_Score ∈ [0, 1]. Four simultaneous risk signals now produce a score of ~0.8. One weak signal produces a score of ~0.1. The aggregation is automatic and continuous.

The weights w1..w5 are produced by the Adaptive Weight Estimator (see separate section below).

### Final Target Position Formula

    Target_Position = Signal_Score × Size_Scalar × (1 - Risk_Score^0.5) × Time_Decay

Breaking this down:
- Signal_Score: direction and signal quality from Layer 1
- Size_Scalar: risk budget per trade in share count
- (1 - Risk_Score^0.5): reduces position as composite risk grows. The square root is deliberate — it makes the reduction aggressive at moderate risk levels rather than waiting until Risk_Score is near 1.0
- Time_Decay: continuously shrinks position as trade ages past the HL_entry schedule

When Risk_Score = 0 (no risk factors active): Target_Position = Signal_Score × Size_Scalar
When Risk_Score = 0.25: position is reduced to 50% of full size
When Risk_Score = 0.64: position is reduced to 20% of full size
When Risk_Score = 1.0: Target_Position = 0 (fully flat)

---

## YOUR SWITCHES — MODE CONFIGURATION

### Switch 1 — Entry
Fully automated. Fires when Signal_Score crosses a minimum threshold (e.g., |Signal_Score| > 0.3, corresponding to roughly z = 0.6 after all validity factors). No human input. No override. If any validity factor is 0, Signal_Score = 0 and no entry fires.

### Switch 2 — Position Sizing
Set once before session:
- Mode A: single fixed Risk_Pct
- Mode B: [min_risk, max_risk] range with z-score scaling

### Switch 3 — Take Profit
Set once before session:

**Mode A — Target only:**
Hold full position until Target_Position crosses zero (meaning z has reverted and Signal_Score has flipped). No partial taking. Clean single exit.

**Mode B — Half-life partial:**
At T = 1 × HL_entry, if gap has closed by ≥ 30% of original deviation: close 50% of position, lock in that profit, let remaining 50% run to full target. If gap has NOT closed by 30%: do not take partial — instead treat this as a risk signal and the Time_Risk factor automatically begins reducing position via the Risk_Score.

Regardless of mode, four dynamic events force partials or full exits immediately:

    Event 1 — Regime deteriorates mid-trade:
    IMM_Regime_Score drops below 0.20 → Regime_Risk spikes → Risk_Score rises → Target_Position automatically reduces. No discrete "take 50%" rule needed — the continuous formula handles it.

    Event 2 — Gate closes mid-trade:
    Hard exit. Bypasses all continuous logic. Target_Position = 0 immediately.

    Event 3 — P_k spikes above P_max mid-trade:
    Filter_Risk = 1.0 → Risk_Score spikes → Target_Position automatically reduces toward 0.

    Event 4 — HL doubles from entry value:
    HL_Jump_Risk grows → Risk_Score rises → Target_Position automatically reduces. The size of the response scales with how much HL has extended.

### Switch 4 — Loss Management
Hard exits (bypass all continuous logic):
- |z_t| > 3.5 → Signal_Score = 0 → Target_Position = 0. Exit immediately.
- Gate closes → Signal_Score = 0 → Target_Position = 0. Exit immediately.

Continuous loss management (handled automatically by the formula):
- Z_Risk grows as |z| increases beyond 2.0 → Risk_Score rises → position shrinks
- Time_Risk grows quadratically → position shrinks automatically with age
- All four dynamic events above produce automatic reductions

No discrete "reduce to 50%" rules needed. The formula produces continuous reductions that aggregate multiple simultaneous signals naturally.

### When to Do Nothing
Signal_Score = 0 whenever ANY of:
- Gate is closed (Gate_Factor = 0)
- P_k ≥ P_max (Filter_Factor = 0)
- IMM_Regime_Score = 0 (Regime_Factor = 0, fully trending)
- Cost_Factor ≤ 0 (edge covered by costs)
- |z_t| too small (Raw_Signal near 0)

No discrete "do nothing" condition is needed. Signal_Score = 0 means Target_Position = 0 means no trade is placed.

---

## LAYER 3 — EXECUTION AND SAFETY

### Purpose
Takes Target_Position and Current_Position. Determines what orders to place. Handles portfolio-level safety that individual instrument logic cannot see.

### Step 1 — Portfolio Heat Check
Compute total portfolio risk = sum of (Risk_Pct × account) across ALL open positions. If total exceeds portfolio heat limit (e.g., 5% of account):

    Scale_Factor = Heat_Limit / Total_Portfolio_Risk
    Target_Position = Target_Position × Scale_Factor

This automatically scales back the new position to keep total portfolio risk within limits. A single position in a well-managed portfolio might be fine at 2% risk. Five simultaneous positions each at 2% = 10% total heat — far too much. Layer 3 detects this and scales.

### Step 2 — Daily Circuit Breaker
Track cumulative realized P&L for the current session. If daily_loss > daily_loss_limit (e.g., -2% of account):

    Target_Position = 0 for remainder of session.

System stops generating new positions. Existing positions are managed normally by Layers 1 and 2, but no new entries fire. This prevents a bad day from becoming a catastrophic day.

### Step 3 — Anti-Whipsaw Filter
Compute required position change: Delta = Target_Position - Current_Position

If |Delta| < minimum_trade_threshold (e.g., 0.10 = 10% of max position):
    Do not execute. Transaction cost of a tiny adjustment exceeds its risk benefit.

After a hard stop exit: apply cooldown. For the next max(1, HL_entry) bars, require |Signal_Score| > 0.7 (a much stronger signal) before allowing re-entry. Prevents immediately re-entering a position that was just stopped out.

After a normal profit-target exit: no cooldown. Re-entry is allowed immediately if conditions are valid.

### Step 4 — Execute
If Delta > minimum_trade_threshold → place buy order for Delta × Max_Shares shares
If Delta < -minimum_trade_threshold → place sell order for |Delta| × Max_Shares shares
If |Delta| ≤ threshold → no action this bar

---

## THE ADAPTIVE WEIGHT ESTIMATOR (w1..w5)

### The Problem It Solves
The five weights in the Composite Risk Score cannot be fixed. The right weights depend on the current market regime:
- In mean-reverting conditions, Z_Risk is most predictive
- In regime transitions, Regime_Risk dominates
- In noisy sessions, Filter_Risk is most important

The Adaptive Weight Estimator runs every bar and produces optimal w1..w5 from three layers:

### Layer A — Predictive Power (empirical, backward-looking)
Over the last 60 bars, for each risk factor, compute the Spearman rank correlation between:
- The factor's value at bar t
- The realized adverse price move at bar t+1

Spearman rank correlation is used (not Pearson) because:
- Risk factors are not normally distributed
- Rank correlation is robust to price spikes and news events
- Monotonic relationship matters more than linear

Result: five correlation values. The factor that has been most predictive of actual bad moves gets the highest weight. A factor with no predictive relationship gets near-zero weight.

### Layer B — Regime-Conditional Scaling (theoretical, forward-looking)
Reads current bar state (IMM probability, ADF p-value, φ stability, σ stability) and computes a multiplier for each factor based on theoretical importance in the current regime:

- Z_Risk multiplier grows when: regime is genuinely mean-reverting AND series is stationary (z-score is most meaningful here)
- Regime_Risk multiplier grows when: IMM probability is near 0.5 (maximum regime uncertainty)
- Filter_Risk multiplier grows when: P_k is near P_max AND φ has been unstable (filter uncertainty is most dangerous here)
- Time_Risk multiplier grows when: φ has been stable (reliable HL makes the time exit meaningful)
- HL_Jump_Risk multiplier grows when: regime is transitional AND φ is unstable (HL extensions are most dangerous here)

Layer B multiplies each factor's predictive score from Layer A. A factor that is both historically predictive AND theoretically important right now gets amplified. A factor that is historically irrelevant OR theoretically inappropriate right now gets dampened.

### Layer C — Stability Dampening (reliability check)
Computes a stability factor ∈ [0, 1] based on:
- Observation count (low count → low stability → trust prior more)
- φ stability over window (σ(φ) < 0.08 → stable, σ(φ) > 0.15 → unreliable)
- Regime certainty (IMM near 0.5 → transition in progress → lower stability)
- σ_OU stability (high coefficient of variation → lower stability)

Final weights:
    estimated = normalize(Layer_A × Layer_B)
    blend = dampening × stability_factor    (dampening = 0.70 default)
    weights = blend × estimated + (1 - blend) × prior

When stability is 0: use prior completely.
When stability is 1 with dampening = 0.70: 70% estimated, 30% prior.

### Prior Weights (Safety Net)
Three regime-specific priors used when data is insufficient:

    Mean-reverting: [0.35, 0.15, 0.20, 0.20, 0.10]
                     (Z_Risk dominates — z-score most predictive in stable OU conditions)

    Transitional:   [0.20, 0.30, 0.30, 0.10, 0.10]
                     (Regime_Risk and Filter_Risk share dominance in uncertain conditions)

    Trending:       [0.15, 0.35, 0.15, 0.10, 0.25]
                     (Regime_Risk and HL_Jump_Risk elevated — most dangerous factors in trends)

### Usage Pattern (One Call Per Bar)
    estimator = AdaptiveWeightEstimator(window=60, stability_min_obs=30, dampening=0.70)

    # Every bar, in this exact order:
    if prev_bar is not None:
        adverse_move = abs(current_price - prev_price)
        estimator.record_outcome(adverse_move)   # ← BEFORE update()

    weights = estimator.update(bar)

    result = compute_composite_risk_score(bar, weights)
    position_scalar = result["position_scalar"]   # multiply Target_Position by this

record_outcome() must come BEFORE update(). It writes the realized outcome of the PREVIOUS bar into the buffer. update() then uses the full buffer including that outcome for Layer A estimation.

### Four Tunable Parameters (Walk-Forward Validated)
    window            : bars used for predictive power estimation (default 60, match OU window)
    stability_min_obs : minimum realized observations before Layer A activates (default 30)
    dampening         : max trust in estimated vs prior weights (default 0.70)
    regime_sensitivity: how aggressively Layer B shifts weights at regime boundaries (default 1.0)

These are the ONLY parameters requiring calibration. Optimize through walk-forward testing in Phase 5. They replace the five fixed w1..w5 values that would otherwise require arbitrary manual specification.

---

## COMPLETE DECISION FLOW — EVERY BAR

    New bar arrives → pipeline produces x̂_k, P_k, φ, μ, σ_OU, HL, IMM_Score, Gate, z_t
    │
    ├─ HARD EXIT CHECK (bypass all layers):
    │   |z_t| > 3.5 OR Gate just closed → Target_Position = 0 → execute immediately → STOP
    │
    ├─ ADAPTIVE WEIGHT ESTIMATOR:
    │   record_outcome(|current_price - prev_price|)   ← outcome of previous bar
    │   weights = estimator.update(bar)                ← new w1..w5 for this bar
    │
    ├─ LAYER 1 — Signal Score:
    │   Raw_Signal    = -tanh(z_t / 2.0)
    │   Gate_Factor   = 1 or 0
    │   Filter_Factor = max(0, 1 - P_k / P_max)
    │   Regime_Factor = IMM_Score
    │   Cost_Factor   = max(0, (gross_edge - costs) / gross_edge)
    │   Signal_Score  = Raw_Signal × Gate × Filter × Regime × Cost
    │
    ├─ LAYER 2 — Target Position:
    │   Risk_Score    = w·[Z_Risk, Regime_Risk, Filter_Risk, Time_Risk, HL_Jump_Risk]
    │   Time_Decay    = max(0, 1 - (age / (3 × HL_entry))²)
    │   Size_Scalar   = (Account × Risk_Pct × IMM_Score) / (|z| × σ_OU × Price)
    │   Target_Pos    = Signal_Score × Size_Scalar × (1 - Risk_Score^0.5) × Time_Decay
    │
    └─ LAYER 3 — Execution:
        Portfolio heat → scale Target_Pos if total risk exceeds limit
        Daily circuit breaker → set Target_Pos = 0 if daily loss limit hit
        Delta = Target_Pos - Current_Pos
        Anti-whipsaw → suppress if |Delta| < minimum threshold
        Cooldown check → require stronger signal if within post-stop cooldown
        Execute Delta → buy or sell the difference

---

## WHAT THIS ARCHITECTURE ELIMINATES

The following concepts from naive decision tree designs are replaced entirely:

| Old Concept | Replaced By |
|---|---|
| Fixed entry at z = 1.5 | Continuous tanh signal — enters at any z, scales with strength |
| Fixed "reduce to 50%" rule | Risk_Score continuously reduces position — no discrete jump |
| Checking conditions sequentially | All factors multiply simultaneously in Signal_Score |
| Static w1..w5 weights | Adaptive Weight Estimator re-derives weights every bar |
| HL extension giving bad trades more time | Time_Decay locked to HL_entry — never extends |
| Binary open/closed position state | Continuous Target_Position — any fraction is valid |
| IMM threshold dead zone | Continuous Regime_Factor — no gap between thresholds |
| Missing portfolio heat check | Layer 3 Step 1 — explicit portfolio risk limit |
| No daily loss limit | Layer 3 Step 2 — session circuit breaker |
| Whipsaw from immediate re-entry | Layer 3 Step 3 — post-stop cooldown |
| Multiple simultaneous signals ignored | Risk_Score aggregates all five factors in weighted sum |

---

## KEY MATHEMATICAL PROPERTIES

1. Signal_Score = 0 whenever ANY validity factor = 0. Gate closure alone is sufficient to kill any signal. No sequential checking needed.

2. Risk_Score is bounded [0, 1] by construction. Each factor is individually bounded [0, 1]. Weighted sum of bounded factors is bounded.

3. Target_Position = 0 whenever Risk_Score = 1. The system exits automatically when composite risk reaches maximum. No separate "exit" rule needed.

4. Time_Decay reaches 0 at exactly 3 × HL_entry bars. Hard time exit is built into the continuous formula — not a separate check.

5. The position scalar (1 - Risk_Score^0.5) uses the square root of Risk_Score. This makes the reduction nonlinear: aggressive at moderate risk levels (Risk_Score = 0.25 → scalar = 0.50), accelerating as risk grows (Risk_Score = 0.64 → scalar = 0.20), complete at maximum (Risk_Score = 1.0 → scalar = 0.0).

6. Layer A (predictive power) and Layer B (regime scaling) are independent. Layer A measures what has worked empirically. Layer B measures what should work theoretically. Both must agree for a factor to receive high weight. A factor that is empirically predictive but theoretically inappropriate gets dampened. A factor that is theoretically important but empirically flat in the current window gets dampened.

7. The adaptive weights w1..w5 sum to 1.0 at all times by construction (renormalized after each estimation step). Risk_Score is therefore always on the same scale regardless of which factors dominate.

---

## PARAMETERS REQUIRING CONFIGURATION (COMPLETE LIST)

The following parameters are configured once before the session based on walk-forward validation results from Phase 5. They are not adjusted during live trading.

    Sizing:
      risk_pct_fixed      (Mode A) or risk_pct_min / risk_pct_max (Mode B)
      portfolio_heat_limit  (e.g., 5% of account total across all positions)
      daily_loss_limit      (e.g., 2% of account per session)
      position_cap_pct      (max single trade as % of account, e.g., 10%)
      position_cap_adv      (max single trade as % of ADV, e.g., 5%)

    Signal:
      z_scale               (tanh scaling factor, default 2.0)
      signal_entry_threshold (minimum |Signal_Score| to allow entry, e.g., 0.30)
      P_max                 (maximum Kalman error covariance for valid signal)

    Execution:
      minimum_trade_threshold  (minimum |Delta| to execute, e.g., 0.10)
      cooldown_bars_after_stop (bars before re-entry allowed after hard stop)
      stronger_signal_threshold (minimum |Signal_Score| during cooldown, e.g., 0.70)

    Adaptive Weights:
      window               (estimation window in bars, e.g., 60)
      stability_min_obs    (minimum realized observations before Layer A activates, e.g., 30)
      dampening            (max blend toward estimated weights, e.g., 0.70)
      regime_sensitivity   (Layer B aggressiveness, e.g., 1.0)

    Profit Taking (Switch 3):
      partial_exit_mode    (A = target only, B = half-life partial)
      partial_exit_threshold_pct  (minimum gap closure % for Mode B partial, e.g., 30%)

---

## HOW TO INTEGRATE THIS WITH THE EXISTING PIPELINE

The pipeline (Phase 3) produces x̂_k, P_k, φ, μ, σ_OU, HL, IMM_Score, Gate, ADF, Hurst every bar. Feed these directly into BarData. Compute z_t = (x̂_k - μ) / σ_OU. Pass BarData to the AdaptiveWeightEstimator and compute_composite_risk_score. Feed the resulting weights and position_scalar into the Layer 2 Target_Position formula. Pass Target_Position to the Layer 3 execution logic.

The adaptive weight estimator and the risk/positioning layers are Phase 4 components. They sit between the Phase 3 pipeline and the Phase 5 walk-forward validation. Phase 5 will validate all parameters listed above by running the complete system over historical data and measuring out-of-sample Sharpe ratio, signal hit rate, gate effectiveness, and whether the adaptive weights produce meaningfully better performance than fixed weights.

---

## IMPORTANT CONCEPTUAL DISTINCTIONS

**Risk_Score vs Signal_Score:** Signal_Score measures opportunity (how good is the entry). Risk_Score measures danger (how dangerous is the current position). They are computed independently. A trade can have a strong initial Signal_Score and a growing Risk_Score simultaneously — this is the normal evolution of a trade as it ages.

**Target_Position vs Current_Position:** Target_Position is what the model wants. Current_Position is what you have. They are only equal immediately after execution. Between bars, market moves make them diverge. The execution layer trades the difference (Delta) every bar.

**HL_entry vs current HL:** HL_entry is locked at the moment of trade entry. Current HL is recomputed every bar from the latest φ. Time_Decay and Time_Risk always use HL_entry. HL_Jump_Risk uses the ratio of current HL to HL_entry. This distinction is critical — conflating them reintroduces the perverse extension problem.

**Predictive power (Layer A) vs regime scaling (Layer B):** Layer A is purely statistical — it measures what actually happened. Layer B is purely theoretical — it measures what should matter given current conditions. The final weights blend both. Neither alone is sufficient: pure statistical estimation overfits to recent history; pure theoretical weights ignore empirical evidence.
