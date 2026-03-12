import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd
from scipy.optimize import minimize

from estimate_ou import *
from imm_filter import run_pipeline, load_spread_data, split_sessions, ou_params_final  # Your current file renamed

def raw_signal(z_score, z_scale=2.0):
    """Raw directional signal: -tanh(z/z_scale)."""
    return -np.tanh(z_score / z_scale)

if __name__ == "__main__":
    # Load Kalman results
    combined = load_spread_data()
    sessions = split_sessions(combined)
    results = run_pipeline(sessions, combined)
    
    # Test Layer 1 Step 1
    z_test = results[0]['z_score']
    raw_sig = raw_signal(z_test)
    
    print("Layer 1 Step 1 — RawSignal:")
    print(f"z range: [{np.min(z_test):.2f}, {np.max(z_test):.2f}]")
    print(f"RawSignal range: [{np.min(raw_sig):.3f}, {np.max(raw_sig):.3f}]")
    print(f"Mean |RawSignal|: {np.mean(np.abs(raw_sig)):.3f}")


def layer1_validity_factors(kalman_result, p_max_factor=3.0):
    """Layer 1 Steps 2-4: Gate, Filter, Regime factors."""
    P = kalman_result['P']
    p1 = kalman_result['p1']
    
    # Gate (placeholder)
    gate_factor = np.ones_like(P)
    
    # Filter: max(0, 1 - P/P_max)
    p_max = p_max_factor * np.median(P)
    filter_factor = np.maximum(0, 1 - P / p_max)
    
    # Regime: IMM p1
    regime_factor = p1
    
    return gate_factor, filter_factor, regime_factor

def signal_score(z_score, validity_factors):
    """Layer 1 complete: RawSignal × validity factors."""
    raw_signal = -np.tanh(z_score / 2.0)
    gate, filt, regime = validity_factors
    return raw_signal * gate * filt * regime

# Test Layer 1 complete
validity = layer1_validity_factors(results[0])
signal_scores = signal_score(results[0]['z_score'], validity)

print("Layer 1 Complete — SignalScore:")
print(f"RawSignal range: [{np.min(-np.tanh(results[0]['z_score']/2)):.3f}, {np.max(-np.tanh(results[0]['z_score']/2)):.3f}]")
print(f"SignalScore range: [{np.min(signal_scores):.3f}, {np.max(signal_scores):.3f}]")
print(f"Mean |SignalScore|: {np.mean(np.abs(signal_scores)):.3f}")
print(f"FilterFactor mean: {np.mean(validity[1]):.3f}")
print(f"RegimeFactor mean: {np.mean(validity[2]):.3f}")

def compute_cost_factor(z_score, ou_params, tc_bps=5):
    """Layer 1 Step 5: Transaction cost adjustment."""
    kappa = ou_params['kappa']
    hl_min = ou_params['half_life_min']
    tc_decimal = tc_bps / 10000  # 5bps → 0.0005
    
    expected_edge = np.abs(z_score) * kappa * hl_min
    cost_factor = np.maximum(0, 1 - tc_decimal / np.maximum(expected_edge, 1e-6))
    return cost_factor

# Complete Layer 1
def layer1_complete(kalman_result, ou_params, p_max_factor=3.0):
    """Full Layer 1 SignalScore."""
    z_score = kalman_result['z_score']
    
    # Validity factors
    P = kalman_result['P']
    p1 = kalman_result['p1']
    gate_factor = np.ones_like(P)
    p_max = p_max_factor * np.median(P)
    filter_factor = np.maximum(0, 1 - P / p_max)
    regime_factor = p1
    cost_factor = compute_cost_factor(z_score, ou_params)
    
    # Raw + factors
    raw_signal = -np.tanh(z_score / 2.0)
    signal_score = raw_signal * gate_factor * filter_factor * regime_factor * cost_factor
    
    return signal_score

# Test
ou_params = {'kappa': 0.04, 'half_life_min': 17.3}  # Your Phase 3
final_signal = layer1_complete(results[0], ou_params)

print("Layer 1 FINAL — w/ CostFactor:")
print(f"SignalScore range: [{np.min(final_signal):.3f}, {np.max(final_signal):.3f}]")
print(f"Mean |SignalScore|: {np.mean(np.abs(final_signal)):.3f}")
print(f"CostFactor mean: {np.mean(compute_cost_factor(results[0]['z_score'], ou_params)):.3f}")

def layer2_target_position(signal_score, kalman_result, ou_params, risk_pct=0.02):
    """Layer 2: SignalScore → TargetPosition % notional."""
    z_score = kalman_result['z_score']
    p1 = kalman_result['p1']
    P = kalman_result['P']
    
    # 1. SizeScalar (fixed risk budget)
    gld_level = np.mean(kalman_result['GLD']) if 'GLD' in kalman_result else 470
    dollar_vol = 0.002 * gld_level  # 0.2% daily vol → $0.94 at GLD=$470
    size_scalar = risk_pct / dollar_vol  # 2% / $0.94 ≈ 0.021
    
    # 2. RiskScore components
    z_risk = np.clip(np.abs(z_score) - 2.0, 0, 1)
    regime_risk = np.clip(1 - p1, 0, 1)
    filter_risk = np.clip(P / (3*np.median(P)), 0, 1)
    time_risk = np.zeros_like(z_score)  # Placeholder (needs entry time)
    hl_jump_risk = np.zeros_like(z_score)  # Placeholder
    
    weights = np.array([0.3, 0.2, 0.2, 0.2, 0.1])
    risk_score = (weights @ np.array([z_risk, regime_risk, filter_risk, time_risk, hl_jump_risk]))
    
    # 3. TimeDecay (placeholder — full version needs entry tracking)
    time_decay = np.ones_like(signal_score)
    
    # TargetPosition
    target_position = (signal_score * size_scalar * 
                      (1 - np.sqrt(np.clip(risk_score, 0, 1))) * time_decay)
    
    return target_position, risk_score

# Phase 3 (one line)
ou_final = ou_params_final(np.std(results[0]['spread'] - results[0]['x_hat']))

target_pos, risk_scores = layer2_target_position(final_signal, results[0], ou_final)
print("\nLayer 2 — TargetPosition:")
print(f"TargetPosition range: [{np.min(target_pos)*100:.2f}%, {np.max(target_pos)*100:.2f}%]")
print(f"Max |TargetPosition|: {np.max(np.abs(target_pos))*100:.1f}%")
print(f"RiskScore mean: {np.mean(risk_scores):.3f}")


def layer3_execution(target_position, current_position, min_delta=0.005):
    """Layer 3: TargetPosition → orders."""
    # Pad current_position to match target (starts at 0)
    current_pos_padded = np.zeros_like(target_position)
    current_pos_padded[1:] = current_position[:-1]  # Shift forward
    
    delta = target_position - current_pos_padded  # Now both (390,)
    
    # Execute meaningful deltas
    execute_mask = np.abs(delta) > min_delta
    orders = np.zeros_like(delta)
    orders[execute_mask] = delta[execute_mask]
    
    # Update position
    new_position = current_pos_padded + orders
    
    return orders, new_position

# Test (pass zeros as initial position)
current_pos = np.zeros_like(target_pos)
orders, final_pos = layer3_execution(target_pos, current_pos, min_delta=0.005)

print("\nLayer 3 — Execution:")
print(f"Orders generated: {np.sum(orders != 0)}")
print(f"Final position range: [{np.min(final_pos)*100:.2f}%, {np.max(final_pos)*100:.2f}%]")
print(f"Max |order|: {np.max(np.abs(orders))*100:.2f}%")

def proper_spread_pnl(session_data, target_position, beta):
    """Correct GLD - β·IAU PnL in bps."""
    gld_prices = session_data['GLD'] if hasattr(session_data['GLD'], 'values') else session_data['GLD']
    iau_prices = session_data['IAU'] if hasattr(session_data['IAU'], 'values') else session_data['IAU']
    
    
    gld_ret = np.diff(np.log(gld_prices))
    iau_ret = np.diff(np.log(iau_prices))
    spread_ret = gld_ret - beta * iau_ret  # Raw spread return
    
    # Position % GLD notional → bps PnL
    pnl = target_position[1:] * spread_ret * 10000  # Convert to bps
    return pnl

def phase5_walkforward_fixed(results, risk_pct=0.02):
    """Phase 5: Full walk-forward w/ correct PnL."""
    all_pnl = []
    session_sharpes = []
    
    for i, result in enumerate(results):
        # Store raw prices for PnL (add to result)
        result['GLD'] = sessions[i]['GLD'].values
        result['IAU'] = sessions[i]['IAU'].values
        
        ou_params = ou_params_final(np.std(result['spread'] - result['x_hat']))
        
        # Phase 4 pipeline
        signal_scores = layer1_complete(result, ou_params)
        target_pos, risk_scores = layer2_target_position(signal_scores, result, ou_params, risk_pct)
        
        # Proper PnL
        beta = result['beta']
        pnl = proper_spread_pnl(result, target_pos, beta)
        
        all_pnl.append(pnl)
        sharpe = np.mean(pnl) / np.std(pnl) * np.sqrt(252 * 78 / 390) if np.std(pnl) > 0 else 0
        session_sharpes.append(sharpe)
        
        print(f"Session {i}: PnL={pnl.sum():+.0f}bps, Sharpe={sharpe:.2f}")
    
    # Aggregate
    all_pnl_flat = np.concatenate(all_pnl)
    total_sharpe = (np.mean(all_pnl_flat) / np.std(all_pnl_flat) * 
                   np.sqrt(252 * 78)) if len(all_pnl_flat) > 0 and np.std(all_pnl_flat) > 0 else 0
    
    print("\n=== PHASE 5 COMPLETE ===")
    print(f"Sessions: {len(results)}")
    print(f"Total PnL: {all_pnl_flat.sum():+.0f} bps")
    print(f"Total Sharpe: {total_sharpe:.2f}")
    print(f"Avg session Sharpe: {np.mean(session_sharpes):.2f}")
    print(f"Win rate: {np.mean(np.sign(all_pnl_flat)):.1%}")
    
    return all_pnl_flat, session_sharpes

# RUN PHASE 5 (add to __main__)
pnl_series, sharpes = phase5_walkforward_fixed(results)

# Plot cumulative PnL
plt.figure(figsize=(12, 6))
cum_pnl = np.cumsum(np.concatenate([[0], pnl_series]))
plt.plot(cum_pnl, linewidth=2)
plt.title('Phase 5: Walk-Forward Cumulative PnL')
plt.ylabel('Cumulative PnL (bps)')
plt.xlabel('1-min Bars')
plt.grid(True, alpha=0.3)
plt.show()


