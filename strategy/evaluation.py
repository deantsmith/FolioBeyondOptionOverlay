"""
Strategy Evaluation

This module orchestrates the complete strategy evaluation pipeline:
1. Load calibrated parameters
2. Run simulations
3. Evaluate spread candidates
4. Compute risk metrics
5. Compare and rank strategies
"""

import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import ModelConfig, RiskConfig, ExitRuleConfig
from calibration import (
    BivariateVasicekParams, VasicekParams,
    TLTRegressionParams,
    VolatilityParams
)
from simulation import (
    run_simulation, simulate_tlt_paths,
    SimulationResult, TLTSimulationResult
)
from pricing import (
    SpreadType, SpreadDefinition, SpreadQuote,
    create_spread_by_delta, price_spread,
    ExitRules, simulate_spread_batch, SimulationBatchResult
)
from strategy.recommendation import (
    apply_constraints,
    extract_core_metrics,
    rank_results
)


@dataclass
class CalibratedModel:
    """Container for all calibrated model components."""
    
    vasicek_params: BivariateVasicekParams
    regression_params: TLTRegressionParams
    volatility_params: VolatilityParams
    current_values: Dict[str, float]
    
    @classmethod
    def from_json(cls, filepath: str) -> 'CalibratedModel':
        """Load calibrated model from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Reconstruct Vasicek params
        vp = data['vasicek_params']
        vasicek_params = BivariateVasicekParams(
            params_20y=VasicekParams(
                kappa=vp['kappa_20y'],
                theta=vp['theta_20y'],
                sigma=vp['sigma_20y']
            ),
            params_30y=VasicekParams(
                kappa=vp['kappa_30y'],
                theta=vp['theta_30y'],
                sigma=vp['sigma_30y']
            ),
            correlation=vp['correlation']
        )
        
        # Reconstruct regression params
        rp = data['regression_params']
        regression_params = TLTRegressionParams(
            alpha=rp['alpha'],
            beta=rp['beta'],
            gamma=rp.get('gamma'),
            r_squared=rp.get('r_squared', 0),
            rmse=rp.get('rmse', 0),
            reference_yield=rp.get('reference_yield', 0),
            implied_duration=rp.get('implied_duration', 0)
        )
        
        # Reconstruct volatility params
        volp = data['volatility_params']
        volatility_params = VolatilityParams(
            base_vol=volp['base_vol'],
            base_iv=volp['base_iv'],
            price_sensitivity=volp['price_sensitivity'],
            reference_price=volp['reference_price'],
            vol_percentile_25=volp.get('vol_percentile_25', volp['base_vol'] * 0.9),
            vol_percentile_50=volp.get('vol_percentile_50', volp['base_vol']),
            vol_percentile_75=volp.get('vol_percentile_75', volp['base_vol'] * 1.1)
        )
        
        return cls(
            vasicek_params=vasicek_params,
            regression_params=regression_params,
            volatility_params=volatility_params,
            current_values=data['current_values']
        )


@dataclass
class StrategyCandidate:
    """A strategy candidate with its evaluation results."""
    
    spread_quote: SpreadQuote
    simulation_result: SimulationBatchResult
    
    # Computed metrics
    expected_return: float = 0.0
    pop: float = 0.0
    cvar_95: float = 0.0
    risk_adjusted_return: float = 0.0
    kelly_fraction: float = 0.0

    # Ranking
    rank: int = 0
    passes_constraints: bool = True
    constraint_violations: List[str] = field(default_factory=list)
    
    def compute_metrics(self):
        """Compute strategy metrics from simulation results."""
        from strategy.metrics import kelly_criterion

        (
            self.expected_return,
            self.pop,
            self.cvar_95,
            self.risk_adjusted_return
        ) = extract_core_metrics(self.simulation_result)

        # Kelly fraction for position sizing
        self.kelly_fraction = kelly_criterion(self.simulation_result.pnls)


@dataclass 
class EvaluationConfig:
    """Configuration for strategy evaluation."""
    
    # Simulation parameters
    n_paths: int = 10000
    random_seed: Optional[int] = 42
    
    # Spread parameters to evaluate
    expiration_days: int = 45
    delta_targets: List[float] = field(default_factory=lambda: [0.15, 0.20, 0.25, 0.30])
    spread_widths: List[float] = field(default_factory=lambda: [2, 3, 5])
    spread_types: List[SpreadType] = field(
        default_factory=lambda: [SpreadType.BULL_PUT, SpreadType.BEAR_CALL]
    )
    
    # Risk constraints
    min_pop: float = 0.70
    max_cvar_loss: Optional[float] = None  # Will be set from risk budget
    
    # Risk-free rate
    risk_free_rate: float = 0.05
    
    # Exit rules
    exit_rules: ExitRules = field(default_factory=ExitRules)


def _kelly_contracts(candidate: StrategyCandidate, nav: float) -> int:
    """
    Calculate number of contracts based on Kelly criterion.

    Parameters
    ----------
    candidate : StrategyCandidate
        Strategy candidate with Kelly fraction computed.
    nav : float
        Portfolio net asset value.

    Returns
    -------
    int
        Recommended number of contracts.
    """
    kelly_capital = nav * candidate.kelly_fraction
    max_loss_per_contract = candidate.spread_quote.definition.width * 100
    contracts = int(kelly_capital / max_loss_per_contract)
    return max(1, contracts)


class StrategyEvaluator:
    """
    Main class for evaluating option strategies.
    
    Usage:
        model = CalibratedModel.from_json('calibration_results.json')
        evaluator = StrategyEvaluator(model)
        results = evaluator.evaluate_strategies()
    """
    
    def __init__(
        self,
        model: CalibratedModel,
        config: Optional[EvaluationConfig] = None
    ):
        self.model = model
        self.config = config or EvaluationConfig()
        
        # Will be populated during evaluation
        self.rate_simulation: Optional[SimulationResult] = None
        self.tlt_simulation: Optional[TLTSimulationResult] = None
        self.candidates: List[StrategyCandidate] = []
    
    def run_simulations(self) -> TLTSimulationResult:
        """Run Monte Carlo simulations for rate and TLT paths."""
        print("Running Monte Carlo simulations...")
        
        # Get current yields
        current = self.model.current_values
        r0_20 = current['yield_20y']
        r0_30 = current['yield_30y']
        
        # Run rate simulation
        self.rate_simulation = run_simulation(
            params=self.model.vasicek_params,
            r0_20=r0_20,
            r0_30=r0_30,
            horizon_days=self.config.expiration_days,
            n_paths=self.config.n_paths,
            random_seed=self.config.random_seed
        )
        
        # Convert to TLT paths
        self.tlt_simulation = simulate_tlt_paths(
            self.rate_simulation,
            self.model.regression_params,
            self.model.volatility_params
        )
        
        print(f"  Generated {self.config.n_paths:,} paths over {self.config.expiration_days} days")
        print(f"  Initial TLT: ${self.tlt_simulation.initial_price:.2f}")
        
        return self.tlt_simulation
    
    def generate_candidates(self) -> List[SpreadQuote]:
        """Generate spread candidates to evaluate."""
        spot = self.model.current_values['tlt_price']
        sigma = self.model.volatility_params.base_iv
        r = self.config.risk_free_rate
        
        candidates = []
        
        for spread_type in self.config.spread_types:
            for delta in self.config.delta_targets:
                for width in self.config.spread_widths:
                    try:
                        definition = create_spread_by_delta(
                            spot, r, sigma, self.config.expiration_days,
                            delta, width, spread_type
                        )
                        quote = price_spread(definition, spot, r, sigma)
                        candidates.append(quote)
                    except Exception as e:
                        continue
        
        print(f"  Generated {len(candidates)} spread candidates")
        return candidates
    
    def evaluate_candidate(self, quote: SpreadQuote) -> StrategyCandidate:
        """Evaluate a single spread candidate."""
        if self.tlt_simulation is None:
            raise ValueError("Run simulations first")
        
        # Run batch simulation
        batch_result = simulate_spread_batch(
            definition=quote.definition,
            initial_credit=quote.net_credit,
            price_paths=self.tlt_simulation.tlt_paths,
            iv_paths=self.tlt_simulation.iv_paths,
            r=self.config.risk_free_rate,
            rules=self.config.exit_rules,
            use_fast=True
        )
        
        candidate = StrategyCandidate(
            spread_quote=quote,
            simulation_result=batch_result
        )
        candidate.compute_metrics()
        
        return candidate
    
    def check_constraints(self, candidate: StrategyCandidate) -> StrategyCandidate:
        """Check if candidate passes risk constraints."""
        passes, violations = apply_constraints(
            pop=candidate.pop,
            cvar_95=candidate.cvar_95,
            min_pop=self.config.min_pop,
            max_cvar_loss=self.config.max_cvar_loss
        )
        candidate.passes_constraints = passes
        candidate.constraint_violations = violations
        
        return candidate
    
    def rank_candidates(self, candidates: List[StrategyCandidate]) -> List[StrategyCandidate]:
        """Rank candidates by risk-adjusted return."""
        return rank_results(
            candidates,
            get_rar=lambda c: c.risk_adjusted_return,
            get_passes=lambda c: c.passes_constraints,
            set_rank=lambda c, rank: setattr(c, "rank", rank)
        )
    
    def evaluate_strategies(
        self,
        verbose: bool = True
    ) -> List[StrategyCandidate]:
        """
        Run complete strategy evaluation pipeline.
        
        Returns list of candidates sorted by rank.
        """
        if verbose:
            print("\n" + "=" * 70)
            print("STRATEGY EVALUATION")
            print("=" * 70)
        
        # Step 1: Run simulations
        self.run_simulations()
        
        # Step 2: Generate candidates
        if verbose:
            print("\nGenerating spread candidates...")
        quotes = self.generate_candidates()
        
        # Step 3: Evaluate each candidate
        if verbose:
            print("\nEvaluating candidates...")
        
        self.candidates = []
        for i, quote in enumerate(quotes):
            candidate = self.evaluate_candidate(quote)
            candidate = self.check_constraints(candidate)
            self.candidates.append(candidate)
            
            if verbose and (i + 1) % 10 == 0:
                print(f"  Evaluated {i + 1}/{len(quotes)} candidates")
        
        # Step 4: Rank candidates
        self.candidates = self.rank_candidates(self.candidates)
        
        if verbose:
            n_passing = sum(1 for c in self.candidates if c.passes_constraints)
            print(f"\n{n_passing}/{len(self.candidates)} candidates pass constraints")
        
        return self.candidates
    
    def get_top_candidates(self, n: int = 5) -> List[StrategyCandidate]:
        """Get top N ranked candidates."""
        return [c for c in self.candidates if c.passes_constraints][:n]
    
    def print_results(self, top_n: int = 10):
        """Print evaluation results."""
        print("\n" + "=" * 90)
        print("TOP STRATEGY CANDIDATES")
        print("=" * 90)
        
        top = self.get_top_candidates(top_n)
        
        if not top:
            print("\nNo candidates pass all constraints!")
            return
        
        # Header
        print(f"\n{'Rank':<6} {'Type':<10} {'Strikes':<12} {'Credit':<10} "
              f"{'POP':<8} {'E[R]':<10} {'CVaR95':<10} {'RAR':<8} {'Kelly%':<8}")
        print("-" * 100)

        for c in top:
            spread = c.spread_quote.definition
            spread_type = "Put" if spread.is_put_spread else "Call"
            strikes = f"{spread.short_strike}/{spread.long_strike}"

            print(f"{c.rank:<6} {spread_type:<10} {strikes:<12} "
                  f"${c.spread_quote.net_credit:<9.2f} "
                  f"{c.pop:<7.1%} "
                  f"${c.expected_return:<9.2f} "
                  f"${c.cvar_95:<9.2f} "
                  f"{c.risk_adjusted_return:<7.2f} "
                  f"{c.kelly_fraction:<7.1%}")

        print("=" * 100)
        
        # Show exit distribution for top candidate
        if top:
            best = top[0]
            stats = best.simulation_result.stats

            print(f"\nTop Strategy Details: {best.spread_quote.definition}")
            print(f"  Initial Credit:     ${best.spread_quote.net_credit:.2f}")
            print(f"  Max Profit:         ${stats['max_profit']:.2f}")
            print(f"  Max Loss:           ${stats['max_loss']:.2f}")
            print(f"  Avg Days Held:      {stats['avg_days_held']:.1f}")

            print(f"\n  Position Sizing (Kelly):")
            print(f"    Kelly Fraction:   {best.kelly_fraction:.1%}")
            print(f"    For $1M NAV:      {_kelly_contracts(best, 1_000_000)} contracts")
            print(f"    For $10M NAV:     {_kelly_contracts(best, 10_000_000)} contracts")

            print(f"\n  Exit Reason Distribution:")
            for reason, pct in stats['exit_reason_pcts'].items():
                if pct > 0:
                    print(f"    {reason:<20} {pct:.1%}")


def run_evaluation(
    calibration_file: str,
    config: Optional[EvaluationConfig] = None,
    verbose: bool = True
) -> List[StrategyCandidate]:
    """
    Convenience function to run complete evaluation.
    
    Parameters
    ----------
    calibration_file : str
        Path to calibration JSON file.
    config : EvaluationConfig, optional
        Evaluation configuration.
    verbose : bool
        Print progress and results.
        
    Returns
    -------
    List[StrategyCandidate]
        Ranked strategy candidates.
    """
    # Load model
    model = CalibratedModel.from_json(calibration_file)
    
    if verbose:
        print(f"Loaded model from {calibration_file}")
        print(f"  Current TLT: ${model.current_values['tlt_price']:.2f}")
        print(f"  Current 20Y: {model.current_values['yield_20y']*100:.3f}%")
        print(f"  Current 30Y: {model.current_values['yield_30y']*100:.3f}%")
    
    # Create evaluator
    evaluator = StrategyEvaluator(model, config)
    
    # Run evaluation
    candidates = evaluator.evaluate_strategies(verbose=verbose)
    
    if verbose:
        evaluator.print_results()
    
    return candidates
