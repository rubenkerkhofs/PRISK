import copy
import pandas as pd
import numpy as np
from tqdm import tqdm


from prisk.utils import events_df, merton_probability_of_default
from prisk.kernel import Kernel, InsuranceDropoutEvent
from prisk.flood import FloodEntitySim, FloodBasinSim

from prisk.insurance import Insurance


def run_simulation(
    copula,
    assets,
    random_numbers,
    firms,
    portfolio,
    simulations=10,
    time_horizon=25,
    seed=0,
    insured=False,
    insurer_capital=2e9,
    dropout_time=None,
    calculate_pds=False,
    sigma=0.2,
):
    """
    Runs portfolio simulations for a specific copula type.

    Parameters:
    - copula (str or None): Copula type ('uniform', 'gaussian', 't', 'vine', 'complete_dependent') or None for 'uniform'.
    - assets (list): List of asset objects. Should be basin-linked if copula is not 'uniform' or None.
    - random_numbers (pd.DataFrame): Random numbers corresponding to the copula.
    - firms (list): List of firm objects for PD calculations.
    - portfolio (Portfolio): Portfolio object managing asset positions.
    - simulations (int): Number of simulation runs.
    - time_horizon (int): Simulation period in years.
    - seed (int): Random seed for reproducibility.
    - insured (bool): Whether assets are insured.
    - insurer_capital (float): Capital allocated to the insurance company.
    - dropout_time (int or None): Time after which insurance drops out. If None, no dropout.
    - calculate_pds (bool): Whether to calculate empirical and Merton PDs.
    - sigma (float): Volatility parameter for the Merton model.

    Returns:
    - result (dict): Dictionary containing portfolio values and PDs.
        - "portfolio_values" (pd.Series)
        - "empirical_pds" (pd.DataFrame) [if calculate_pds=True]
        - "merton_pds" (pd.DataFrame) [if calculate_pds=True]
    """
    np.random.seed(seed)
    portfolio_values = []
    empirical_pds = [] if calculate_pds else None
    merton_pds = [] if calculate_pds else None

    for sim in tqdm(
        range(simulations), desc=f"Simulating {copula or 'Uniform'} Copula"
    ):
        # Deep copy assets to avoid mutation across simulations
        assets_copy = copy.deepcopy(assets)

        # Initialize Insurance if applicable
        insurer = (
            Insurance("Insurance Company", capital=insurer_capital, subscribers=[])
            if insured
            else None
        )
        insurers = [insurer] if insurer else []

        # Initialize Kernel
        kernel = Kernel(assets=assets_copy, insurers=insurers)

        # Simulate Flood Events based on Copula
        if copula in ["gaussian", "t", "vine", "complete_dependent"]:
            # Basin-level Dependence
            for asset in assets_copy:
                flood_basin = (
                    asset.HYBAS_ID
                )  # Assumes each asset has a HYBAS_ID attribute
                events_asset = events_df(
                    random_numbers=random_numbers, years=time_horizon
                )
                FloodBasinSim(asset, events_asset).simulate(kernel=kernel)
        else:
            # Independent Asset-level Events ('uniform' or None)
            for asset in assets_copy:
                FloodEntitySim(asset).simulate(time_horizon=time_horizon, kernel=kernel)

        # Optionally Add Insurance to Assets
        if insured and insurer:
            for asset in assets_copy:
                asset.add_insurer(insurer)
                if dropout_time is not None:
                    # Schedule Insurance Dropout Event
                    InsuranceDropoutEvent(
                        kernel.internal_time + dropout_time, asset
                    ).send(kernel)

        # Run the Kernel Simulation
        kernel.run(time_horizon=time_horizon, verbose=0)

        # Collect Portfolio Value
        portfolio_values.append(portfolio.underlying_value)

        # Calculate PDs if applicable
        if calculate_pds:
            # Empirical PDs
            empirical_pds_sim = [firm.delta_pd for firm in firms]
            empirical_pds.append(empirical_pds_sim)

            # Merton PDs
            merton_pds_sim = [
                -merton_probability_of_default(
                    V=firm.base_value, sigma_V=sigma, D=firm.original_liabilities
                )
                + merton_probability_of_default(
                    V=firm.npv, sigma_V=sigma, D=firm.original_liabilities
                )
                for firm in firms
            ]
            merton_pds.append(merton_pds_sim)

        # Reset Assets
        for asset in assets_copy:
            asset.reset()

    # Compile Results
    result = {"portfolio_values": pd.Series(portfolio_values)}

    if calculate_pds:
        result["empirical_pds"] = pd.DataFrame(
            empirical_pds, columns=[firm.name for firm in firms]
        )
        result["merton_pds"] = pd.DataFrame(
            merton_pds, columns=[firm.name for firm in firms]
        )

    return result
