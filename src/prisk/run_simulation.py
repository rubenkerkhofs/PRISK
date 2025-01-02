import copy
import pandas as pd
import numpy as np
from tqdm import tqdm


from prisk.utils import events_df, merton_probability_of_default
from prisk.kernel import Kernel, InsuranceDropoutEvent
from prisk.flood import FloodEntitySim, FloodBasinSim

from prisk.insurance import Insurance

import copy
import pandas as pd
import numpy as np
from tqdm import tqdm


def run_simulation(
    random_numbers,
    assets_df,
    firms,
    portfolio,
    simulations=10,
    time_horizon=25,
    seed=0,
    insured=False,
    insurer_capital=2e9,
    dropout_time=None,
    calculate_pds=False,
    sigma=None,
    return_period_columns=None,
):
    """
    Runs portfolio simulations based on the presence or absence of a copula.

    Parameters:
    - random_numbers (pd.DataFrame or None): Random numbers for copula. If None, no copula is applied.
    - assets (list): List of asset objects.
    - firms (list): List of firm objects for PD calculations.
    - portfolio (Portfolio): Portfolio object managing asset positions.
    - simulations (int): Number of simulation runs.
    - time_horizon (int): Simulation period in years.
    - seed (int): Random seed for reproducibility.
    - insured (bool): Whether to include insurance in the simulation.
    - insurer_capital (float): Capital allocated to the insurance company.
    - dropout_time (int or None): Time after which insurance drops out. If None, no dropout.
    - calculate_pds (bool): Whether to calculate empirical and Merton PDs.
    - sigma (float): Volatility parameter for the Merton model.
    - return_period_columns (list): return period columns in the assets_df

    Returns:
    - result (dict): Dictionary containing portfolio values and PDs (if applicable).
        - "portfolio_values" (pd.Series)
        - "empirical_pds" (pd.DataFrame) [if calculate_pds=True]
        - "merton_pds" (pd.DataFrame) [if calculate_pds=True]
    """
    np.random.seed(seed)
    portfolio_values = []
    empirical_pds = [] if calculate_pds else None
    merton_pds = [] if calculate_pds else None

    copula_type = "No Copula" if random_numbers is None else "With Copula"

    for sim in tqdm(range(simulations), desc=f"Simulating {copula_type}"):
        assets = assets_df.asset.tolist()

        # Deep copy assets to avoid mutation across simulations
        # TODO FIX so that the function can work with deepcopy (not with pointers)
        # assets_copy = copy.deepcopy(assets)
        assets_copy = assets

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
        if random_numbers is not None:
            events = events_df(
                random_numbers=random_numbers,
                return_period_columns=return_period_columns,
                years=time_horizon,
            )
            # Basin-level Dependence using FloodBasinSim
            for asset in assets_copy:
                flood_basin = assets_df[assets_df.asset == asset].HYBAS_ID.iloc[0]
                events_asset = events[events.basin == flood_basin]
                FloodBasinSim(asset, events_asset).simulate(kernel=kernel)
        else:
            # Independent Asset-level Events using FloodEntitySim
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
