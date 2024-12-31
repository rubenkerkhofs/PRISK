# tests/test_flood.py

import os
import pytest
import numpy as np
import pandas as pd
from tqdm import tqdm

from prisk.utils import extract_firms, convert_to_continous_damage
from prisk.kernel import Kernel, InsuranceDropoutEvent
from prisk.flood import FloodEntitySim
from prisk.portfolio import Portfolio
from prisk.insurance import Insurance


@pytest.fixture(scope="session")
def test_data_dir():
    return os.path.join("tests", "data")


@pytest.fixture
def load_power_data(test_data_dir):
    power_file = os.path.join(test_data_dir, "power.xlsx")
    power = pd.read_excel(power_file)
    return power


@pytest.fixture
def setup_india_data(load_power_data, flood_protection_value=100, country="India"):
    india = (
        load_power_data[load_power_data["Country"] == country]
        .copy()
        .reset_index(drop=True)
        .drop(columns=[2])
    )
    india["flood_protection"] = flood_protection_value
    return india


@pytest.fixture
def indian_firm_mapping(test_data_dir):
    indian_firms_file = os.path.join(test_data_dir, "Indian_firms.xlsx")
    indian_firms = pd.read_excel(indian_firms_file)  # Load damage curves
    return {
        row["name"]: row["clean"]
        for _, row in indian_firms[["name", "clean"]].iterrows()
    }


@pytest.fixture
def leverage_ratios(test_data_dir):
    financial_data_file = os.path.join(test_data_dir, "Indian_firms.xlsx")
    financial_data = pd.read_excel(financial_data_file)
    median_ratio = financial_data["Leverage Ratio"].median()
    financial_data["Leverage Ratio"].fillna(median_ratio, inplace=True)
    return {
        firm: leverage
        for firm, leverage in zip(
            financial_data["clean"], financial_data["Leverage Ratio"]
        )
    }


@pytest.fixture
def damage_curves(test_data_dir):
    """
    Fixture to load damage curves and convert them to continuous curves.
    """
    damage_curves_file = os.path.join(test_data_dir, "damage_curves.xlsx")
    damage_curves = pd.read_excel(damage_curves_file)  # Load damage curves
    continuous_curves = convert_to_continous_damage(
        damage_curves
    )  # Convert to continuous curves
    return continuous_curves


def test_independant_flood(
    setup_india_data, indian_firm_mapping, leverage_ratios, damage_curves
):
    # Test parameters
    country = "India"
    flood_protection = 100
    insured = True  # You can parametrize this as needed
    insurer_capital = 2e9
    simulations = 5  # Reduced number for test speed
    random_seed = 0
    sigma = 0.2
    time_horizon = 25

    # Define or load your damage_curves and return_period_columns
    return_period_columns = [5, 10, 25, 50, 100, 200, 500, 1000]

    # Ensure return_period_columns exist in the data
    india = setup_india_data

    # Ensure necessary columns exist
    if "Value" not in india.columns:
        india["Value"] = 1e6  # Assign default values or load actual data
    if "Capacity (MW)" not in india.columns:
        india["Capacity (MW)"] = 100  # Assign default values or load actual data

    # Extract firms using the refactored function
    firms, india_with_assets = extract_firms(
        assets=india,
        damage_curves=damage_curves,
        return_period_columns=return_period_columns,
        leverage_ratios=leverage_ratios,
        indian_firm_mapping=indian_firm_mapping,
        discount_rate=0.05,
        unit_price=60,
        margin=0.2,
        time_horizon=time_horizon,
    )

    # Create a portfolio
    portfolio = Portfolio(f"{country} power assets")
    nav = 1
    for firm in firms:
        # Assuming each firm has an npv attribute
        # Replace `firm.npv` with the actual method or attribute to get NPV
        if hasattr(firm, "npv") and firm.npv != 0:
            portfolio.add_position(firm, nav / (len(firms) * firm.npv))
        else:
            portfolio.add_position(firm, 0)  # Handle firms with zero NPV appropriately

    np.random.seed(random_seed)

    portfolio_values = []
    for i in tqdm(range(simulations), desc="Simulating"):
        assets = india_with_assets["asset"].tolist()
        insurer = Insurance(
            "Insurance company", capital=insurer_capital, subscribers=[]
        )
        kernel = Kernel(assets=assets, insurers=[insurer])

        for asset in assets:
            if insured:
                asset.add_insurer(insurer)
                # Assuming dropout_time is defined or replace with actual value
                dropout_time = 5  # Example value; replace as needed
                InsuranceDropoutEvent(kernel.internal_time + dropout_time, asset).send(
                    kernel
                )

            # Simulate floods at the asset level
            FloodEntitySim(asset).simulate(time_horizon=time_horizon, kernel=kernel)

        kernel.run(time_horizon=time_horizon, verbose=0)
        portfolio_values.append(portfolio.underlying_value)

        # Reset the assets to their initial state
        for asset in assets:
            asset.reset()

    # Assertions
    assert (
        len(portfolio_values) == simulations
    ), "Number of portfolio values should match simulations count"

    # Additional domain-specific assertions can be added here
    # For example, check if portfolio values are within expected ranges
    for value in portfolio_values:
        assert isinstance(value, (int, float)), "Portfolio values should be numeric"
        # Add more checks as per your domain requirements
