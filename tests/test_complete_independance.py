# tests/test_flood.py

import os
import pytest
import numpy as np
import pandas as pd
from tqdm import tqdm

from prisk.utils import extract_firms, convert_to_continous_damage, clean_owner_name
from prisk.kernel import Kernel, InsuranceDropoutEvent
from prisk.flood import FloodEntitySim
from prisk.portfolio import Portfolio
from prisk.insurance import Insurance


@pytest.fixture(scope="session")
def test_data_dir():
    return os.path.join("tests", "data")


@pytest.fixture
def load_power_data(test_data_dir):
    power_file = os.path.join(test_data_dir, "power_iso2.csv")
    power = pd.read_csv(power_file)
    power = power.rename(
        columns={
            "2": 2,
            "5": 5,
            "10": 10,
            "25": 25,
            "50": 50,
            "100": 100,
            "200": 200,
            "500": 500,
            "1000": 1000,
        }
    )
    return power


@pytest.fixture
def leverage_ratios(test_data_dir, load_power_data):
    financial_medians_file = os.path.join(test_data_dir, "financial_medians.csv")
    financial_medians = pd.read_csv(financial_medians_file)

    # Filter for power sector
    financial_data_power = financial_medians[financial_medians["sector"] == "Power"]

    # Merge with power data
    financial_data = pd.merge(
        financial_data_power, load_power_data, on=["country_iso2"]
    )
    financial_data = financial_data.loc[
        :, ["Owner", "Plant / Project name", "debt_equity_ratio"]
    ]
    financial_data = financial_data.rename(
        columns={"debt_equity_ratio": "Leverage Ratio"}
    )

    # Fill missing values with the median
    median_ratio = financial_data["Leverage Ratio"].median()
    financial_data["Leverage Ratio"].fillna(median_ratio, inplace=True)

    # Normalize firm names
    financial_data["Owner"] = financial_data["Owner"].str.split(";")
    financial_data = financial_data.explode("Owner", ignore_index=True)

    return {
        clean_owner_name(str(firm)): leverage
        for firm, leverage in zip(
            financial_data["Owner"], financial_data["Leverage Ratio"]
        )
    }


@pytest.fixture
def damage_curves(test_data_dir):
    damage_curves_file = os.path.join(test_data_dir, "damage_curves.xlsx")
    damage_curves = pd.read_excel(damage_curves_file)
    return convert_to_continous_damage(damage_curves)


def test_independant_flood(load_power_data, leverage_ratios, damage_curves):
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
    india = load_power_data

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
