import os
import pytest
import numpy as np
import pandas as pd
from tqdm import tqdm


from prisk.utils import (
    extract_firms,
    convert_to_continous_damage,
    merton_probability_of_default,
    link_basins,
    events_df,
)
from prisk.kernel import Kernel
from prisk.flood import FloodBasinSim
from prisk.portfolio import Portfolio
from prisk.insurance import Insurance
import geopandas as gpd


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


@pytest.fixture
def country_basins(test_data_dir):
    """
    Fixture to load country basins from the basin outlet file.
    """
    basin_outlet_file = os.path.join(
        test_data_dir, "HA_L6_outlets_India_constrained.csv"
    )
    country_basins = pd.read_csv(basin_outlet_file).HYBAS_ID.to_list()
    return country_basins


@pytest.fixture
def basins(test_data_dir):
    """
    Fixture to load GeoDataFrame of hydrological basins.
    """
    hybas_basins_file = os.path.join(test_data_dir, "hybas_as_lev06_v1c.shp")
    basins = gpd.read_file(hybas_basins_file)
    return basins


def test_basin_dependence(
    setup_india_data,
    indian_firm_mapping,
    leverage_ratios,
    damage_curves,
    country_basins,
    basins,
):
    # Test parameters
    flood_protection = 100
    insured = True
    insurer_capital = 2e9
    simulations = 5  # Reduced number for test speed
    random_seed = 0
    sigma = 0.2
    time_horizon = 25

    # Set up random numbers for event simulation
    np.random.seed(random_seed)
    uniform_random_numbers = pd.Series(np.random.uniform(size=1000))

    # Define return period columns
    return_period_columns = ["RP10", "RP50", "RP100"]

    # Prepare India data and ensure required columns
    india = setup_india_data
    for col in return_period_columns:
        if col not in india.columns:
            india[col] = 0  # Assign default values

    # Ensure necessary columns exist
    if "Value" not in india.columns:
        india["Value"] = 1e6
    if "Capacity (MW)" not in india.columns:
        india["Capacity (MW)"] = 100

    # Extract firms
    firms, india_with_assets = extract_firms(
        assets=india,
        damage_curves=damage_curves,
        return_period_columns=return_period_columns,
        indian_firm_mapping=indian_firm_mapping,
        leverage_ratios=leverage_ratios,
        discount_rate=0.05,
        unit_price=60,
        margin=0.2,
        time_horizon=time_horizon,
    )

    portfolio = Portfolio("Thailand power assets")
    nav = 1
    for firm in firms:
        portfolio.add_position(firm, nav / (len(firms) * firm.npv))

    # Link basins
    india_with_assets, basins = link_basins(
        india_with_assets, basins=basins, country_basins=country_basins, visualize=True
    )

    empirical_independent_basins = []
    merton_independent_basins = []
    portfolio_values_independent_basins = []

    for i in tqdm(range(simulations), desc="Simulating Basin Dependence"):
        # Generate events for basin-level dependence
        events = events_df(uniform_random_numbers, years=time_horizon)
        assets = india_with_assets["asset"].tolist()
        insurer = Insurance("Insurance company", capital=insurer_capital)
        kernel = Kernel(assets=assets, insurers=[insurer])

        for asset in assets:
            if insured:
                asset.add_insurer(insurer)
            flood_basin = india_with_assets[
                india_with_assets.asset == asset
            ].HYBAS_ID.iloc[0]
            events_asset = events[events.basin == str(flood_basin)]
            FloodBasinSim(asset, events_asset).simulate(kernel=kernel)

        kernel.run(time_horizon=time_horizon, verbose=0)

        # Calculate empirical and Merton probabilities of default
        empirical_pds = [firm.delta_pd for firm in firms]
        merton_pds = [
            -merton_probability_of_default(
                V=firm.base_value, sigma_V=sigma, D=firm.original_liabilities
            )
            + merton_probability_of_default(
                V=firm.npv, sigma_V=sigma, D=firm.original_liabilities
            )
            for firm in firms
        ]
        empirical_independent_basins.append(empirical_pds)
        merton_independent_basins.append(merton_pds)

        # Record portfolio value
        portfolio_values_independent_basins.append(portfolio.underlying_value)

        # Reset the assets
        for asset in assets:
            asset.reset()

    # Assertions
    assert (
        len(portfolio_values_independent_basins) == simulations
    ), "Number of portfolio values should match simulations count"
    assert (
        len(empirical_independent_basins) == simulations
    ), "Empirical PDs should match simulations count"
    assert (
        len(merton_independent_basins) == simulations
    ), "Merton PDs should match simulations count"

    # Check data types and ranges for probabilities of default
    for empirical_pds, merton_pds in zip(
        empirical_independent_basins, merton_independent_basins
    ):
        assert all(
            isinstance(pd, float) for pd in empirical_pds
        ), "Empirical PDs should be floats"
        assert all(
            isinstance(pd, float) for pd in merton_pds
        ), "Merton PDs should be floats"

    # Check that portfolio values are numeric
    for value in portfolio_values_independent_basins:
        assert isinstance(value, (int, float)), "Portfolio values should be numeric"
