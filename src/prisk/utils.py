import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go

def convert_to_continous_damage(damage_curves):
    continuous_curves = pd.DataFrame({"index": range(0, int(max((damage_curves["depth"] + 0.01)*100)))})
    continuous_curves["index"] = continuous_curves["index"]/100
    continuous_curves.set_index("index", inplace=True)
    continuous_curves = continuous_curves.merge(damage_curves, how="left", left_index=True, right_on="depth")
    continuous_curves.interpolate(method="linear", inplace=True)
    continuous_curves.set_index("depth", inplace=True)
    return continuous_curves


def bootstrap_mean(data, column_name, samples=100):
    bootstrapped = data.sample(n=samples, replace=True)
    return bootstrapped[column_name].mean()

def plot_bootstrap(data, column_name):
    bootstraps = [bootstrap_mean(data, column_name, samples=100) for i in range(10000)]
    bootstraps = np.array(bootstraps)
    plt.hist(bootstraps, bins=50, density=True)
    plt.axvline(np.quantile(bootstraps, 0.01), color="red")
    plt.axvline(np.quantile(bootstraps, 0.99), color="red");
    print("Width of CI: ", round(np.quantile(bootstraps, 0.95) - np.quantile(bootstraps, 0.05), 4))
    print("Mean of CI:  ", round(np.mean(bootstraps), 4))
    print("Std of CI:   ", round(np.std(bootstraps), 4))
    print("Q1:          ", round(np.quantile(bootstraps, 0.05), 4))
    print("Q99:         ", round(np.quantile(bootstraps, 0.95), 4))
    print("Skewness:    ", round(pd.Series(bootstraps).skew(), 4))
    print("Kurtosis:    ", round(pd.Series(bootstraps).kurtosis(), 4))


def plot_risk_factors(base_value, capital_damages, business_disruption, fair_premium, insurance_adjustment, npv):
    fig = go.Figure(go.Waterfall(
        name = "PRISK - Waterfall", 
        orientation = "v",
        measure = ["relative", "relative", "relative", "relative", "relative", "total"],
        x = ["Base Value", "Capital Damages", "Business disruptions", "Fair insurance premiums",
                "Insurance adjustments", "Adjusted Value"],
        textposition = "outside",
        text = ["{:,.2f}M".format(base_value/1e6), 
                "{:,.2f}M".format(capital_damages/1e6), 
                "{:,.2f}M".format(business_disruption/1e6), 
                "{:,.2f}M".format(fair_premium/1e6),
                "{:,.2f}M".format(abs(insurance_adjustment)/1e6),
                "{:,.2f}M".format(npv/1e6)],
        y = [base_value, 
            -capital_damages,
            -business_disruption,
            -fair_premium, 
            -insurance_adjustment,
            npv],
        connector = {"line":{"color":"rgb(63, 63, 63)"}},
    ))

    fig.update_layout(
            title = "PRISK - Waterfall chart",
            showlegend = False,
            template="simple_white",
            margin=dict(l=20, r=20, t=40, b=20),
            xaxis_title="Asset value impacts",
            yaxis_title="Value",
            yaxis_tickprefix="$",
            yaxis_tickformat=",",
            yaxis_showgrid=True,
    )
    return fig