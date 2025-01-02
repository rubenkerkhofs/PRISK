import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_assets_and_basins(assets_with_basins, basins_processed, save=False):
    basins_processed.plot(color=basins_processed.color, figsize=(20, 20))
    plt.scatter(
        assets_with_basins.Longitude, assets_with_basins.Latitude, c="red", s=50
    )
    if save:
        plt.savefig("map.png", transparent=True)


def compare_value_distributions(
    data_dict,
    bins=20,
    colors=None,
    labels=None,
    alpha=0.5,
    density=True,
    quantiles=None,
    xlim=None,
    xlabel="Portfolio value",
    title="Portfolio Value Distribution",
    hide_yaxis=True,
    font_size=14,
    font_family="Times New Roman",
    save_path=None,
):
    """
    Compare portfolio value distributions using histograms.

    Parameters:
    - data_dict: Dictionary where keys are labels and values are data series.
    - bins: Number of bins for the histogram.
    - colors: List of colors for each histogram.
    - labels: List of labels for the legend.
    - alpha: Transparency level for histograms.
    - density: Whether to normalize the histogram.
    - quantiles: Dictionary of quantiles to plot as vertical lines {label: value}.
    - xlim: Tuple specifying the x-axis limits.
    - xlabel: Label for the x-axis.
    - title: Title of the plot.
    - hide_yaxis: Boolean to hide the y-axis.
    - font_size: Size of the font for labels and title.
    - font_family: Font family for labels and title.
    - save_path: Path to save the plot. If None, the plot is not saved.
    """
    plt.figure(figsize=(20, 5))

    for idx, (key, data) in enumerate(data_dict.items()):
        plt.hist(
            data,
            bins=bins,
            color=colors[idx] if colors else None,
            label=labels[idx] if labels else key,
            alpha=alpha,
            density=density,
        )

    if quantiles:
        for key, value in quantiles.items():
            plt.axvline(
                value,
                color=colors[list(data_dict.keys()).index(key)] if colors else None,
                linestyle=":",
                label=f"{key} {int(list(quantiles.keys()).index(key)+1)}% quantile",
            )

    plt.xlabel(xlabel, fontsize=font_size, family=font_family)
    plt.title(title, fontsize=font_size, family=font_family)
    if hide_yaxis:
        plt.gca().axes.get_yaxis().set_visible(False)
    if xlim:
        plt.xlim(xlim)
    plt.legend(prop={"size": font_size, "family": font_family})
    if save_path:
        plt.savefig(save_path, transparent=True)
    plt.show()


def visualize_dependency_models(
    data_list,
    labels,
    colors,
    quantiles_dict,
    ylabel="Portfolio value",
    title="Dependency Model Comparison",
    ylim=(0.96, 1),
    font_size=14,
    font_family="Times New Roman",
    save_path=None,
):
    """
    Visualize portfolio value distributions across dependency models with violin plots.

    Parameters:
    - data_list: List of data series.
    - labels: List of labels for each dataset.
    - colors: List of colors for each violin.
    - quantiles_dict: Dictionary where keys are labels and values are tuples (Q1, lower_bound, upper_bound).
    - ylabel: Label for the y-axis.
    - title: Title of the plot.
    - ylim: Tuple specifying the y-axis limits.
    - font_size: Size of the font for labels and title.
    - font_family: Font family for labels and title.
    - save_path: Path to save the plot. If None, the plot is not saved.
    """
    plt.figure(figsize=(20, 5))
    vp = plt.violinplot(data_list, showmeans=False, showextrema=False, bw_method=0.4)

    for i, pc in enumerate(vp["bodies"]):
        pc.set_facecolor(colors[i])
        pc.set_edgecolor(colors[i])
        pc.set_alpha(0.5)

    for i, label in enumerate(labels):
        q1, lb, up = quantiles_dict[label]
        plt.hlines(
            q1, i + 0.75, i + 1.25, color=colors[i], label=f"{label} - 1% quantile"
        )
        plt.hlines(lb, i + 0.75, i + 1.25, color=colors[i], linestyle=":")
        plt.hlines(up, i + 0.75, i + 1.25, color=colors[i], linestyle=":")

    plt.ylabel(ylabel, fontsize=font_size, family=font_family)
    plt.title(title, fontsize=font_size, family=font_family)
    plt.ylim(ylim)
    plt.legend(loc="lower left", prop={"size": font_size, "family": font_family})
    plt.xticks(ticks=range(1, len(labels) + 1), labels=labels, fontfamily=font_family)
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)

    if save_path:
        plt.savefig(save_path, transparent=True)
    plt.show()


def analyze_bootstrap_stability(
    m_values,
    means,
    variances,
    title="Bootstrap Stability Analysis",
    xlabel="Subsampling Ratio (m)",
    ylabel="Bootstrap Sample Mean",
    font_size=14,
    font_family="Times New Roman",
    save_path=None,
):
    """
    Analyze bootstrap stability by plotting means and variances over different sampling ratios.

    Parameters:
    - m_values: Array of m values used in bootstrap.
    - means: List of mean values corresponding to m_values.
    - variances: List of variance values corresponding to m_values.
    - title: Title of the plot.
    - xlabel: Label for the x-axis.
    - ylabel: Label for the y-axis.
    - font_size: Size of the font for labels and title.
    - font_family: Font family for labels and title.
    - save_path: Path to save the plot. If None, the plot is not saved.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(m_values, means, label="Mean", color="blue")
    plt.fill_between(
        m_values,
        np.array(means) - np.array(variances),
        np.array(means) + np.array(variances),
        color="blue",
        alpha=0.2,
        label="Variance",
    )
    plt.title(title, fontsize=font_size, family=font_family)
    plt.xlabel(xlabel, fontsize=font_size, family=font_family)
    plt.ylabel(ylabel, fontsize=font_size, family=font_family)
    plt.legend(fontsize=font_size, family=font_family)
    if save_path:
        plt.savefig(save_path, transparent=True)
    plt.show()


def compare_risk_models(
    empirical_pds,
    merton_pds,
    title_scatter="Empirical vs Merton ΔPD (Log Scale)",
    title_line="Empirical vs Merton ΔPD (Over Time)",
    xlabel_scatter="Log ΔPD (Empirical)",
    ylabel_scatter="Log ΔPD (Merton)",
    xlabel_line="Iteration",
    ylabel_line="ΔPD",
    font_size=14,
    font_family="Times New Roman",
    save_scatter=None,
    save_line=None,
):
    """
    Compare empirical and Merton risk models with scatter and line plots.

    Parameters:
    - empirical_pds: List or array of empirical PDs.
    - merton_pds: List or array of Merton model PDs.
    - title_scatter: Title for the scatter plot.
    - title_line: Title for the line plot.
    - xlabel_scatter: X-axis label for scatter plot.
    - ylabel_scatter: Y-axis label for scatter plot.
    - xlabel_line: X-axis label for line plot.
    - ylabel_line: Y-axis label for line plot.
    - font_size: Size of the font for labels and titles.
    - font_family: Font family for labels and titles.
    - save_scatter: Path to save the scatter plot. If None, not saved.
    - save_line: Path to save the line plot. If None, not saved.
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(np.log(empirical_pds), np.log(merton_pds), alpha=0.7)
    plt.xlabel(xlabel_scatter, fontsize=font_size, family=font_family)
    plt.ylabel(ylabel_scatter, fontsize=font_size, family=font_family)
    plt.title(title_scatter, fontsize=font_size, family=font_family)
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    if save_scatter:
        plt.savefig(save_scatter, transparent=True)
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(merton_pds, label="Merton Model", alpha=0.9)
    plt.plot(empirical_pds, label="Empirical Model", alpha=0.7)
    plt.xlabel(xlabel_line, fontsize=font_size, family=font_family)
    plt.ylabel(ylabel_line, fontsize=font_size, family=font_family)
    plt.title(title_line, fontsize=font_size, family=font_family)
    plt.legend(fontsize=font_size, family=font_family)
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    if save_line:
        plt.savefig(save_line, transparent=True)
    plt.show()


def generate_summary_table(
    average_impacts, median_impacts, q5, q1, q5_interval, q1_interval, index_labels
):
    """
    Generate a summary table of impacts and quantile intervals.

    Parameters:
    - average_impacts: List of average impacts.
    - median_impacts: List of median impacts.
    - q5: List of 5% quantiles.
    - q1: List of 1% quantiles.
    - q5_interval: List of 5% quantile intervals as strings.
    - q1_interval: List of 1% quantile intervals as strings.
    - index_labels: List of index labels for the table.

    Returns:
    - pandas DataFrame representing the summary table.
    """
    table = pd.DataFrame(
        {
            "Average Impact (%)": average_impacts,
            "Median Impact (%)": median_impacts,
            "Q5 (%)": q5,
            "Q1 (%)": q1,
            "Q5 Interval": q5_interval,
            "Q1 Interval": q1_interval,
        },
        index=index_labels,
    )
    return table
