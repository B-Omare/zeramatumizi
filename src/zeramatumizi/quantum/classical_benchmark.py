"""
classical_benchmark.py
Classical linear programming resource allocation benchmark
for ZeraMatumizi D6.

Solves the NACADA treatment resource allocation problem using
classical linear programming (PuLP) as a benchmark for comparison
with the QAOA quantum approach.

Problem:
Given scarce treatment resources (residential slots, community
counsellors, outreach teams), allocate optimally across Kenyan
counties to maximise lives saved per shilling, subject to:
- Total resource budget constraints
- Minimum allocation per county
- Equity constraints (no county left with zero resources)
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pulp

REPORTS_PATH = os.path.join("docs", "reports")


def ensure_directories():
    os.makedirs(REPORTS_PATH, exist_ok=True)


def load_county_need_scores() -> pd.DataFrame:
    """
    Loads county risk profiles and computes a composite need score
    that drives the optimisation objective.

    Need score combines:
    - Disorder progression rate (primary driver)
    - Unemployment rate (proxy for treatment-seeking barriers)
    - Facility distance (proxy for access gap)
    - Population size (to compute absolute burden)
    """
    print("Loading county need scores...")

    try:
        df = pd.read_parquet("data/raw/kdhs_sample.parquet")
    except FileNotFoundError:
        print("KDHS data not found - run loader.py first")
        raise

    df["unemployed_num"] = (df["employment_status"] == "unemployed").astype(int)
    df["hiv_positive_num"] = (df["hiv_status"] == "positive").astype(int)
    df["polysubstance"] = (
        (df["alcohol_use"] + df["cannabis_use"] + df["khat_use"]) >= 2
    ).astype(int)

    county_profiles = df.groupby("county").agg(
        n_respondents=("respondent_id", "count"),
        disorder_rate=("disorder_progression", "mean"),
        unemployment_rate=("unemployed_num", "mean"),
        hiv_rate=("hiv_positive_num", "mean"),
        polysubstance_rate=("polysubstance", "mean"),
        alcohol_rate=("alcohol_use", "mean"),
    ).round(4)

    # Simulate county population (Kenya 2022 estimates scaled to our counties)
    county_populations = {
        "Kisumu": 1155600, "Kakamega": 1867579, "Siaya": 993183,
        "Homa Bay": 1131950, "Migori": 1116436, "Nairobi": 4397073,
        "Mombasa": 1208333, "Nakuru": 2162202, "Kisii": 1266860,
        "Nyamira": 598252,
    }
    county_profiles["population"] = county_profiles.index.map(
        lambda c: county_populations.get(c, 500000)
    )

    # Simulate facility distance (km to nearest rehab centre)
    np.random.seed(42)
    county_profiles["facility_distance_km"] = np.random.uniform(10, 120, len(county_profiles))

    # Composite need score (0-1 normalised)
    county_profiles["need_score"] = (
        0.40 * county_profiles["disorder_rate"] +
        0.20 * county_profiles["unemployment_rate"] +
        0.15 * county_profiles["hiv_rate"] +
        0.15 * county_profiles["polysubstance_rate"] +
        0.10 * (county_profiles["facility_distance_km"] / 120)
    )

    # Normalise to 0-1
    county_profiles["need_score"] = (
        county_profiles["need_score"] - county_profiles["need_score"].min()
    ) / (county_profiles["need_score"].max() - county_profiles["need_score"].min())

    # Absolute burden = need score * population
    county_profiles["absolute_burden"] = (
        county_profiles["need_score"] * county_profiles["population"]
    ).astype(int)

    county_profiles = county_profiles.sort_values("need_score", ascending=False)

    print(f"County need scores computed for {len(county_profiles)} counties")
    print(f"\n  Top 5 highest-need counties:")
    for county, row in county_profiles.head(5).iterrows():
        print(f"    {county:<15}: need={row['need_score']:.3f}, "
              f"disorder={row['disorder_rate']:.1%}, "
              f"burden={row['absolute_burden']:,}")

    return county_profiles


def define_resources() -> dict:
    """
    Defines Kenya's available treatment resource pool for allocation.
    Based on NACADA 2022 annual report estimates.
    """
    return {
        "residential_slots": {
            "total": 500,
            "min_per_county": 5,
            "cost_per_unit": 45000,   # KES per month
            "impact_weight": 1.0,
        },
        "community_counsellors": {
            "total": 120,
            "min_per_county": 1,
            "cost_per_unit": 35000,   # KES per month salary
            "impact_weight": 0.7,
        },
        "outreach_teams": {
            "total": 30,
            "min_per_county": 0,
            "cost_per_unit": 80000,   # KES per month (vehicle + team)
            "impact_weight": 0.5,
        },
    }


def solve_classical_lp(
    county_profiles: pd.DataFrame,
    resources: dict
) -> dict:
    """
    Solves the resource allocation problem using classical linear
    programming (PuLP). Maximises weighted impact across counties
    subject to budget and equity constraints.
    """
    print("\nSolving classical LP resource allocation...")

    counties = county_profiles.index.tolist()
    n = len(counties)
    need_scores = county_profiles["need_score"].values

    results = {}

    for resource_name, resource_config in resources.items():
        total = resource_config["total"]
        min_per = resource_config["min_per_county"]
        impact = resource_config["impact_weight"]

        prob = pulp.LpProblem(
            f"NACADA_{resource_name}_allocation",
            pulp.LpMaximize
        )

        # Decision variables: units allocated to each county
        x = {
            county: pulp.LpVariable(
                f"x_{county.replace(' ', '_')}",
                lowBound=min_per,
                upBound=total,
                cat="Integer"
            )
            for county in counties
        }

        # Objective: maximise need-weighted allocation
        prob += pulp.lpSum(
            need_scores[i] * impact * x[county]
            for i, county in enumerate(counties)
        )

        # Constraint: total allocation cannot exceed available resources
        prob += pulp.lpSum(x[county] for county in counties) <= total

        # Equity constraint: higher-need counties get proportionally more
        for i, county in enumerate(counties):
            if i < n - 1:
                if need_scores[i] > need_scores[i + 1] + 0.05:
                    prob += x[county] >= x[counties[i + 1]]

        prob.solve(pulp.PULP_CBC_CMD(msg=0))

        county_allocations = {
            county: int(x[county].value() or min_per)
            for county in counties
        }
        total_allocated = sum(county_allocations.values())
        results[resource_name] = {
            "allocations": county_allocations,
            "total_allocated": total_allocated,
            "status": pulp.LpStatus[prob.status],
            "objective": pulp.value(prob.objective),
        }

        print(f"  {resource_name}: {pulp.LpStatus[prob.status]}, "
              f"allocated {total_allocated}/{total}")

    return results


def print_allocation_table(
    county_profiles: pd.DataFrame,
    lp_results: dict
):
    """Prints a formatted allocation table for all counties."""
    print("\n--- Classical LP Resource Allocation Results ---")
    print(f"\n  {'County':<15} {'Need':>6} {'Slots':>7} "
          f"{'Counsellors':>12} {'Teams':>7}")
    print(f"  {'-'*50}")

    for county in county_profiles.index:
        need = county_profiles.loc[county, "need_score"]
        slots = lp_results["residential_slots"]["allocations"].get(county, 0)
        counsellors = lp_results["community_counsellors"]["allocations"].get(county, 0)
        teams = lp_results["outreach_teams"]["allocations"].get(county, 0)
        print(f"  {county:<15} {need:>6.3f} {slots:>7} {counsellors:>12} {teams:>7}")

    print(f"\n  Totals:")
    for resource, result in lp_results.items():
        print(f"    {resource}: {result['total_allocated']} allocated")
    print("------------------------------------------------\n")


def save_classical_allocation_plot(
    county_profiles: pd.DataFrame,
    lp_results: dict
):
    """Saves a visualisation of the classical LP allocation."""
    ensure_directories()

    counties = county_profiles.index.tolist()
    need_scores = county_profiles["need_score"].values

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        "ZeraMatumizi - Classical LP Resource Allocation\n"
        "NACADA Treatment Resources: Optimal Distribution",
        fontsize=13, fontweight="bold"
    )

    resource_names = list(lp_results.keys())
    colours = ["#E74C3C", "#3498DB", "#27AE60"]

    for ax, resource_name, colour in zip(axes, resource_names, colours):
        allocations = [
            lp_results[resource_name]["allocations"].get(c, 0)
            for c in counties
        ]

        sorted_idx = np.argsort(allocations)[::-1]
        sorted_counties = [counties[i] for i in sorted_idx]
        sorted_allocations = [allocations[i] for i in sorted_idx]
        sorted_needs = [need_scores[i] for i in sorted_idx]

        bars = ax.bar(
            range(len(counties)), sorted_allocations,
            color=[colour] * len(counties), edgecolor="white", alpha=0.85
        )

        ax.set_xticks(range(len(counties)))
        ax.set_xticklabels(sorted_counties, rotation=45, ha="right", fontsize=8)
        ax.set_title(
            f"{resource_name.replace('_', ' ').title()}\n"
            f"(Total: {lp_results[resource_name]['total_allocated']})",
            fontweight="bold"
        )
        ax.set_ylabel("Units Allocated")
        ax.grid(True, alpha=0.3, axis="y")

        for bar, val in zip(bars, sorted_allocations):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.2,
                str(val), ha="center", va="bottom", fontsize=7
            )

    plt.tight_layout()
    output_path = os.path.join(REPORTS_PATH, "classical_allocation.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Classical allocation plot saved: {output_path}")

    return lp_results


if __name__ == "__main__":
    print("=" * 60)
    print("ZeraMatumizi - Classical LP Resource Allocation Benchmark")
    print("=" * 60)

    ensure_directories()

    county_profiles = load_county_need_scores()
    resources = define_resources()

    lp_results = solve_classical_lp(county_profiles, resources)
    print_allocation_table(county_profiles, lp_results)
    save_classical_allocation_plot(county_profiles, lp_results)

    print("Classical LP benchmark complete!")