"""
Streamlit dashboard for monitoring Telco customer churn.

This app expects the Telco Customer Churn CSV (WA_Fn-UseC_-Telco-Customer-Churn.csv
from the original notebook) to be available locally. Point `data_path` in the
sidebar to wherever the file lives in your environment.
"""

from __future__ import annotations

import pathlib
from typing import Iterable, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


st.set_page_config(
    page_title="Telco Retention Command Center",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    return df


def safe_multiselect(
    label: str, options: Iterable[str], default: Iterable[str] | None = None
) -> list[str]:
    clean_default = list(default) if default else list(options)
    return (
        st.sidebar.multiselect(label, options=list(options), default=clean_default)
        if options
        else []
    )


def filter_frame(
    df: pd.DataFrame,
    contracts: list[str],
    services: list[str],
    payments: list[str],
    tenure_range: Tuple[int, int],
    charge_range: Tuple[float, float],
) -> pd.DataFrame:
    masks = [
        df["Contract"].isin(contracts) if contracts else True,
        df["InternetService"].isin(services) if services else True,
        df["PaymentMethod"].isin(payments) if payments else True,
        df["tenure"].between(*tenure_range),
        df["MonthlyCharges"].between(*charge_range),
    ]
    combined = np.logical_and.reduce(masks)
    return df.loc[combined].copy()


def kpi_row(df: pd.DataFrame) -> None:
    total = len(df)
    churned = (df["Churn"] == "Yes").sum()
    churn_rate = (churned / total * 100) if total else 0
    monthly_revenue = df["MonthlyCharges"].sum()
    revenue_at_risk = df.loc[df["Churn"] == "Yes", "MonthlyCharges"].sum()
    avg_tenure = df["tenure"].mean()

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Customers", f"{total:,}")
    col2.metric("Churned", f"{churned:,}", f"{churn_rate:0.1f}%")
    col3.metric("Avg tenure (mo)", f"{avg_tenure:0.1f}")
    col4.metric("Monthly revenue", f"${monthly_revenue:,.0f}")
    col5.metric("Revenue at risk", f"${revenue_at_risk:,.0f}")


def churn_by_category(df: pd.DataFrame, column: str, title: str) -> None:
    counts = (
        df.groupby([column, "Churn"])
        .size()
        .reset_index(name="Customers")
        .sort_values("Customers", ascending=False)
    )
    if counts.empty:
        st.info(f"No data for {title.lower()}.")
        return

    fig = px.bar(
        counts,
        x=column,
        y="Customers",
        color="Churn",
        barmode="group",
        title=title,
        color_discrete_map={"Yes": "#e45756", "No": "#4c78a8"},
    )
    fig.update_layout(legend_title_text="")
    st.plotly_chart(fig, use_container_width=True, theme="streamlit")


def tenure_distribution(df: pd.DataFrame) -> None:
    fig = px.histogram(
        df,
        x="tenure",
        color="Churn",
        nbins=30,
        title="Tenure distribution",
        color_discrete_map={"Yes": "#e45756", "No": "#4c78a8"},
    )
    fig.update_layout(legend_title_text="")
    st.plotly_chart(fig, use_container_width=True, theme="streamlit")


def charges_vs_churn(df: pd.DataFrame) -> None:
    fig = px.box(
        df,
        x="Churn",
        y="MonthlyCharges",
        color="Churn",
        title="Monthly charges vs churn",
        color_discrete_map={"Yes": "#e45756", "No": "#4c78a8"},
    )
    fig.update_layout(legend_title_text="")
    st.plotly_chart(fig, use_container_width=True, theme="streamlit")


def risk_table(df: pd.DataFrame) -> None:
    risk = (
        df.loc[df["Churn"] == "Yes"]
        .assign(
            tenure_bucket=lambda d: pd.cut(
                d["tenure"], bins=[0, 6, 12, 24, 48, np.inf], labels=False
            )
        )
        .sort_values(["MonthlyCharges", "tenure"], ascending=[False, True])
        [
            [
                "customerID",
                "tenure",
                "Contract",
                "InternetService",
                "PaymentMethod",
                "MonthlyCharges",
                "TotalCharges",
            ]
        ]
        .head(30)
    )
    st.subheader("Accounts most at risk (top 30 by monthly charges)")
    st.dataframe(risk, use_container_width=True, height=500)


def show_existing_visuals(base_path: pathlib.Path) -> None:
    images = {
        "Confusion matrix": base_path / "ConfusionMatrix.png",
        "Top 10 feature importance": base_path / "top10MostImportantFeatures.png",
        "Churn by contract type": base_path / "ChurnBYContract_type.png",
    }
    with st.expander("Existing visuals from previous analysis"):
        cols = st.columns(3)
        for col, (title, path) in zip(cols, images.items()):
            if path.exists():
                col.image(str(path), caption=title, use_column_width=True)
            else:
                col.info(f"{title} (missing: {path.name})")


def main() -> None:
    st.title("Telco Retention Command Center")
    st.markdown(
        "Monitor churn trends, surface at-risk accounts, and explore the drivers "
        "behind customer departures. Use the sidebar to point to the Telco churn "
        "CSV and filter by contract, services, tenure, and payment method."
    )

    default_path = "/Users/ismaelzerrouk/.cache/kagglehub/datasets/blastchar/telco-customer-churn/versions/1/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    data_path = st.sidebar.text_input("Data path", value=default_path)

    try:
        df = load_data(data_path)
    except FileNotFoundError:
        st.error(f"Could not find file at '{data_path}'.")
        st.stop()
    except Exception as exc:  # pragma: no cover - surfaced in UI
        st.error(f"Could not load data: {exc}")
        st.stop()

    st.sidebar.header("Filters")
    contract_opts = sorted(df["Contract"].dropna().unique().tolist())
    service_opts = sorted(df["InternetService"].dropna().unique().tolist())
    payment_opts = sorted(df["PaymentMethod"].dropna().unique().tolist())

    contract_sel = safe_multiselect("Contract type", contract_opts, contract_opts)
    service_sel = safe_multiselect("Internet service", service_opts, service_opts)
    payment_sel = safe_multiselect("Payment method", payment_opts, payment_opts)

    tenure_min, tenure_max = int(df["tenure"].min()), int(df["tenure"].max())
    charges_min, charges_max = (
        float(df["MonthlyCharges"].min()),
        float(df["MonthlyCharges"].max()),
    )

    tenure_range = st.sidebar.slider(
        "Tenure (months)", min_value=tenure_min, max_value=tenure_max, value=(tenure_min, tenure_max)
    )
    charge_range = st.sidebar.slider(
        "Monthly charges", min_value=charges_min, max_value=charges_max, value=(charges_min, charges_max)
    )

    filtered = filter_frame(
        df,
        contracts=contract_sel,
        services=service_sel,
        payments=payment_sel,
        tenure_range=tenure_range,
        charge_range=charge_range,
    )

    st.caption(f"Filtered view: {len(filtered):,} customers")
    kpi_row(filtered)

    col_a, col_b = st.columns(2)
    with col_a:
        churn_by_category(filtered, "Contract", "Churn by contract")
        churn_by_category(filtered, "InternetService", "Churn by internet service")
    with col_b:
        tenure_distribution(filtered)
        charges_vs_churn(filtered)

    risk_table(filtered)
    show_existing_visuals(base_path=pathlib.Path("."))


if __name__ == "__main__":
    main()
