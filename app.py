"""Interactive Streamlit dashboard for Ebola outbreak surveillance and SIR scenarios."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from epi_model import SIRParams, fit_sir, model_metrics, simulate_sir

# ---------------------------------------------------------------------------
# Configuration and visual constants
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

COLORS = {
    "teal": "#2dd4bf",
    "orange": "#fb923c",
    "coral": "#f87171",
    "blue": "#60a5fa",
    "violet": "#a78bfa",
    "amber": "#fbbf24",
    "muted": "#94a3b8",
    "ink": "#e8eef6",
}
PALETTE = [
    COLORS["teal"],
    COLORS["orange"],
    COLORS["blue"],
    COLORS["violet"],
    COLORS["coral"],
    COLORS["amber"],
    "#34d399",
    "#f472b6",
    "#38bdf8",
    "#c084fc",
]
POPULATIONS = {
    "Guinea": 13_270_000,
    "Liberia": 4_500_000,
    "Sierra Leone": 7_079_000,
    "Nigeria": 178_500_000,
    "Mali": 17_100_000,
    "Senegal": 14_700_000,
    "Italy": 60_800_000,
    "Spain": 46_500_000,
    "United Kingdom": 65_600_000,
    "United States of America": 318_900_000,
}
DEFAULT_COUNTRIES = ["Guinea", "Liberia", "Sierra Leone"]
DEFAULT_DRC_POPULATION = 109_000_000

# ---------------------------------------------------------------------------
# Page setup and custom styling
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Ebola Epi Watch",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 2.1rem; padding-bottom: 3rem; max-width: 1420px; }
    .hero {
        border: 1px solid #D2E3FC;
        background: linear-gradient(135deg, #E8F0FE 0%, #F1F6FD 100%);
        border-radius: 16px;
        padding: 26px 30px 24px 30px;
        margin-bottom: 16px;
    }
    .hero .kicker {
        color: #185ABC;
        font-size: 0.76rem;
        font-weight: 700;
        letter-spacing: 0.16em;
        text-transform: uppercase;
    }
    .hero h1 {
        margin: 0.28rem 0 0.35rem 0;
        font-size: 2.25rem;
        letter-spacing: -0.02em;
        color: #0D47A1;
        font-weight: 700;
    }
    .hero p {
        color: #174EA6;
        max-width: 860px;
        margin: 0;
        font-size: 1rem;
        line-height: 1.55;
    }
    .source-strip { display: flex; flex-wrap: wrap; gap: 8px; margin: 0 0 18px 2px; }
    .pill {
        border: 1px solid #D2E3FC;
        background: #E8F0FE;
        color: #174EA6;
        padding: 4px 11px;
        border-radius: 999px;
        font-size: 0.74rem;
        font-weight: 500;
    }
    div[data-testid="stMetric"] {
        padding: 4px 6px;
    }
    div[data-testid="stMetricValue"] { font-size: 1.62rem; }
    .section-copy { color: #5F6368; font-size: 0.88rem; margin: -2px 0 10px 0; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Cached data access
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def load_table(path_str: str, signature: int) -> pd.DataFrame:
    """Read a normalized CSV and parse its date column when present."""
    del signature
    frame = pd.read_csv(path_str, low_memory=False)
    if "date" in frame.columns:
        frame["date"] = pd.to_datetime(frame["date"])
    return frame


@st.cache_data(show_spinner=False)
def load_metadata_file(path_str: str, signature: int) -> dict:
    """Read ingestion metadata, returning an empty mapping when unavailable."""
    del signature
    path = Path(path_str)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def baseline_fit(
    cumulative: tuple[float, ...],
    population: float,
    initial_cases: float,
    initial_infectious: float,
) -> tuple[float, float, float]:
    """Estimate dataset-specific starting values for the beta and gamma sliders."""
    values = np.asarray(cumulative, dtype=float)
    fit = fit_sir(
        values,
        np.arange(len(values), dtype=float),
        population,
        initial_cases,
        initial_infectious,
    )
    return fit.beta, fit.gamma, fit.rmse


def file_signature(path: Path) -> int:
    """Return a cache-busting signature based on file size and modification time."""
    stat = path.stat()
    return stat.st_mtime_ns + stat.st_size


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Load all normalized tables and fail with a clear recovery instruction."""
    west_path = DATA_DIR / "ebola_west_africa.csv"
    drc_path = DATA_DIR / "ebola_drc_2026.csv"
    province_path = DATA_DIR / "ebola_drc_subnational.csv"
    metadata_path = DATA_DIR / "ingestion_metadata.json"
    missing = [path.name for path in [west_path, drc_path, province_path] if not path.exists()]
    if missing:
        st.error(
            "Normalized data is missing: "
            + ", ".join(missing)
            + ". Run `python ingest.py` in the project folder, then reload."
        )
        st.stop()
    west = load_table(str(west_path), file_signature(west_path))
    drc = load_table(str(drc_path), file_signature(drc_path))
    provinces = load_table(str(province_path), file_signature(province_path))
    metadata = (
        load_metadata_file(str(metadata_path), file_signature(metadata_path))
        if metadata_path.exists()
        else {}
    )
    return west, drc, provinces, metadata


# ---------------------------------------------------------------------------
# Formatting and chart helpers
# ---------------------------------------------------------------------------


def human_int(value: float) -> str:
    """Format large counts compactly for KPI cards."""
    value = float(value)
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:,.2f}M"
    if abs(value) >= 1_000:
        return f"{value / 1_000:,.1f}k"
    return f"{value:,.0f}"


def pct(value: float) -> str:
    """Format a proportion as a one-decimal percentage."""
    return f"{value * 100:,.1f}%"


def latest_change(series: pd.Series) -> float | None:
    """Compare the latest seven days with the preceding seven days."""
    last_week = float(series.tail(7).sum())
    prior_week = float(series.iloc[-14:-7].sum()) if len(series) >= 14 else 0.0
    if prior_week <= 0:
        return None
    return (last_week - prior_week) / prior_week


def style_figure(
    fig: go.Figure,
    height: int = 410,
    y_title: str | None = None,
    x_title: str | None = None,
    unified_hover: bool = True,
) -> go.Figure:
    """Apply the dashboard's shared dark Plotly layout."""
    fig.update_layout(
        template="plotly_dark",
        height=height,
        margin=dict(l=8, r=8, t=18, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Segoe UI, Inter, sans-serif", size=13, color="#dbe4ee"),
        hovermode="x unified" if unified_hover else "closest",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            x=0,
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=11),
        ),
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.07)", zeroline=False)
    if y_title:
        fig.update_yaxes(title_text=y_title)
    if x_title:
        fig.update_xaxes(title_text=x_title)
    return fig


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------


def build_west_view(
    west: pd.DataFrame, countries: list[str], start: pd.Timestamp, end: pd.Timestamp
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate selected West African countries into one analysis window."""
    subset = west[west["country"].isin(countries)]
    case_matrix = subset.pivot(index="date", columns="country", values="cases").sort_index()
    death_matrix = subset.pivot(index="date", columns="country", values="deaths").sort_index()

    # Fill after a country's final report, but treat dates before its first
    # report as zero rather than carrying the first value backwards.
    case_matrix = case_matrix.ffill().fillna(0.0)
    death_matrix = death_matrix.reindex(case_matrix.index).ffill().fillna(0.0)
    combined = pd.DataFrame(
        {"cases": case_matrix.sum(axis=1), "deaths": death_matrix.sum(axis=1)}
    )
    combined = combined.loc[(combined.index >= start) & (combined.index <= end)]
    combined["new_cases"] = combined["cases"].diff().fillna(0.0).clip(lower=0.0)
    combined["new_deaths"] = combined["deaths"].diff().fillna(0.0).clip(lower=0.0)
    combined["new_cases_7d"] = combined["new_cases"].rolling(7, min_periods=1).mean()
    country_matrix = case_matrix.loc[
        (case_matrix.index >= start) & (case_matrix.index <= end)
    ]
    return combined, country_matrix


def build_drc_view(
    drc: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp
) -> pd.DataFrame:
    """Prepare the national DRC series for the selected analysis window."""
    view = drc.set_index("date")[["cases", "deaths", "suspected_cases"]].sort_index()
    view = view.loc[(view.index >= start) & (view.index <= end)].copy()
    view["new_cases"] = view["cases"].diff().fillna(0.0).clip(lower=0.0)
    view["new_deaths"] = view["deaths"].diff().fillna(0.0).clip(lower=0.0)
    view["new_cases_7d"] = view["new_cases"].rolling(7, min_periods=1).mean()
    return view


# ---------------------------------------------------------------------------
# Chart builders
# ---------------------------------------------------------------------------


def cumulative_chart(view: pd.DataFrame, log_scale: bool, case_label: str) -> go.Figure:
    """Plot cumulative reported cases and deaths."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=view.index,
            y=view["cases"],
            name=case_label,
            mode="lines",
            line=dict(color=COLORS["teal"], width=3),
            fill="tozeroy",
            fillcolor="rgba(45, 212, 191, 0.08)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=view.index,
            y=view["deaths"],
            name="Cumulative deaths",
            mode="lines",
            line=dict(color=COLORS["coral"], width=2, dash="dot"),
        )
    )
    style_figure(fig, y_title="People", unified_hover=True)
    if log_scale:
        fig.update_yaxes(type="log", tickformat=",.0f", title_text="People (log scale)")
    else:
        fig.update_yaxes(tickformat=",.0f")
    return fig


def incidence_chart(view: pd.DataFrame, case_label: str) -> go.Figure:
    """Plot daily reported cases, the 7-day average, and smoothed deaths."""
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=view.index,
            y=view["new_cases"],
            name=f"New {case_label.lower()}",
            marker_color="rgba(251, 146, 60, 0.38)",
            hovertemplate="%{y:,.1f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=view.index,
            y=view["new_cases_7d"],
            name="7-day average",
            mode="lines",
            line=dict(color=COLORS["orange"], width=2.6),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=view.index,
            y=view["new_deaths"].rolling(7, min_periods=1).mean(),
            name="Deaths (7-day average)",
            mode="lines",
            line=dict(color=COLORS["coral"], width=1.6, dash="dot"),
        )
    )
    style_figure(fig, y_title="New reports per day")
    if view["new_cases"].max() > 0:
        fig.update_yaxes(rangemode="tozero")
    return fig


def country_chart(country_matrix: pd.DataFrame, combined: pd.DataFrame) -> go.Figure:
    """Compare individual country trajectories with the combined total."""
    fig = go.Figure()
    for index, country in enumerate(country_matrix.columns):
        fig.add_trace(
            go.Scatter(
                x=country_matrix.index,
                y=country_matrix[country],
                name=country,
                mode="lines",
                line=dict(color=PALETTE[index % len(PALETTE)], width=2),
            )
        )
    fig.add_trace(
        go.Scatter(
            x=combined.index,
            y=combined["cases"],
            name="Combined",
            mode="lines",
            line=dict(color=COLORS["ink"], width=2.4, dash="dash"),
        )
    )
    style_figure(fig, y_title="Cumulative cases")
    fig.update_yaxes(tickformat=",.0f")
    return fig


def province_chart(provinces: pd.DataFrame) -> go.Figure:
    """Show the health zones with the highest confirmed case totals."""
    top = provinces.head(12).sort_values("cases")
    max_cases = float(top["cases"].max()) if not top.empty and top["cases"].max() > 0 else 1.0

    fig = go.Figure(
        go.Bar(
            x=top["cases"],
            y=top["location_name"],
            orientation="h",
            marker=dict(
                color=top["cases"],
                colorscale=[
                    [0.0, "#93C5FD"],  # Soft tint of brand blue for lower counts
                    [0.5, "#1A73E8"],  # Core Public Health Brand Blue
                    [1.0, "#0D47A1"],  # Deep Brand Navy for top hotspots
                ],
            ),
            text=top["cases"].map(lambda value: f"{value:,.0f}"),
            textposition="outside",
            textfont=dict(
                size=12,
                color="#0D47A1",       # Dark brand navy for crisp, legible values
                family="Segoe UI, Inter, sans-serif",
            ),
            cliponaxis=False,
            hovertemplate="%{y}: %{x:,.0f} confirmed cases<extra></extra>",
        )
    )
    style_figure(fig, height=430, x_title="Confirmed cases (cumulative)", unified_hover=False)
    fig.update_xaxes(tickformat=",.0f", range=[0, max_cases * 1.15])
    fig.update_yaxes(tickfont=dict(color="#202124", size=12))
    fig.update_xaxes(tickfont=dict(color="#5F6368", size=11), title_font=dict(color="#202124", size=12))
    fig.update_layout(showlegend=False)
    return fig


def sir_cumulative_chart(
    view: pd.DataFrame,
    model_dates: pd.DatetimeIndex,
    model_cumulative: np.ndarray,
    beta: float,
    gamma: float,
    log_scale: bool,
    case_label: str,
) -> go.Figure:
    """Overlay reported cumulative cases with the fitted or tuned SIR curve."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=view.index,
            y=view["cases"],
            name=f"Reported {case_label.lower()}",
            mode="lines+markers",
            marker=dict(size=3, color="rgba(45, 212, 191, 0.65)"),
            line=dict(color=COLORS["teal"], width=3),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=model_dates,
            y=model_cumulative,
            name=f"SIR cumulative (beta={beta:.2f}, gamma={gamma:.2f})",
            mode="lines",
            line=dict(color=COLORS["orange"], width=2.6, dash="dash"),
        )
    )
    # Mark where observed data end and the model projection begins.
    cutoff = view.index[-1]
    fig.add_vline(x=cutoff, line_width=1, line_dash="dot", line_color=COLORS["muted"])
    if model_dates[-1] > cutoff:
        fig.add_vrect(
            x0=cutoff,
            x1=model_dates[-1],
            fillcolor=COLORS["orange"],
            opacity=0.045,
            line_width=0,
            layer="below",
        )
        fig.add_annotation(
            x=cutoff,
            y=1.0,
            yref="paper",
            text="projection",
            showarrow=False,
            xanchor="left",
            font=dict(size=10, color=COLORS["orange"]),
        )
    style_figure(fig, height=450, y_title="Cumulative cases")
    if log_scale:
        fig.update_yaxes(type="log", tickformat=",.0f", title_text="Cumulative cases (log scale)")
    else:
        fig.update_yaxes(tickformat=",.0f")
    return fig


def sir_incidence_chart(
    view: pd.DataFrame,
    model_dates: pd.DatetimeIndex,
    model_incidence: np.ndarray,
) -> go.Figure:
    """Compare reported daily cases with infections generated by the SIR model."""
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=view.index,
            y=view["new_cases"],
            name="Reported new cases",
            marker_color="rgba(148, 163, 184, 0.28)",
            hovertemplate="%{y:,.1f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=view.index,
            y=view["new_cases_7d"],
            name="Reported 7-day average",
            mode="lines",
            line=dict(color=COLORS["teal"], width=2.2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=model_dates,
            y=model_incidence,
            name="SIR daily new infections",
            mode="lines",
            line=dict(color=COLORS["orange"], width=2.4, dash="dash"),
        )
    )
    style_figure(fig, height=390, y_title="New infections per day")
    fig.update_yaxes(tickformat=",.0f", rangemode="tozero")
    return fig


def active_infection_chart(
    model_dates: pd.DatetimeIndex, trajectory_infectious: np.ndarray
) -> go.Figure:
    """Plot the modeled number of currently infectious people and its peak."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=model_dates,
            y=trajectory_infectious,
            name="Active infections",
            mode="lines",
            line=dict(color=COLORS["blue"], width=2.6),
            fill="tozeroy",
            fillcolor="rgba(96, 165, 250, 0.14)",
        )
    )
    peak_index = int(np.argmax(trajectory_infectious))
    fig.add_trace(
        go.Scatter(
            x=[model_dates[peak_index]],
            y=[trajectory_infectious[peak_index]],
            name="Peak",
            mode="markers+text",
            marker=dict(color=COLORS["amber"], size=9, symbol="diamond"),
            text=["peak"],
            textposition="top center",
            textfont=dict(size=10, color=COLORS["amber"]),
        )
    )
    style_figure(fig, height=370, y_title="Active infections")
    fig.update_yaxes(tickformat=",.0f", rangemode="tozero")
    return fig


def phase_chart(susceptible: np.ndarray, infectious: np.ndarray) -> go.Figure:
    """Show the SIR trajectory through susceptible-infectious space."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=susceptible / 1_000_000,
            y=infectious,
            name="SIR trajectory",
            mode="lines",
            line=dict(color=COLORS["violet"], width=2.4),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[susceptible[0] / 1_000_000],
            y=[infectious[0]],
            name="Start",
            mode="markers",
            marker=dict(color=COLORS["teal"], size=11, symbol="circle"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[susceptible[-1] / 1_000_000],
            y=[infectious[-1]],
            name="End",
            mode="markers",
            marker=dict(color=COLORS["coral"], size=11, symbol="x"),
        )
    )
    style_figure(fig, height=370, x_title="Susceptible (millions)", y_title="Active infections")
    fig.update_yaxes(tickformat=",.0f", rangemode="tozero")
    return fig


def outbreak_series(
    west: pd.DataFrame, drc: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build combined outbreak series aligned by days since first report."""
    case_matrix = west.pivot(index="date", columns="country", values="cases").sort_index()
    death_matrix = west.pivot(index="date", columns="country", values="deaths").sort_index()
    case_matrix = case_matrix.ffill().fillna(0.0)
    death_matrix = death_matrix.reindex(case_matrix.index).ffill().fillna(0.0)
    west_series = pd.DataFrame(
        {"cases": case_matrix.sum(axis=1), "deaths": death_matrix.sum(axis=1)}
    )
    west_series["days"] = (west_series.index - west_series.index[0]).days
    drc_series = drc.set_index("date")[["cases", "deaths"]].sort_index()
    drc_series["days"] = (drc_series.index - drc_series.index[0]).days
    return west_series, drc_series


def comparison_chart(west_series: pd.DataFrame, drc_series: pd.DataFrame, log_scale: bool) -> go.Figure:
    """Overlay cumulative cases and deaths for both outbreaks."""
    fig = go.Figure()
    for frame, name, color in [
        (west_series, "West Africa 2014-16", COLORS["teal"]),
        (drc_series, "DRC 2026 (BDBV)", COLORS["orange"]),
    ]:
        fig.add_trace(
            go.Scatter(
                x=frame["days"],
                y=frame["cases"],
                name=f"{name} cases",
                mode="lines",
                line=dict(color=color, width=2.8),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=frame["days"],
                y=frame["deaths"],
                name=f"{name} deaths",
                mode="lines",
                line=dict(color=color, width=1.5, dash="dot"),
            )
        )
    style_figure(fig, height=460, x_title="Days since first reported case", y_title="Cumulative people")
    if log_scale:
        fig.update_yaxes(type="log", tickformat=",.0f", title_text="Cumulative people (log scale)")
    else:
        fig.update_yaxes(tickformat=",.0f")
    return fig


# ---------------------------------------------------------------------------
# Page header and data load
# ---------------------------------------------------------------------------

west_data, drc_data, province_data, metadata = load_data()

st.markdown(
    """
    <div class="hero">
        <div class="kicker">Global outbreak intelligence</div>
        <h1>Ebola Epi Watch</h1>
        <p>Historical Ebola surveillance, outbreak comparison, and an interactive SIR transmission
        model. Adjust transmission and recovery rates to see how a stylized epidemic curve compares
        with the reported record.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## Control room")

    # -----------------------------------------------------------------------
    # 1. Outbreak Dataset Selection
    # Toggle between West Africa 2014-16 historical data and DRC 2026 surveillance
    # -----------------------------------------------------------------------
    dataset_label = st.radio(
        "Outbreak dataset",
        ["West Africa 2014-16", "DRC 2026 (BDBV)"],
        key="dataset",
    )
    dataset_key = "west_africa" if dataset_label == "West Africa 2014-16" else "drc"
    is_drc = dataset_key == "drc"

    # -----------------------------------------------------------------------
    # 2. Geographic Filtering
    # Multiselect countries for West Africa; locked to national DRC scope for DRC
    # -----------------------------------------------------------------------
    if is_drc:
        selected_countries = ["Democratic Republic of the Congo"]
        date_min = drc_data["date"].min().date()
        date_max = drc_data["date"].max().date()
    else:
        country_options = sorted(west_data["country"].dropna().unique().tolist())
        selected_countries = st.multiselect(
            "Countries",
            options=country_options,
            default=[country for country in DEFAULT_COUNTRIES if country in country_options],
            key="countries",
        )
        if not selected_countries:
            st.warning("Select at least one country.")
            st.stop()
        date_min = west_data["date"].min().date()
        date_max = west_data["date"].max().date()

    # -----------------------------------------------------------------------
    # 3. Temporal Range / Analysis Window Slider
    # Dataset-specific session key isolates date ranges across outbreaks
    # -----------------------------------------------------------------------
    window_key = f"window_{dataset_key}"
    if window_key not in st.session_state:
        st.session_state[window_key] = (date_min, date_max)
    window = st.slider(
        "Analysis window",
        min_value=date_min,
        max_value=date_max,
        key=window_key,
    )
    start_date = pd.Timestamp(window[0])
    end_date = pd.Timestamp(window[1])
    if end_date <= start_date:
        st.error("The analysis window must span at least two days.")
        st.stop()

    # -----------------------------------------------------------------------
    # 4. Construct Sliced Active View & Suggested Population Total
    # -----------------------------------------------------------------------
    if is_drc:
        active_view = build_drc_view(drc_data, start_date, end_date)
        country_matrix = pd.DataFrame()
        suggested_population = DEFAULT_DRC_POPULATION / 1_000_000
    else:
        active_view, country_matrix = build_west_view(
            west_data, selected_countries, start_date, end_date
        )
        suggested_population = sum(
            POPULATIONS.get(country, 0) for country in selected_countries
        ) / 1_000_000
    if active_view.empty:
        st.warning("No observations in the selected window.")
        st.stop()

    case_label = "Confirmed cases" if is_drc else "Reported cases"
    population_key = f"population_{dataset_key}"
    if population_key not in st.session_state:
        st.session_state[population_key] = round(suggested_population, 1)

    # -----------------------------------------------------------------------
    # 5. Susceptible Population Baseline Input (Millions)
    # -----------------------------------------------------------------------
    population_millions = st.number_input(
        "Population (millions)",
        min_value=0.1,
        max_value=2000.0,
        step=1.0,
        format="%.1f",
        key=population_key,
    )
    population = float(population_millions) * 1_000_000
    if is_drc:
        st.caption(
            "Default is the national population. Use a smaller affected-area population to explore local dynamics."
        )
    else:
        st.caption(
            f"Selected countries total about {suggested_population:,.1f}M using 2014 estimates."
        )

    st.markdown("---")
    st.markdown("### SIR parameters")

    # -----------------------------------------------------------------------
    # 6. Model Seeding: Initial Active Infectious Cases (I0)
    # Seeded from empirical cumulative cases at the window start
    # -----------------------------------------------------------------------
    initial_cases = float(max(active_view["cases"].iloc[0], 1.0))
    max_initial = int(max(active_view["cases"].max(), 10.0))
    i0_key = f"i0_{dataset_key}"
    if i0_key in st.session_state and st.session_state[i0_key] > max_initial:
        st.session_state[i0_key] = max_initial
    if i0_key not in st.session_state:
        st.session_state[i0_key] = int(np.clip(round(initial_cases * 0.2), 1, max_initial))
    initial_infectious = st.number_input(
        "Active infections at window start",
        min_value=1,
        max_value=max_initial,
        step=1,
        key=i0_key,
    )

    beta_key = f"beta_{dataset_key}"
    gamma_key = f"gamma_{dataset_key}"
    fit_note_key = f"fit_note_{dataset_key}"

    # -----------------------------------------------------------------------
    # 7. Initial Automatic Calibration Baseline
    # Calibrates an initial realistic curve rather than setting arbitrary default sliders
    # -----------------------------------------------------------------------
    if beta_key not in st.session_state or gamma_key not in st.session_state:
        with st.spinner("Calibrating a baseline SIR curve"):
            base_beta, base_gamma, _base_rmse = baseline_fit(
                tuple(np.round(active_view["cases"].to_numpy(dtype=float), 4)),
                population,
                initial_cases,
                float(initial_infectious),
            )
        st.session_state[beta_key] = float(np.clip(round(base_beta, 2), 0.05, 2.00))
        st.session_state[gamma_key] = float(np.clip(round(base_gamma, 2), 0.02, 1.00))
    if "forecast_days" not in st.session_state:
        st.session_state["forecast_days"] = 90

    # -----------------------------------------------------------------------
    # 8. Non-Linear Least Squares Auto-Fit Action
    # Optimizes beta and gamma against empirical cumulative cases using Scipy
    # -----------------------------------------------------------------------
    if st.button("Auto-fit beta and gamma", type="primary", key="fit_button"):
        with st.spinner("Calibrating SIR to the reported curve"):
            try:
                fit = fit_sir(
                    active_view["cases"].to_numpy(dtype=float),
                    np.arange(len(active_view), dtype=float),
                    population,
                    initial_cases,
                    float(initial_infectious),
                )
            except Exception as exc:
                fit = None
                st.error(f"Fit failed: {exc}")
        if fit is not None:
            if fit.success:
                st.session_state[beta_key] = float(np.clip(round(fit.beta, 2), 0.05, 2.00))
                st.session_state[gamma_key] = float(np.clip(round(fit.gamma, 2), 0.02, 1.00))
                boundary = (
                    fit.beta <= 0.051
                    or fit.beta >= 1.99
                    or fit.gamma <= 0.021
                    or fit.gamma >= 0.99
                )
                note = (
                    f"Auto-fit set beta={fit.beta:.2f} and gamma={fit.gamma:.2f} "
                    f"(log-scale RMSE {fit.rmse:.3f})."
                )
                if boundary:
                    note += " The estimate sits at a search boundary; the constant-parameter SIR cannot capture every feature of this curve."
                st.session_state[fit_note_key] = {
                    "beta": st.session_state[beta_key],
                    "gamma": st.session_state[gamma_key],
                    "text": note,
                }
                st.rerun()
            else:
                st.warning("The fit did not converge. Try a longer analysis window.")

    # -----------------------------------------------------------------------
    # 9. Manual SIR Parameter Sliders & Forecast Horizon
    # -----------------------------------------------------------------------
    beta = st.slider(
        "Transmission rate beta (per day)",
        min_value=0.05,
        max_value=2.00,
        step=0.01,
        key=beta_key,
    )
    gamma = st.slider(
        "Recovery rate gamma (per day)",
        min_value=0.02,
        max_value=1.00,
        step=0.01,
        key=gamma_key,
    )
    forecast_days = st.slider(
        "Projection horizon (days)",
        min_value=0,
        max_value=180,
        step=15,
        key="forecast_days",
    )

    # -----------------------------------------------------------------------
    # 10. Live Reproduction Number (R0) & Infectious Duration Summary Card
    # -----------------------------------------------------------------------
    with st.container(border=True):
        st.metric("Basic reproduction number", f"R0 = {beta / gamma:.2f}")
        st.caption(
            f"Infectious period is about {1 / gamma:.1f} days. "
            f"Effective R at model start is {beta * max(population - initial_cases, 0) / (gamma * population):.2f}."
        )
    fit_state = st.session_state.get(fit_note_key)
    if isinstance(fit_state, dict):
        if (
            abs(beta - fit_state["beta"]) <= 0.005
            and abs(gamma - fit_state["gamma"]) <= 0.005
        ):
            st.info(fit_state["text"])
        else:
            st.caption("Sliders have been adjusted since the last auto-fit.")

    # Show where the selected dataset came from and when it was ingested.
    active_metadata = metadata.get("datasets", {}).get(dataset_key, {})
    with st.expander("Data provenance"):
        st.write(active_metadata.get("title", dataset_label))
        st.caption(active_metadata.get("organization", "Humanitarian Data Exchange"))
        if active_metadata.get("start"):
            st.caption(
                f"Coverage {active_metadata['start']} to {active_metadata['end']} | "
                f"{active_metadata.get('rows', 0):,} normalized rows"
            )
        if metadata.get("generated_at"):
            st.caption(f"Ingested {metadata['generated_at']}")
        gho = metadata.get("who_gho_probe", {})
        if gho.get("status") == "reachable":
            st.caption(
                f"WHO GHO probe: {gho.get('ebola_indicator_count', 0)} Ebola indicators found."
            )

# ---------------------------------------------------------------------------
# Executive KPI Strip: Outbreak Surveillance Summary
# Computes cumulative totals, case fatality ratio (CFR), peak incidence,
# and 7-day week-over-week velocity changes.
# ---------------------------------------------------------------------------

st.markdown(
    f'<div class="source-strip">'
    f'<span class="pill">Source: {active_metadata.get("organization", "HDX")}</span>'
    f'<span class="pill">{start_date:%b %d, %Y} to {end_date:%b %d, %Y}</span>'
    f'<span class="pill">{len(selected_countries)} {"country" if len(selected_countries) == 1 else "countries"}</span>'
    f'<span class="pill">SIR: beta {beta:.2f} / gamma {gamma:.2f}</span>'
    f"</div>",
    unsafe_allow_html=True,
)

# Extract final observations and compute key epidemiological ratios
last_date = active_view.index[-1]
total_cases = float(active_view["cases"].iloc[-1])
total_deaths = float(active_view["deaths"].iloc[-1])

# Guard against division by zero when the selected window has no cases yet.
if total_cases > 0:
    cfr = total_deaths / total_cases
else:
    cfr = 0.0
peak_7d = float(active_view["new_cases_7d"].max())
peak_7d_date = active_view["new_cases_7d"].idxmax()

# Week-over-week velocity comparisons (current 7 days vs previous 7 days)
case_delta = latest_change(active_view["new_cases"])
death_delta = latest_change(active_view["new_deaths"])

metric_columns = st.columns(5)
with metric_columns[0]:
    with st.container(border=True):
        st.metric(
            case_label,
            f"{total_cases:,.0f}",
            delta=f"{case_delta:+.0%} vs prior 7d" if case_delta is not None else None,
            delta_color="inverse",
            help="Cumulative cases at the end of the selected window.",
        )
with metric_columns[1]:
    with st.container(border=True):
        st.metric(
            "Deaths",
            f"{total_deaths:,.0f}",
            delta=f"{death_delta:+.0%} vs prior 7d" if death_delta is not None else None,
            delta_color="inverse",
            help="Cumulative deaths at the end of the selected window.",
        )
with metric_columns[2]:
    with st.container(border=True):
        st.metric(
            "Case fatality ratio",
            pct(cfr),
            help="Cumulative deaths divided by cumulative cases. Reporting differences affect this ratio.",
        )
with metric_columns[3]:
    with st.container(border=True):
        st.metric(
            "Peak 7-day incidence",
            f"{peak_7d:,.0f}",
            help=f"Highest 7-day average of new cases, reached on {peak_7d_date:%b %d, %Y}.",
        )
with metric_columns[4]:
    with st.container(border=True):
        if is_drc:
            affected_zones = int((province_data["cases"] > 0).sum())
            st.metric(
                "Affected health zones",
                f"{affected_zones:,}",
                help="Health zones with confirmed cases.",
            )
        else:
            st.metric(
                "Countries in view",
                f"{len(selected_countries):,}",
                help="Countries included in the aggregated series.",
            )

# ---------------------------------------------------------------------------
# Main analysis tabs
# ---------------------------------------------------------------------------

tab_trend, tab_sir, tab_compare = st.tabs(
    ["Outbreak curve", "SIR simulator", "Global comparison"]
)

# ---------------------------------------------------------------------------
# Tab 1: Reported Surveillance Curves & Geographic Breakdowns
# Displays cumulative totals, daily incidence, rolling averages, and regional splits
# ---------------------------------------------------------------------------
with tab_trend:
    # 1.1 Contextual Outbreak Scope Header
    if is_drc:
        st.markdown(
            f'<div class="section-copy">Daily confirmed cases and deaths reported nationally. '
            f"The latest observation is {last_date:%B %d, %Y}.</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="section-copy">Combined curve for {", ".join(selected_countries)}. '
            "Cumulative counts are linearly interpolated between situation reports.</div>",
            unsafe_allow_html=True,
        )

    # 1.2 Cumulative Epidemic Curve Card (Linear vs. Logarithmic Toggle)
    with st.container(border=True):
        log_scale = st.checkbox("Logarithmic cumulative axis", value=False, key="trend_log")
        st.plotly_chart(
            cumulative_chart(active_view, log_scale, case_label),
            width="stretch",
            config={"displayModeBar": False},
        )

    # 1.3 Daily Incidence & 7-Day Rolling Smoothed Average Card
    with st.container(border=True):
        st.plotly_chart(
            incidence_chart(active_view, case_label),
            width="stretch",
            config={"displayModeBar": False},
        )

    # 1.4 Subnational / Geographic Breakdown Card (DRC Zones or Country Comparisons)
    with st.container(border=True):
        if is_drc:
            st.markdown("#### Health zones with the most confirmed cases")
            st.plotly_chart(
                province_chart(province_data),
                width="stretch",
                config={"displayModeBar": False},
            )
        else:
            st.markdown("#### Country trajectories")
            st.plotly_chart(
                country_chart(country_matrix, active_view),
                width="stretch",
                config={"displayModeBar": False},
            )

    # 1.5 Interactive Data Viewer & CSV Export Expander

    with st.expander("View and download the aggregated series"):
        table = active_view.reset_index().rename(
            columns={
                "date": "Date",
                "cases": "Cumulative cases",
                "deaths": "Cumulative deaths",
                "new_cases": "New cases",
                "new_cases_7d": "7-day average",
                "new_deaths": "New deaths",
                "suspected_cases": "Suspected cases",
            }
        )
        st.dataframe(
            table.tail(30).iloc[::-1],
            width="stretch",
            hide_index=True,
        )
        st.download_button(
            "Download CSV",
            data=table.to_csv(index=False).encode("utf-8"),
            file_name=f"ebola_{dataset_key}_window.csv",
            mime="text/csv",
        )

# ---------------------------------------------------------------------------
# Tab 2: SIR Simulator - Compartmental Transmission Modeling
# Integrates differential equations, evaluates R0/effective R, and projects curves
# ---------------------------------------------------------------------------
with tab_sir:
    # 2.1 Numerical ODE Integration
    # Solves dS/dt, dI/dt, dR/dt forward in time across the analysis + forecast horizon
    model_horizon = len(active_view) - 1 + int(forecast_days)
    try:
        trajectory = simulate_sir(
            SIRParams(
                beta=float(beta),
                gamma=float(gamma),
                population=population,
                initial_cases=initial_cases,
                initial_infectious=float(initial_infectious),
            ),
            model_horizon,
            t_eval=np.arange(0, model_horizon + 1, dtype=float),
        )
    except Exception as exc:
        st.error(f"Model could not be simulated: {exc}")
        st.stop()

    # Map simulation time index to calendar dates
    model_dates = pd.date_range(
        start=active_view.index[0], periods=model_horizon + 1, freq="D"
    )
    metrics = model_metrics(trajectory, beta, gamma, population)
    projected_cases = float(trajectory.cumulative[-1])
    additional_cases = projected_cases - total_cases

    st.markdown(
        '<div class="section-copy">The SIR model divides a closed population into susceptible, '
        "infectious, and recovered compartments. Adjust beta and gamma in the sidebar, or auto-fit "
        "them to the reported curve.</div>",
        unsafe_allow_html=True,
    )

    # 2.2 Epidemiological Model Diagnostic Metric Cards
    model_metrics_columns = st.columns(4)
    with model_metrics_columns[0]:
        with st.container(border=True):
            st.metric("R0 = beta / gamma", f"{metrics['r0']:.2f}", help="Basic reproduction number.")
    with model_metrics_columns[1]:
        with st.container(border=True):
            st.metric(
                "Effective R at start",
                f"{metrics['effective_r']:.2f}",
                help="Transmission potential after accounting for the initially susceptible share.",
            )
    with model_metrics_columns[2]:
        with st.container(border=True):
            st.metric(
                "Peak active infections",
                human_int(metrics["peak_active"]),
                help=f"Modeled peak on day {metrics['peak_day']:.0f} after the window start.",
            )
    with model_metrics_columns[3]:
        with st.container(border=True):
            st.metric(
                f"Projected by {model_dates[-1]:%b %d, %Y}",
                human_int(projected_cases),
                delta=f"{additional_cases:+,.0f} vs reported" if forecast_days > 0 else None,
                delta_color="inverse",
                help="Cumulative infections at the end of the projection horizon.",
            )

    st.caption(
        f"Herd immunity threshold: {pct(metrics['herd_immunity'])} | "
        f"Peak on {model_dates[int(metrics['peak_day'])]:%b %d, %Y} | "
        f"Model attack rate: {pct(metrics['attack_rate'])} | "
        f"Fit RMSE on log scale is reported when auto-fit is used."
    )
    st.caption(
        f"Model time zero is {active_view.index[0]:%b %d, %Y}, seeded with "
        f"{int(initial_infectious):,} active infections among {initial_cases:,.0f} cumulative cases. "
        "Fitted beta and gamma describe post-seed dynamics, not necessarily the outbreak's intrinsic "
        "reproduction number. A constant-parameter SIR cannot reproduce interventions or reporting changes."
    )

    # 2.3 Cumulative Infections: Empirical Reported vs. SIR Modeled Projection
    with st.container(border=True):
        sir_log = st.checkbox("Logarithmic cumulative axis", value=False, key="sir_log")
        st.plotly_chart(
            sir_cumulative_chart(
                active_view,
                model_dates,
                trajectory.cumulative,
                beta,
                gamma,
                sir_log,
                case_label,
            ),
            width="stretch",
            config={"displayModeBar": False},
        )

    # 2.4 Daily Incidence Comparison: Empirical Daily Cases vs. Modeled Rate
    with st.container(border=True):
        st.plotly_chart(
            sir_incidence_chart(active_view, model_dates, trajectory.incidence),
            width="stretch",
            config={"displayModeBar": False},
        )

    # 2.5 Compartmental State Dynamics: Active Infections & Phase Plane (S vs. I)
    lower_left, lower_right = st.columns(2)
    with lower_left:
        with st.container(border=True):
            st.markdown("#### Modeled active infections")
            st.plotly_chart(
                active_infection_chart(model_dates, trajectory.infectious),
                width="stretch",
                config={"displayModeBar": False},
            )
    with lower_right:
        with st.container(border=True):
            st.markdown("#### Phase plane")
            st.plotly_chart(
                phase_chart(trajectory.susceptible, trajectory.infectious),
                width="stretch",
                config={"displayModeBar": False},
            )
    st.caption(
        "SIR assumptions: closed population, homogeneous mixing, no interventions, no reporting "
        "delay, and no separate death compartment. Use this as a scenario explorer, not an "
        "official forecast."
    )

# ---------------------------------------------------------------------------
# Tab 3: Global Comparison - Cross-Outbreak Normalized Analysis
# Overlays distinct epidemics aligned to Day 0 (first reported case)
# ---------------------------------------------------------------------------
with tab_compare:
    west_series, drc_series = outbreak_series(west_data, drc_data)
    st.markdown(
        '<div class="section-copy">Cumulative trajectories aligned by days since the first '
        "reported case. The y-axis can be switched to a logarithmic scale to compare growth "
        "phases across outbreaks of very different sizes.</div>",
        unsafe_allow_html=True,
    )

    # 3.1 Normalized Timeline Growth Overlay Card (Day 0 Normalization)
    with st.container(border=True):
        compare_log = st.checkbox("Logarithmic cumulative axis", value=True, key="compare_log")
        st.plotly_chart(
            comparison_chart(west_series, drc_series, compare_log),
            width="stretch",
            config={"displayModeBar": False},
        )

    def stats_row(frame: pd.DataFrame, name: str, start: pd.Timestamp) -> dict:
        """Build one row of the cross-outbreak summary table."""
        cases = float(frame["cases"].iloc[-1])
        deaths = float(frame["deaths"].iloc[-1])
        return {
            "Outbreak": name,
            "First report": f"{start:%b %d, %Y}",
            "Last report": f"{frame.index[-1]:%b %d, %Y}",
            "Cases": f"{cases:,.0f}",
            "Deaths": f"{deaths:,.0f}",
            "CFR": pct(deaths / cases if cases else 0.0),
            "Duration (days)": f"{(frame.index[-1] - start).days:,}",
        }

    # 3.2 Cross-Outbreak Metric Summary Table Card
    comparison_table = pd.DataFrame(
        [
            stats_row(west_series, "West Africa 2014-16", west_series.index[0]),
            stats_row(drc_series, "DRC 2026 (BDBV)", drc_series.index[0]),
        ]
    )
    with st.container(border=True):
        st.dataframe(comparison_table, width="stretch", hide_index=True)

    # 3.3 Side-by-Side Comparative Geographic Distributions
    compare_left, compare_right = st.columns(2)
    with compare_left:
        with st.container(border=True):
            st.markdown("#### West Africa: cumulative cases by country")
            west_country_matrix = (
                west_data.pivot(index="date", columns="country", values="cases")
                .sort_index()
                .ffill()
                .fillna(0.0)
            )
            west_combined = pd.DataFrame({"cases": west_country_matrix.sum(axis=1)})
            st.plotly_chart(
                country_chart(west_country_matrix, west_combined),
                width="stretch",
                config={"displayModeBar": False},
            )
    with compare_right:
        with st.container(border=True):
            st.markdown("#### DRC 2026: top health zones")
            st.plotly_chart(
                province_chart(province_data.head(10)),
                width="stretch",
                config={"displayModeBar": False},
            )

    # 3.4 Data Provenance and Open-Data Source Register
    # Combines ingestion metadata with active data ranges for full auditing
    st.markdown("#### Source register")
    source_rows = []
    for key, label, frame in [
        ("west_africa", "West Africa 2014-16", west_series),
        ("drc", "DRC 2026 (BDBV)", drc_series),
    ]:
        info = metadata.get("datasets", {}).get(key, {})
        source_rows.append(
            {
                "Dataset": label,
                "Publisher": info.get("organization", "HDX"),
                "Coverage": (
                    f"{info.get('start', '')} to {info.get('end', '')}"
                    if info.get("start")
                    else f"{frame.index[0]:%Y-%m-%d} to {frame.index[-1]:%Y-%m-%d}"
                ),
                "Resource": info.get("title", label),
            }
        )
    st.dataframe(pd.DataFrame(source_rows), width="stretch", hide_index=True)
    st.caption(
        "West Africa data: WHO GAR situation reports distributed through HDX. "
        "DRC data: INRB-UMIE BDBV2026 surveillance streams distributed through HDX."
    )

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.markdown("---")
st.caption(
    "Ebola Epi Watch built by Chijioke C. Uhegwu | Historical surveillance and scenario modeling. "
    "Reported curves reflect reporting revisions, whereas SIR output is a deterministic compartmental model."
)

