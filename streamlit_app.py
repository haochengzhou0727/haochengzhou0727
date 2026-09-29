from __future__ import annotations

from pathlib import Path
from urllib.parse import quote

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from carry_dashboard import DEFAULT_BASKET, DEFAULT_FLIES, DEFAULT_LEVELS, DEFAULT_SPREADS, DEFAULT_WORKBOOK, ModelControls, PythonModel, build_model

st.set_page_config(page_title="Carry & Rolldown Python Model", page_icon="C", layout="wide")

CACHE_VERSION = "cross-ranking-v7-lookback-cr-volatility"


@st.cache_data(show_spinner="Calculating curves, carry, rolldown, and scores in Python...")
def calculate(path: str, modified_ns: int, controls: ModelControls, cache_version: str) -> PythonModel:
    del modified_ns
    del cache_version
    return build_model(path, controls)


def plot_history(frame: pd.DataFrame, value_columns: list[str], title: str) -> None:
    figure = go.Figure()
    reference_columns = {"Mean", "+1 SD", "-1 SD", "+2 SD", "-2 SD"}
    primary_columns = [column for column in value_columns if column not in reference_columns]
    for column in value_columns:
        reference_styles = {
            "Mean": {"color": "#111827", "width": 2.25, "dash": "solid"},
            "+1 SD": {"color": "#2563eb", "width": 1.5, "dash": "dash"},
            "-1 SD": {"color": "#2563eb", "width": 1.5, "dash": "dash"},
            "+2 SD": {"color": "#d97706", "width": 1.5, "dash": "dot"},
            "-2 SD": {"color": "#d97706", "width": 1.5, "dash": "dot"},
        }
        line = reference_styles.get(column)
        if column in primary_columns and any(term in column.lower() for term in ("c&r", "carry", "roll")):
            line = {"color": "#15803d", "width": 3, "dash": "solid"}
        elif column in primary_columns:
            line = {"color": "#1e3a8a", "width": 3, "dash": "solid"}
        figure.add_trace(
            go.Scatter(
                x=frame.index,
                y=frame[column],
                mode="lines",
                name=column,
                line=line,
                hovertemplate="%{y:.2f}<extra>%{fullData.name}</extra>",
            )
        )
    if len(primary_columns) == 1 and not reference_columns.intersection(value_columns):
        series = pd.to_numeric(frame[primary_columns[0]], errors="coerce").dropna()
        if len(series) > 1:
            mean = float(series.mean())
            stdev = float(series.std())
            for label, value, color, dash in (
                ("Mean", mean, "#111827", "solid"),
                ("+1 SD", mean + stdev, "#2563eb", "dash"),
                ("-1 SD", mean - stdev, "#2563eb", "dash"),
                ("+2 SD", mean + 2 * stdev, "#d97706", "dot"),
                ("-2 SD", mean - 2 * stdev, "#d97706", "dot"),
            ):
                figure.add_trace(
                    go.Scatter(
                        x=frame.index,
                        y=[value] * len(frame.index),
                        mode="lines",
                        name=label,
                        line={"color": color, "width": 2 if label == "Mean" else 1.5, "dash": dash},
                        hoverinfo="skip",
                        showlegend=True,
                    )
                )
    figure.update_layout(
        template="plotly_white",
        title={"text": title, "x": 0, "xanchor": "left", "font": {"size": 18, "color": "#172033"}},
        height=500,
        hovermode="x unified",
        hoverlabel={"bgcolor": "#172033", "font": {"color": "white"}},
        margin={"l": 24, "r": 24, "t": 64, "b": 24},
        paper_bgcolor="#ffffff",
        plot_bgcolor="#f8fafc",
        xaxis={
            "title": None,
            "rangeslider": {"visible": True, "thickness": 0.08},
            "type": "date",
            "showgrid": False,
            "linecolor": "#cbd5e1",
            "tickfont": {"color": "#64748b"},
        },
        yaxis={
            "title": None,
            "showgrid": True,
            "gridcolor": "#e2e8f0",
            "zeroline": True,
            "zerolinecolor": "#94a3b8",
            "tickfont": {"color": "#64748b"},
        },
        legend={"orientation": "h", "y": 1.02, "x": 0, "font": {"color": "#334155"}},
    )
    st.plotly_chart(figure, use_container_width=True, config={"scrollZoom": True, "displaylogo": False})


def query_value(name: str) -> str | None:
    value = st.query_params.get(name)
    if isinstance(value, list):
        return value[0] if value else None
    return str(value) if value is not None else None


def query_int(name: str, default: int) -> int:
    value = query_value(name)
    try:
        return int(value) if value is not None else default
    except ValueError:
        return default


def query_float(name: str, default: float) -> float:
    value = query_value(name)
    try:
        return float(value) if value is not None else default
    except ValueError:
        return default


def query_bool(name: str, default: bool) -> bool:
    value = query_value(name)
    if value is None:
        return default
    return value.lower() in ("1", "true", "yes", "on")


workbook_path = Path(st.sidebar.text_input("Raw-data workbook", str(DEFAULT_WORKBOOK)))
if not workbook_path.exists():
    st.error(f"Workbook not found: {workbook_path}")
    st.stop()

st.sidebar.subheader("Python model controls")
with st.sidebar.expander("Structures and basket", expanded=True):
    basket_text = st.text_input("Basket currencies", query_value("basket") or ", ".join(DEFAULT_BASKET), help="Comma-separated currencies used for basket RV.")
    levels_text = st.text_input("Levels", query_value("levels") or ", ".join(DEFAULT_LEVELS), help="Comma-separated tenors, for example 1M, 6M, 1Y, 2Y.")
    spreads_text = st.text_input("Spreads", query_value("spreads") or ", ".join(DEFAULT_SPREADS), help="Comma-separated structures, for example 1s2s, 2s5s.")
    flies_text = st.text_input("Flies", query_value("flies") or ", ".join(DEFAULT_FLIES), help="Comma-separated structures, for example 1s2s5s.")
with st.sidebar.expander("Lookback and ranking", expanded=True):
    lookback = st.number_input("Lookback (months)", min_value=1, max_value=120, value=query_int("lookback", 6))
    horizon = st.number_input("Carry horizon (months)", min_value=1, max_value=24, value=query_int("horizon", 3))
    min_tenor = st.number_input("Minimum cross-trade tenor (months)", min_value=1, max_value=120, value=query_int("min_tenor", 12))
    top_z = st.number_input("Top / bottom z-score", min_value=1, max_value=50, value=query_int("top_z", 3))
    top_percentile = st.number_input("Top / bottom percentile", min_value=1, max_value=50, value=query_int("top_percentile", 3))
    top_carry_vol = st.number_input("Top C&R / volatility", min_value=1, max_value=50, value=query_int("top_carry_vol", 3))
    top_n = st.number_input("Top composite trades", min_value=1, max_value=50, value=query_int("top_n", 5))
with st.sidebar.expander("Composite weights", expanded=True):
    use_usd_benchmark = st.checkbox("Use USD as benchmark", value=query_bool("use_usd", True), help="Include USD relative-value adjustment in cross-currency composite scores. USD still has its own analysis.")
    weight_levels = st.number_input("Weight: level RV", value=query_float("weight_levels", 0.5), step=0.05)
    weight_usd = st.number_input("Weight: USD RV", value=query_float("weight_usd", 0.7), step=0.05)
    weight_basket = st.number_input("Weight: basket RV", value=query_float("weight_basket", 0.3), step=0.05)
    weight_carry = st.number_input("Weight: carry and rolldown", value=query_float("weight_carry", 0.15), step=0.05)

def parse_list(value: str) -> tuple[str, ...]:
    return tuple(item.strip().upper() for item in value.split(",") if item.strip())


controls = ModelControls(
    lookback_months=int(lookback),
    carry_horizon_months=int(horizon),
    levels=parse_list(levels_text),
    spreads=parse_list(spreads_text),
    flies=parse_list(flies_text),
    basket=parse_list(basket_text),
    min_tenor_months=int(min_tenor),
    top_n_zscore=int(top_z),
    top_n_percentile=int(top_percentile),
    top_n_carry_vol=int(top_carry_vol),
    top_n_composite=int(top_n),
    weight_levels=weight_levels,
    weight_usd=weight_usd,
    weight_basket=weight_basket,
    weight_carry_vol=weight_carry,
    use_usd_benchmark=use_usd_benchmark,
)

if st.sidebar.button("Recalculate in Python", type="primary", use_container_width=True):
    calculate.clear()
    st.rerun()

model = calculate(str(workbook_path), workbook_path.stat().st_mtime_ns, controls, CACHE_VERSION)
currencies = list(model.raw)
views = ["Top composite", "Analysis", "Watchlist", "Strategy builder", "Tenor explorer"]
as_of_dates = [frame["Date"].max() for frame in model.raw.values() if not frame.empty]
as_of = max(as_of_dates) if as_of_dates else None
st.title("Carry & Rolldown")
st.caption("Excel supplies raw rates. All downstream calculations run in Python.")
st.caption(f"As of: {as_of:%d %b %Y}" if as_of is not None else "As of: unavailable")
query_currency = query_value("currency")
query_view = query_value("view")
query_trade = query_value("trade")
currency = st.sidebar.selectbox("Currency", currencies, index=currencies.index(query_currency) if query_currency in currencies else 0)
view = st.sidebar.selectbox("Python output", views, index=views.index(query_view) if query_view in views else 0)

st.query_params.update({
    "view": view,
    "currency": currency,
    "basket": basket_text,
    "levels": levels_text,
    "spreads": spreads_text,
    "flies": flies_text,
    "lookback": int(lookback),
    "horizon": int(horizon),
    "min_tenor": int(min_tenor),
    "top_z": int(top_z),
    "top_percentile": int(top_percentile),
    "top_carry_vol": int(top_carry_vol),
    "top_n": int(top_n),
    "use_usd": str(use_usd_benchmark).lower(),
    "weight_levels": weight_levels,
    "weight_usd": weight_usd,
    "weight_basket": weight_basket,
    "weight_carry": weight_carry,
})

if view == "Top composite":
    st.subheader("Cross-currency composite rankings")
    ranking_data = model.cross.copy()

    def ranking_table(title: str, ranked: pd.DataFrame) -> None:
        display = ranked.copy()
        display["Open history"] = [f"?view=Top+composite&trade={index}" for index in display.index]
        st.subheader(title)
        st.dataframe(
            display,
            use_container_width=True,
            hide_index=True,
            column_config={"Open history": st.column_config.LinkColumn("Open history", display_text="Open history")},
        )

    ranking_columns = [
        "Trade", "Direction", "Ccy 1", "Leg 1", "Ccy 2", "Leg 2", "Level RV", "Z-Score", "Percentile",
        "Carry (bp)", "Roll (bp)", "C&R (bp)", "C&R Vol (bp)", "C&R/Vol", "USD Level", "USD Z", "USD %ile",
        "Basket Level", "Basket Z", "Basket %ile", "USD RV", "Basket RV", "COMPOSITE",
    ]
    available_columns = [column for column in ranking_columns if column in ranking_data.columns]
    for column in ("Z-Score", "Percentile", "C&R/Vol", "COMPOSITE"):
        if column in ranking_data.columns:
            ranking_data[column] = pd.to_numeric(ranking_data[column], errors="coerce")
    required_ranking_columns = ["Z-Score", "Percentile", "C&R/Vol", "COMPOSITE"]
    missing_ranking_columns = [column for column in required_ranking_columns if column not in ranking_data.columns]
    if missing_ranking_columns:
        for column in missing_ranking_columns:
            ranking_data[column] = pd.NA

    if query_trade is not None:
        try:
            trade_index = int(query_trade)
        except ValueError:
            trade_index = -1
        if 0 <= trade_index < len(model.cross):
            selected_trade = model.cross.iloc[trade_index]
            left_history = model.final[selected_trade["Ccy 1"]].set_index("Date")
            right_history = model.final[selected_trade["Ccy 2"]].set_index("Date")
            left_leg = selected_trade["Leg 1"]
            right_leg = selected_trade["Leg 2"]
            history = pd.concat(
                {
                    "Level RV": left_history[left_leg] - right_history[right_leg],
                    "Carry (bp)": left_history[f"{left_leg} Carry"] - right_history[f"{right_leg} Carry"],
                    "Roll (bp)": left_history[f"{left_leg} Roll"] - right_history[f"{right_leg} Roll"],
                    "C&R (bp)": left_history[f"{left_leg} C&R"] - right_history[f"{right_leg} C&R"],
                },
                axis=1,
                join="inner",
            ).dropna(how="all")
            history["Carry and Roll (bp)"] = history["Carry (bp)"] + history["Roll (bp)"]
            st.subheader(f"{selected_trade['Direction']}: {selected_trade['Trade']}")
            st.dataframe(history.reset_index(), use_container_width=True, hide_index=True)
            plot_history(history, ["Level RV"], "Cross-currency historical level")
            plot_history(history, ["Carry and Roll (bp)"], "Cross-currency historical carry and roll")
        else:
            st.info("The selected cross-currency trade is no longer available.")

    def ranked_rows(metric: str, count: int, ascending: bool = False) -> pd.DataFrame:
        valid = ranking_data.dropna(subset=[metric])
        if valid.empty:
            return valid
        return valid.nsmallest(count, metric) if ascending else valid.nlargest(count, metric)

    composite_rows = ranked_rows("COMPOSITE", int(top_n))
    if composite_rows.empty:
        st.info("No cross-currency composite scores are available for the selected structures and lookback.")
    else:
        ranking_table(
            f"Top {int(top_n)} cross-currency composite score",
            composite_rows[available_columns],
        )

    positive_z = ranking_data[ranking_data["Z-Score"] > 0].dropna(subset=["Z-Score"])
    ranking_table(f"Top {int(top_z)} positive cross-currency Z-score", positive_z.nlargest(int(top_z), "Z-Score")[available_columns])
    ranking_table(f"Bottom {int(top_z)} cross-currency Z-score", ranked_rows("Z-Score", int(top_z), ascending=True)[available_columns])
    ranking_table(f"Top {int(top_percentile)} cross-currency percentile", ranked_rows("Percentile", int(top_percentile))[available_columns])
    ranking_table(f"Bottom {int(top_percentile)} cross-currency percentile", ranked_rows("Percentile", int(top_percentile), ascending=True)[available_columns])
    ranking_table(f"Top {int(top_carry_vol)} cross-currency C&R / C&R volatility", ranked_rows("C&R/Vol", int(top_carry_vol))[available_columns])
elif view == "Analysis":
    st.subheader(f"{currency} Python analysis")
    analysis = model.analysis[currency]
    analysis_tabs = st.tabs(["Levels", "Spreads", "Flies"])
    for tab, group in zip(analysis_tabs, ("Levels", "Spreads", "Flies")):
        with tab:
            frame = analysis[analysis["Group"] == group].drop(columns=["Group"], errors="ignore")
            structures = frame["Structure"].tolist()
            frame["Open history"] = [
                f"?view=Analysis&currency={quote(currency)}&group={quote(group)}&structure={quote(structure)}"
                for structure in structures
            ]
            st.dataframe(
                frame,
                use_container_width=True,
                hide_index=True,
                column_config={"Open history": st.column_config.LinkColumn("Open history", display_text="Open history")},
            )
            query_structure = query_value("structure")
            query_group = query_value("group")
            if query_group == group and query_structure in structures:
                selected_structure = query_structure
                history = model.final[currency].set_index("Date")
                history_columns = [column for column in (selected_structure, f"{selected_structure} C&R") if column in history.columns]
                detail = history[history_columns].copy()
                st.subheader(f"{currency} {selected_structure} history")
                st.dataframe(detail.reset_index(), use_container_width=True, hide_index=True)
                if selected_structure in detail.columns:
                    plot_history(detail, [selected_structure], f"{currency} {selected_structure} historical level")
                cr_column = f"{selected_structure} C&R"
                if cr_column in detail.columns:
                    plot_history(detail, [cr_column], f"{currency} {selected_structure} historical C&R")
            else:
                st.info("Click a structure name in this table to open its history.")
elif view == "Watchlist":
    st.subheader("Cross-currency watchlist")
    st.caption("Select currencies, structures, and fields to build one combined watchlist.")

    selected_currencies: list[str] = []
    currency_columns = st.columns(max(1, min(5, len(currencies))))
    for index, item in enumerate(currencies):
        with currency_columns[index % len(currency_columns)]:
            if st.checkbox(item, value=False, key=f"watchlist_currency_{item}"):
                selected_currencies.append(item)

    selected_structures: list[str] = []
    structure_tabs = st.tabs(["Levels", "Spreads", "Flies"])
    for tab, group in zip(structure_tabs, ("Levels", "Spreads", "Flies")):
        with tab:
            structures = list(dict.fromkeys(
                structure
                for item in currencies
                for structure in model.analysis[item].loc[model.analysis[item]["Group"] == group, "Structure"]
            ))
            structure_columns = st.columns(max(1, min(4, len(structures))))
            for index, structure in enumerate(structures):
                with structure_columns[index % len(structure_columns)]:
                    if st.checkbox(structure, value=False, key=f"watchlist_structure_{structure}"):
                        selected_structures.append(structure)

    field_options = [column for column in model.analysis[currencies[0]].columns if column not in ("Group", "Structure")]
    selected_fields: list[str] = []
    field_columns = st.columns(4)
    for index, field in enumerate(field_options):
        with field_columns[index % len(field_columns)]:
            if st.checkbox(field, value=False, key=f"watchlist_field_{field}"):
                selected_fields.append(field)

    if selected_currencies and selected_structures and selected_fields:
        watchlist_rows = []
        for item in selected_currencies:
            frame = model.analysis[item]
            selected = frame[frame["Structure"].isin(selected_structures)].copy()
            selected.insert(0, "Currency", item)
            watchlist_rows.append(selected)
        watchlist = pd.concat(watchlist_rows, ignore_index=True)
        st.dataframe(watchlist[["Currency", "Structure", *selected_fields]], use_container_width=True, hide_index=True)
    else:
        st.info("Select at least one currency, structure, and field to populate the watchlist.")
elif view == "Strategy builder":
    st.subheader("Strategy builder")
    st.caption("Build a weighted receive/pay strategy from any currency structures.")
    structure_options = sorted({structure for item in currencies for structure in model.analysis[item]["Structure"]})
    leg_rows = []
    for leg_number in range(1, 5):
        enabled = st.checkbox(f"Include leg {leg_number}", value=leg_number <= 2, key=f"strategy_enabled_{leg_number}")
        if not enabled:
            continue
        leg_columns = st.columns(4)
        with leg_columns[0]:
            action = st.selectbox("Action", ["Receive", "Pay"], key=f"strategy_action_{leg_number}")
        with leg_columns[1]:
            weight = st.number_input("Weight", value=1.0, step=0.25, key=f"strategy_weight_{leg_number}")
        with leg_columns[2]:
            leg_currency = st.selectbox("Currency", currencies, key=f"strategy_currency_{leg_number}")
        with leg_columns[3]:
            structure = st.selectbox("Structure", structure_options, key=f"strategy_structure_{leg_number}")
        sign = 1.0 if action == "Receive" else -1.0
        row = model.analysis[leg_currency].set_index("Structure").loc[structure]
        leg_rows.append({"Action": action, "Weight": weight, "Currency": leg_currency, "Structure": structure, "Sign": sign, "Analysis": row})

    if leg_rows:
        metric_names = ["Current", "Average", "StDev", "Z-Score", "Percentile", "Low", "High", "Daily Vol", "Carry (bp)", "Roll (bp)", "C&R (bp)", "C&R Z-Score", "C&R/Vol", "USD RV", "Basket RV", "COMPOSITE"]
        strategy_values = {}
        usd_values = {}
        basket_values = {}
        basket_currencies = set(controls.basket)
        usd_analysis = model.analysis.get("USD", pd.DataFrame()).set_index("Structure")
        for metric in metric_names:
            strategy_terms = [float(leg["Weight"]) * leg["Sign"] * float(leg["Analysis"].get(metric)) for leg in leg_rows if pd.notna(leg["Analysis"].get(metric))]
            usd_terms = [float(leg["Weight"]) * leg["Sign"] * float(usd_analysis.loc[leg["Structure"], metric]) for leg in leg_rows if leg["Structure"] in usd_analysis.index and pd.notna(usd_analysis.loc[leg["Structure"], metric])]
            basket_terms = [float(leg["Weight"]) * leg["Sign"] * float(leg["Analysis"].get(metric)) for leg in leg_rows if leg["Currency"] in basket_currencies and pd.notna(leg["Analysis"].get(metric))]
            strategy_values[metric] = sum(strategy_terms) if strategy_terms else None
            usd_values[metric] = sum(usd_terms) if usd_terms else None
            basket_values[metric] = sum(basket_terms) if basket_terms else None
        level_rv = strategy_values.get("Z-Score")
        usd_rv = usd_values.get("Z-Score") if use_usd_benchmark else 0.0
        basket_rv = basket_values.get("Z-Score")
        carry_vol = strategy_values.get("C&R/Vol")
        composite_terms = [
            controls.weight_levels * level_rv if level_rv is not None else None,
            controls.weight_usd * usd_rv if usd_rv is not None else None,
            controls.weight_basket * basket_rv if basket_rv is not None else None,
            controls.weight_carry_vol * carry_vol if carry_vol is not None else None,
        ]
        composite = sum(composite_terms) if all(value is not None and pd.notna(value) for value in composite_terms) else None
        component_summary = pd.DataFrame({
            "Component": ["Level RV", "USD RV", "Basket RV", "C&R / Vol", "Composite score"],
            "Value": [level_rv, usd_rv, basket_rv, carry_vol, composite],
            "Weight": [controls.weight_levels, controls.weight_usd, controls.weight_basket, controls.weight_carry_vol, None],
        })
        st.subheader("Strategy composite")
        st.dataframe(component_summary, use_container_width=True, hide_index=True)
        st.caption("One composite score is calculated from the weighted Level RV, USD RV, Basket RV, and C&R / Vol components.")
        st.subheader("Strategy component diagnostics")
        st.dataframe(pd.DataFrame([strategy_values, usd_values, basket_values], index=["Strategy legs", "USD same legs", "Basket same legs"]), use_container_width=True)
        st.dataframe(pd.DataFrame([{key: value for key, value in leg.items() if key in ("Action", "Weight", "Currency", "Structure")} for leg in leg_rows]), use_container_width=True, hide_index=True)

        strategy_history_frames = []
        for leg in leg_rows:
            leg_history = model.final[leg["Currency"]].set_index("Date")
            structure = leg["Structure"]
            strategy_history_frames.append(
                pd.DataFrame(
                    {
                        f"{leg['Action']} {leg['Currency']} {structure} level": leg_history[structure] * leg["Weight"] * leg["Sign"],
                        f"{leg['Action']} {leg['Currency']} {structure} carry": leg_history[f"{structure} Carry"] * leg["Weight"] * leg["Sign"],
                        f"{leg['Action']} {leg['Currency']} {structure} roll": leg_history[f"{structure} Roll"] * leg["Weight"] * leg["Sign"],
                    }
                )
            )
        strategy_history = pd.concat(strategy_history_frames, axis=1).sort_index()
        strategy_history["Strategy level"] = strategy_history.filter(regex=" level$").sum(axis=1)
        strategy_history["Strategy carry"] = strategy_history.filter(regex=" carry$").sum(axis=1)
        strategy_history["Strategy roll"] = strategy_history.filter(regex=" roll$").sum(axis=1)
        strategy_history["Strategy C&R"] = strategy_history["Strategy carry"] + strategy_history["Strategy roll"]
        st.subheader("Strategy history")
        st.dataframe(strategy_history[["Strategy level", "Strategy carry", "Strategy roll"]].reset_index(), use_container_width=True, hide_index=True)
        plot_history(strategy_history, ["Strategy level"], "Weighted strategy historical level")
        plot_history(strategy_history, ["Strategy C&R"], "Weighted strategy historical carry and roll")
    else:
        st.info("Include at least one strategy leg.")
elif view == "Tenor explorer":
    st.subheader(f"{currency} tenor explorer")
    chart_frame = model.final[currency].set_index("Date")
    level_options = [column for column in chart_frame.columns if column in controls.levels]
    structure_options = [column for column in chart_frame.columns if column in controls.spreads or column in controls.flies]
    tenor_options = level_options + structure_options
    selected_tenor = st.selectbox("Select tenor or structure", tenor_options, key=f"explorer_tenor_{currency}")
    metric_tabs = st.tabs(["Levels", "C&R"])
    with metric_tabs[0]:
        level_series = pd.to_numeric(chart_frame[selected_tenor], errors="coerce").dropna()
        level_stats = pd.DataFrame({
            "Date": level_series.index,
            selected_tenor: level_series.values,
        }).set_index("Date")
        mean = float(level_series.mean())
        stdev = float(level_series.std())
        for label, value in (("Mean", mean), ("+1 SD", mean + stdev), ("-1 SD", mean - stdev), ("+2 SD", mean + 2 * stdev), ("-2 SD", mean - 2 * stdev)):
            level_stats[label] = value
        st.dataframe(level_stats.reset_index(), use_container_width=True, hide_index=True)
        plot_history(level_stats, [selected_tenor, "Mean", "+1 SD", "-1 SD", "+2 SD", "-2 SD"], f"{currency} {selected_tenor} historical level")
    with metric_tabs[1]:
        cr_column = f"{selected_tenor} C&R"
        if cr_column not in chart_frame.columns:
            st.info("C&R is not available for this selection.")
        else:
            cr_series = pd.to_numeric(chart_frame[cr_column], errors="coerce").dropna()
            cr_stats = pd.DataFrame({"Date": cr_series.index, cr_column: cr_series.values}).set_index("Date")
            mean = float(cr_series.mean())
            stdev = float(cr_series.std())
            for label, value in (("Mean", mean), ("+1 SD", mean + stdev), ("-1 SD", mean - stdev), ("+2 SD", mean + 2 * stdev), ("-2 SD", mean - 2 * stdev)):
                cr_stats[label] = value
            st.dataframe(cr_stats.reset_index(), use_container_width=True, hide_index=True)
            plot_history(cr_stats, [cr_column, "Mean", "+1 SD", "-1 SD", "+2 SD", "-2 SD"], f"{currency} {selected_tenor} historical C&R")
st.sidebar.divider()
st.sidebar.caption(f"Python source: {workbook_path.name} | Excel outputs and VBA are not used")
