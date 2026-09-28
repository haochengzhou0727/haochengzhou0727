from __future__ import annotations

from pathlib import Path
from urllib.parse import quote

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from carry_dashboard import DEFAULT_BASKET, DEFAULT_FLIES, DEFAULT_LEVELS, DEFAULT_SPREADS, DEFAULT_WORKBOOK, ModelControls, PythonModel, build_model

st.set_page_config(page_title="Carry & Rolldown Python Model", page_icon="C", layout="wide")
st.title("Carry & Rolldown")
st.caption("Excel supplies raw rates. All downstream calculations run in Python.")


@st.cache_data(show_spinner="Calculating curves, carry, rolldown, and scores in Python...")
def calculate(path: str, modified_ns: int, controls: ModelControls) -> PythonModel:
    del modified_ns
    return build_model(path, controls)


def plot_history(frame: pd.DataFrame, value_columns: list[str], title: str) -> None:
    figure = go.Figure()
    for column in value_columns:
        reference_styles = {
            "Mean": {"color": "#111827", "width": 2, "dash": "solid"},
            "+1 SD": {"color": "#2563eb", "width": 1.5, "dash": "dash"},
            "-1 SD": {"color": "#2563eb", "width": 1.5, "dash": "dash"},
            "+2 SD": {"color": "#f59e0b", "width": 1.5, "dash": "dot"},
            "-2 SD": {"color": "#f59e0b", "width": 1.5, "dash": "dot"},
        }
        line = reference_styles.get(column)
        if column == "Strategy level":
            line = {"color": "#dc2626", "width": 2.5, "dash": "solid"}
        figure.add_trace(go.Scatter(x=frame.index, y=frame[column], mode="lines", name=column, line=line))
    figure.update_layout(
        title=title,
        height=560,
        hovermode="x unified",
        margin={"l": 20, "r": 20, "t": 55, "b": 20},
        xaxis={"title": "Date", "rangeslider": {"visible": True}, "type": "date"},
        yaxis={"title": "Value"},
        legend={"orientation": "h", "y": 1.02, "x": 0},
    )
    st.plotly_chart(figure, use_container_width=True, config={"scrollZoom": True, "displaylogo": False})


def query_value(name: str) -> str | None:
    value = st.query_params.get(name)
    if isinstance(value, list):
        return value[0] if value else None
    return str(value) if value is not None else None


workbook_path = Path(st.sidebar.text_input("Raw-data workbook", str(DEFAULT_WORKBOOK)))
if not workbook_path.exists():
    st.error(f"Workbook not found: {workbook_path}")
    st.stop()

st.sidebar.subheader("Python model controls")
with st.sidebar.expander("Structures and basket", expanded=True):
    basket_text = st.text_input("Basket currencies", ", ".join(DEFAULT_BASKET), help="Comma-separated currencies used for basket RV.")
    levels_text = st.text_input("Levels", ", ".join(DEFAULT_LEVELS), help="Comma-separated tenors, for example 1M, 6M, 1Y, 2Y.")
    spreads_text = st.text_input("Spreads", ", ".join(DEFAULT_SPREADS), help="Comma-separated structures, for example 1s2s, 2s5s.")
    flies_text = st.text_input("Flies", ", ".join(DEFAULT_FLIES), help="Comma-separated structures, for example 1s2s5s.")
with st.sidebar.expander("Lookback and ranking", expanded=True):
    lookback = st.number_input("Lookback (months)", min_value=1, max_value=120, value=6)
    horizon = st.number_input("Carry horizon (months)", min_value=1, max_value=24, value=3)
    min_tenor = st.number_input("Minimum cross-trade tenor (months)", min_value=1, max_value=120, value=12)
    top_z = st.number_input("Top / bottom z-score", min_value=1, max_value=50, value=3)
    top_percentile = st.number_input("Top / bottom percentile", min_value=1, max_value=50, value=3)
    top_carry_vol = st.number_input("Top C&R / volatility", min_value=1, max_value=50, value=3)
    top_n = st.number_input("Top composite trades", min_value=1, max_value=50, value=5)
with st.sidebar.expander("Composite weights", expanded=True):
    use_usd_benchmark = st.checkbox("Use USD as benchmark", value=True, help="Include USD relative-value adjustment in cross-currency composite scores. USD still has its own analysis.")
    weight_levels = st.number_input("Weight: level RV", value=0.5, step=0.05)
    weight_usd = st.number_input("Weight: USD RV", value=0.7, step=0.05)
    weight_basket = st.number_input("Weight: basket RV", value=0.3, step=0.05)
    weight_carry = st.number_input("Weight: carry and rolldown", value=0.15, step=0.05)

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

model = calculate(str(workbook_path), workbook_path.stat().st_mtime_ns, controls)
currencies = list(model.raw)
views = ["Top composite", "Analysis", "Watchlist", "Strategy builder", "Tenor explorer", "Charts", "Carry", "Rolldown", "Final", "Cleaned raw"]
query_currency = query_value("currency")
query_view = query_value("view")
currency = st.sidebar.selectbox("Currency", currencies, index=currencies.index(query_currency) if query_currency in currencies else 0)
view = st.sidebar.selectbox("Python output", views, index=views.index(query_view) if query_view in views else 0)

if view == "Top composite":
    st.subheader("Top composite cross-currency trades")
    composite_table = model.cross.copy()
    composite_table["Open history"] = [
        f"?view=Top+composite&trade={index}"
        for index in composite_table.index
    ]
    st.dataframe(
        composite_table,
        use_container_width=True,
        hide_index=True,
        column_config={"Open history": st.column_config.LinkColumn("Open history", display_text="Open history")},
    )
    if not model.cross.empty:
        query_trade = query_value("trade")
        if query_trade and query_trade.isdigit() and int(query_trade) in model.cross.index:
            trade_index = int(query_trade)
            trade = model.cross.loc[trade_index]
        else:
            trade = None
        if trade is not None:
            left_currency, right_currency = trade["Ccy 1"], trade["Ccy 2"]
            left_structure, right_structure = trade["Leg 1"], trade["Leg 2"]
            left_history = model.final[left_currency].set_index("Date")
            right_history = model.final[right_currency].set_index("Date")
            strategy_history = pd.DataFrame(index=left_history.index.union(right_history.index).sort_values())
            strategy_history[f"{left_currency} {left_structure}"] = left_history[left_structure]
            strategy_history[f"{right_currency} {right_structure}"] = right_history[right_structure]
            strategy_history["Strategy level"] = strategy_history.iloc[:, 0] - strategy_history.iloc[:, 1]
            left_cr = f"{left_structure} C&R"
            right_cr = f"{right_structure} C&R"
            if left_cr in left_history.columns and right_cr in right_history.columns:
                strategy_history["Strategy C&R"] = left_history[left_cr] - right_history[right_cr]
            st.subheader("Selected composite history")
            st.dataframe(strategy_history.reset_index(), use_container_width=True, hide_index=True)
            strategy_level = pd.to_numeric(strategy_history["Strategy level"], errors="coerce")
            strategy_level_stats = strategy_history[["Strategy level"]].copy()
            mean = float(strategy_level.mean())
            stdev = float(strategy_level.std())
            for label, value in (("Mean", mean), ("+1 SD", mean + stdev), ("-1 SD", mean - stdev), ("+2 SD", mean + 2 * stdev), ("-2 SD", mean - 2 * stdev)):
                strategy_level_stats[label] = value
            plot_history(strategy_level_stats, ["Strategy level", "Mean", "+1 SD", "-1 SD", "+2 SD", "-2 SD"], "Selected composite historical strategy level")
            if "Strategy C&R" in strategy_history:
                plot_history(strategy_history, ["Strategy C&R"], "Selected composite historical C&R")
        else:
            st.info("Click a trade name in the table to open its history.")
    st.subheader(f"Top {top_n} {currency} structures")
    st.dataframe(model.analysis[currency].head(int(top_n)), use_container_width=True, hide_index=True)
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
            if st.checkbox(item, value=True, key=f"watchlist_currency_{item}"):
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
                    if st.checkbox(structure, value=True, key=f"watchlist_structure_{structure}"):
                        selected_structures.append(structure)

    field_options = [column for column in model.analysis[currencies[0]].columns if column not in ("Group", "Structure")]
    selected_fields: list[str] = []
    field_columns = st.columns(4)
    for index, field in enumerate(field_options):
        with field_columns[index % len(field_columns)]:
            if st.checkbox(field, value=True, key=f"watchlist_field_{field}"):
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
        metric_names = ["Current", "Average", "StDev", "Z-Score", "Percentile", "Low", "High", "Daily Vol", "Carry (bp)", "Roll (bp)", "C&R (bp)", "C&R Z-Score", "C&R/Vol", "COMPOSITE"]
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
elif view == "Charts":
    st.subheader(f"{currency} historical charts")
    chart_frame = model.final[currency].set_index("Date")
    available_structures = [column for column in chart_frame.columns if " Carry" not in column and " Roll" not in column and " C&R" not in column]
    selected_chart_structures = st.multiselect("Structures", available_structures, default=available_structures[:3])
    chart_tabs = st.tabs(["Historical levels", "Historical C&R"])
    with chart_tabs[0]:
        level_columns = [structure for structure in selected_chart_structures if structure in chart_frame.columns]
        if level_columns:
            plot_history(chart_frame, level_columns, f"{currency} historical levels")
        else:
            st.info("Select at least one structure.")
    with chart_tabs[1]:
        cr_columns = [f"{structure} C&R" for structure in selected_chart_structures if f"{structure} C&R" in chart_frame.columns]
        if cr_columns:
            plot_history(chart_frame, cr_columns, f"{currency} historical C&R")
        else:
            st.info("Selected structures do not have C&R history.")
elif view == "Carry":
    st.subheader(f"{currency} Python carry")
    st.dataframe(model.carry[currency].tail(250), use_container_width=True, hide_index=True)
elif view == "Rolldown":
    st.subheader(f"{currency} Python rolldown")
    st.dataframe(model.rolldown[currency].tail(250), use_container_width=True, hide_index=True)
elif view == "Final":
    st.subheader(f"{currency} Python final")
    st.dataframe(model.final[currency].tail(250), use_container_width=True, hide_index=True)
else:
    st.subheader(f"{currency} cleaned raw rates")
    st.dataframe(model.cleaned[currency].tail(250), use_container_width=True, hide_index=True)

st.sidebar.divider()
st.sidebar.caption(f"Python source: {workbook_path.name} | Excel outputs and VBA are not used")
