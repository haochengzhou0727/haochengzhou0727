from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd

DEFAULT_WORKBOOK = Path(__file__).with_name("raw data.xlsx")
TENOR_MONTHS = {"1M": 1, "3M": 3, "6M": 6, "9M": 9, "1Y": 12, "2Y": 24, "3Y": 36, "4Y": 48, "5Y": 60, "7Y": 84, "10Y": 120}
DEFAULT_LEVELS = ["3M", "6M", "9M", "1Y", "2Y", "3Y", "4Y", "5Y", "7Y", "10Y"]
DEFAULT_SPREADS = ["1s2s", "2s3s", "2s5s", "3s5s", "2s10s"]
DEFAULT_FLIES = ["1s2s3s", "1s2s5s", "3s4s5s", "2s3s5s", "2s5s10s", "5s7s10s"]
DEFAULT_BASKET = ["SGD", "MYR", "THB", "IDR"]
PAYMENT_FREQUENCY = {"THB": 1, "SGD": 2, "MYR": 4, "USD": 1, "IDR": 2}
DEFAULT_STARTING_TENOR = {"THB": 1, "SGD": 1, "MYR": 6, "USD": 1, "IDR": 1}


@dataclass(frozen=True)
class ModelControls:
    lookback_months: int = 6
    carry_horizon_months: int = 3
    levels: tuple[str, ...] = tuple(DEFAULT_LEVELS)
    spreads: tuple[str, ...] = tuple(DEFAULT_SPREADS)
    flies: tuple[str, ...] = tuple(DEFAULT_FLIES)
    basket: tuple[str, ...] = tuple(DEFAULT_BASKET)
    min_tenor_months: int = 12
    top_n_zscore: int = 3
    top_n_percentile: int = 3
    top_n_carry_vol: int = 3
    top_n_composite: int = 5
    weight_levels: float = 0.5
    weight_usd: float = 0.7
    weight_basket: float = 0.3
    weight_carry_vol: float = 0.15
    use_usd_benchmark: bool = True


@dataclass
class PythonModel:
    workbook_path: Path
    raw: dict[str, pd.DataFrame]
    cleaned: dict[str, pd.DataFrame] = field(default_factory=dict)
    carry: dict[str, pd.DataFrame] = field(default_factory=dict)
    rolldown: dict[str, pd.DataFrame] = field(default_factory=dict)
    final: dict[str, pd.DataFrame] = field(default_factory=dict)
    analysis: dict[str, pd.DataFrame] = field(default_factory=dict)
    cross: pd.DataFrame = field(default_factory=pd.DataFrame)


def _tenor(value: str) -> str:
    value = value.strip().upper()
    return value[:-1] + "Y" if value.endswith("S") else value


def _excel_date(value: Any) -> pd.Timestamp | None:
    if pd.isna(value):
        return None
    if isinstance(value, (int, float, np.number)) and not isinstance(value, bool):
        return pd.Timestamp("1899-12-30") + pd.to_timedelta(float(value), unit="D")
    try:
        return pd.Timestamp(value).normalize()
    except (TypeError, ValueError):
        return None


def _excel_number(value: Any) -> float | None:
    if pd.isna(value):
        return None
    if isinstance(value, (pd.Timestamp, datetime, date)):
        # Excel's phantom 1900-02-29 shifts date-formatted values below 60.
        epoch = pd.Timestamp("1899-12-31") if value.year == 1900 else pd.Timestamp("1899-12-30")
        return float((value - epoch).total_seconds() / 86400)
    if isinstance(value, (int, float, np.number)) and not isinstance(value, bool):
        return float(value)
    return None


def _parse_raw_sheet(frame: pd.DataFrame) -> pd.DataFrame:
    headers = frame.iloc[7] if len(frame) > 7 else pd.Series(dtype=object)
    columns: dict[str, dict[pd.Timestamp, float]] = {}
    date_series: list[pd.Timestamp] = []
    master_length: int | None = None
    for index in range(0, frame.shape[1] - 1, 2):
        header = str(headers.iloc[index + 1]) if index + 1 < len(headers) else ""
        if "(" in header and ")" in header:
            header = header.split("(", 1)[1].split(")", 1)[0].strip()
        header = _tenor(header)
        if header not in TENOR_MONTHS:
            continue
        values: dict[pd.Timestamp, float] = {}
        pair_rows = list(frame.iloc[8:, [index, index + 1]].itertuples(index=False, name=None))
        raw_dates = [parsed for date, _ in pair_rows if (parsed := _excel_date(date)) is not None]
        for date, value in pair_rows:
            numeric_value = _excel_number(value)
            if pd.isna(date) or numeric_value is None:
                continue
            value = numeric_value
            if value < -5 or value > 30:
                continue
            parsed_date = _excel_date(date)
            if parsed_date is not None:
                values.setdefault(parsed_date, value)
        columns.setdefault(header, values)
        if master_length is None or len(raw_dates) < master_length:
            master_length = len(raw_dates)
            date_series = raw_dates
    rows = [{"Date": date, **{header: values.get(date, np.nan) for header, values in columns.items()}} for date in date_series]
    if not rows or not date_series:
        raise ValueError("No dated rate observations found in raw sheet")
    result = pd.DataFrame(rows).groupby("Date", as_index=False).first()
    value_columns = [column for column in result.columns if column != "Date"]
    result = result.dropna(subset=value_columns, how="any")
    return result.sort_values("Date").reset_index(drop=True)


def load_raw_workbook(path: str | Path = DEFAULT_WORKBOOK) -> dict[str, pd.DataFrame]:
    path = Path(path)
    excel = pd.ExcelFile(path, engine="openpyxl")
    raw: dict[str, pd.DataFrame] = {}
    for currency in ("IDR", "USD", "THB", "SGD", "MYR"):
        sheet = f"{currency} Raw"
        if sheet in excel.sheet_names:
            raw[currency] = _parse_raw_sheet(pd.read_excel(path, sheet_name=sheet, header=None, engine="openpyxl"))
    return raw


def load_curve_config(path: str | Path) -> dict[str, tuple[int, int]]:
    frame = pd.read_excel(path, sheet_name="Curve", header=None, engine="openpyxl")
    config: dict[str, tuple[int, int]] = {}
    for row in frame.itertuples(index=False, name=None):
        if len(row) < 3 or pd.isna(row[0]) or pd.isna(row[1]) or pd.isna(row[2]):
            continue
        currency = str(row[0]).strip().upper()
        try:
            frequency = int(row[1])
            start_text = str(row[2]).strip().upper()
            match = re.fullmatch(r"(\d+)([MY])", start_text)
            if match:
                start_months = int(match.group(1)) * (12 if match.group(2) == "Y" else 1)
                config[currency] = (frequency, start_months)
        except (TypeError, ValueError):
            continue
    return config


def _interpolate(curve: pd.Series, months: float, starting_months: int) -> float:
    points = sorted((TENOR_MONTHS[key], float(value)) for key, value in curve.items() if key in TENOR_MONTHS and pd.notna(value))
    if not points:
        return np.nan
    start = next((value for tenor, value in points if tenor == starting_months), np.nan)
    if months < starting_months:
        return float(start)
    points = [(tenor, value) for tenor, value in points if tenor >= starting_months]
    if not points:
        return np.nan
    x, y = zip(*points)
    return float(np.interp(months, x, y))


def _discount_factors(curve: pd.Series, frequency: int, starting_months: int) -> tuple[dict[int, float], int]:
    period = 12 // frequency
    max_months = max((TENOR_MONTHS[key] for key in curve.index if key in TENOR_MONTHS and pd.notna(curve[key])), default=0)
    grid_max = (max_months // period) * period
    factors: dict[int, float] = {0: 1.0}
    sum_pv = 0.0
    for months in range(period, grid_max + 1, period):
        rate = _interpolate(curve, months, starting_months) / 100
        alpha = period / 12
        factors[months] = (1 - rate * sum_pv) / (1 + rate * alpha)
        if factors[months] <= 0:
            return {}, grid_max
        sum_pv += alpha * factors[months]
    return factors, grid_max


def _discount_factor(factors: dict[int, float], months: float) -> float:
    points = [(key, value) for key, value in factors.items() if pd.notna(value)]
    if not points:
        return np.nan
    x, y = zip(*sorted(points))
    if months in factors:
        return float(factors[months])
    step = int(min(np.diff(x))) if len(x) > 1 else 0
    lower = int(months // step) * step if step else 0
    upper = lower + step
    if upper not in factors or factors.get(lower, 0) <= 0 or factors[upper] <= 0:
        return np.nan
    return float(factors[lower] * np.exp(np.log(factors[upper] / factors[lower]) * (months - lower) / (upper - lower)))


def _par_rate(curve: pd.Series, months: int, starting_months: int) -> float:
    exact = curve.get(next((key for key, value in TENOR_MONTHS.items() if value == months), ""), np.nan)
    if pd.notna(exact):
        return float(exact)
    return _interpolate(curve, months, starting_months)


def _exact_rate(curve: pd.Series, months: int) -> float:
    key = next((key for key, value in TENOR_MONTHS.items() if value == months), "")
    value = curve.get(key, np.nan)
    return float(value) if pd.notna(value) else np.nan


def _df_at(curve: pd.Series, factors: dict[int, float], months: int, frequency: int, starting_months: int, grid_max: int) -> float:
    if months == 0:
        return 1.0
    period = 12 // frequency
    if months <= grid_max and months % period == 0:
        return factors.get(months, np.nan)
    if starting_months <= months < period:
        rate = _exact_rate(curve, months)
        if pd.notna(rate):
            return 1 / (1 + rate / 100 * months / 12)
    lower = (months // period) * period
    upper = lower + period
    if upper <= grid_max:
        lower_df = factors.get(lower, 1.0)
        upper_df = factors.get(upper, np.nan)
        if lower_df > 0 and upper_df > 0:
            return lower_df * np.exp(np.log(upper_df / lower_df) * (months - lower) / (upper - lower))
    rate = _par_rate(curve, months, starting_months)
    if pd.isna(rate):
        return np.nan
    sum_pv = 0.0
    previous = 0
    for payment in range(period, months, period):
        payment_df = _df_at(curve, factors, payment, frequency, starting_months, grid_max)
        if not pd.notna(payment_df) or payment_df <= 0:
            return np.nan
        sum_pv += (payment - previous) / 12 * payment_df
        previous = payment
    accrual = (months - previous) / 12
    return (1 - rate / 100 * sum_pv) / (1 + rate / 100 * accrual)


def _swap_rate_from_factors(curve: pd.Series, factors: dict[int, float], start_months: int, end_months: int, frequency: int, starting_months: int, grid_max: int) -> float:
    if end_months <= start_months:
        return np.nan
    period = 12 // frequency
    start_factor = _df_at(curve, factors, start_months, frequency, starting_months, grid_max)
    end_factor = _df_at(curve, factors, end_months, frequency, starting_months, grid_max)
    denominator = 0.0
    previous = start_months
    for payment in range(start_months + period, end_months, period):
        denominator += (payment - previous) / 12 * _df_at(curve, factors, payment, frequency, starting_months, grid_max)
        previous = payment
    denominator += (end_months - previous) / 12 * end_factor
    return (start_factor - end_factor) / denominator if denominator else np.nan


def _outright_rate(curve: pd.Series, factors: dict[int, float], months: int, frequency: int, starting_months: int, grid_max: int) -> float:
    return _swap_rate_from_factors(curve, factors, 0, months, frequency, starting_months, grid_max) * 100


def _aged_spot_rate(curve: pd.Series, factors: dict[int, float], months: int, frequency: int, starting_months: int, grid_max: int) -> float:
    exact = _exact_rate(curve, months)
    if pd.notna(exact):
        return exact
    return _outright_rate(curve, factors, months, frequency, starting_months, grid_max)


def _structure_value(curve: pd.Series, factors: dict[int, float], structure: str, frequency: int, starting_months: int, grid_max: int, shift_months: int = 0) -> float:
    tokens = [f"{number}Y" for number in re.findall(r"\d+", structure)]
    values = [_outright_rate(curve, factors, TENOR_MONTHS[token] + shift_months, frequency, starting_months, grid_max) for token in tokens if token in TENOR_MONTHS and TENOR_MONTHS[token] + shift_months > 0]
    if len(values) == 1:
        return values[0]
    if len(values) == 2:
        return values[1] - values[0]
    if len(values) == 3:
        return values[0] - 2 * values[1] + values[2]
    return np.nan


def _combine_structure(values: dict[str, float], structure: str) -> float:
    legs = [f"{number}Y" for number in re.findall(r"\d+", structure)]
    if any(leg not in values or pd.isna(values[leg]) for leg in legs):
        return np.nan
    if len(legs) == 2:
        return values[legs[1]] - values[legs[0]]
    if len(legs) == 3:
        return 2 * values[legs[1]] - values[legs[0]] - values[legs[2]]
    return np.nan


def _stats(values: pd.Series, current: float) -> dict[str, float]:
    values = pd.to_numeric(values, errors="coerce").dropna()
    mean = float(values.mean()) if len(values) else np.nan
    std = float(values.std(ddof=1)) if len(values) > 1 else np.nan
    z = (current - mean) / std if std and pd.notna(std) else np.nan
    return {"Current": current, "Average": mean, "StDev": std, "Z-Score": z, "Percentile": float((values <= current).mean()) if len(values) else np.nan, "Low": float(values.min()) if len(values) else np.nan, "High": float(values.max()) if len(values) else np.nan, "N": float(len(values))}


def build_analysis(model: PythonModel, currency: str, controls: ModelControls) -> pd.DataFrame:
    final = model.final[currency]
    end = final["Date"].max()
    history = final[final["Date"].between(end - pd.DateOffset(months=controls.lookback_months), end)]
    level_names = [name for name in controls.levels if name in final.columns]
    spread_names = [name for name in controls.spreads if name in final.columns]
    fly_names = [name for name in controls.flies if name in final.columns]
    names = level_names + spread_names + fly_names
    rows = []
    for name in names:
        level = pd.to_numeric(history[name], errors="coerce").dropna()
        current = float(level.iloc[-1]) if len(level) else np.nan
        cr = pd.to_numeric(history.get(f"{name} C&R", pd.Series(dtype=float)), errors="coerce").dropna()
        cr_current = float(cr.iloc[-1]) if len(cr) else np.nan
        vol = float(level.diff().std()) if len(level) > 1 else np.nan
        row = {"Structure": name, **_stats(level, current), "Carry (bp)": np.nan, "Roll (bp)": np.nan, "C&R (bp)": cr_current, "C&R Z-Score": _stats(cr, cr_current)["Z-Score"], "Daily Vol": vol}
        for suffix in (" Carry", " Roll"):
            series = pd.to_numeric(history.get(name + suffix, pd.Series(dtype=float)), errors="coerce").dropna()
            row[f"{suffix.strip()} (bp)"] = float(series.iloc[-1]) if len(series) else np.nan
        row["C&R/Vol"] = row["C&R (bp)"] / vol if vol and pd.notna(vol) else np.nan
        row["COMPOSITE"] = controls.weight_levels * row["Z-Score"] + controls.weight_carry_vol * row["C&R/Vol"] if pd.notna(row["Z-Score"]) and pd.notna(row["C&R/Vol"]) else row["Z-Score"]
        rows.append(row)
    result = pd.DataFrame(rows)
    category_order = {name: "Levels" for name in level_names} | {name: "Spreads" for name in spread_names} | {name: "Flies" for name in fly_names}
    result["_Category"] = result["Structure"].map(category_order)
    result["_Order"] = result["Structure"].map({name: index for index, name in enumerate(names)})
    return result.sort_values(["_Order"]).rename(columns={"_Category": "Group"}).drop(columns=["_Order"]).reset_index(drop=True)


def build_cross_analysis(model: PythonModel, controls: ModelControls) -> pd.DataFrame:
    rows = []
    currencies = list(model.analysis)
    structures = list(controls.spreads) + list(controls.flies)
    for left_index, left in enumerate(currencies):
        for right in currencies[left_index + 1:]:
            left_frame = model.analysis[left].set_index("Structure")
            right_frame = model.analysis[right].set_index("Structure")
            for left_structure in structures:
                for right_structure in structures:
                    if left_structure not in left_frame.index or right_structure not in right_frame.index:
                        continue
                    l, r = left_frame.loc[left_structure], right_frame.loc[right_structure]
                    z = l["Z-Score"] - r["Z-Score"]
                    carry_vol = l["C&R/Vol"] - r["C&R/Vol"]
                    usd_frame = model.analysis.get("USD", pd.DataFrame()).set_index("Structure")
                    usd_z = 0.0
                    if left_structure in usd_frame.index and right_structure in usd_frame.index:
                        usd_z = float(usd_frame.loc[left_structure, "Z-Score"] - usd_frame.loc[right_structure, "Z-Score"])
                    usd_adjustment = controls.weight_usd * usd_z if controls.use_usd_benchmark else 0.0
                    basket_adjustment = 0.0 if left in controls.basket and right in controls.basket else -controls.weight_basket * z
                    composite = controls.weight_levels * z + usd_adjustment + basket_adjustment + controls.weight_carry_vol * carry_vol
                    rows.append({"Trade": f"{left} {left_structure} vs {right} {right_structure}", "Direction": f"Receive {left} / Pay {right}", "Ccy 1": left, "Leg 1": left_structure, "Ccy 2": right, "Leg 2": right_structure, "Z-Score": z, "C&R/Vol": carry_vol, "USD RV": usd_adjustment, "Basket RV": basket_adjustment, "COMPOSITE": composite})
    if not rows:
        return pd.DataFrame(columns=["Trade", "Direction", "Ccy 1", "Leg 1", "Ccy 2", "Leg 2", "Z-Score", "C&R/Vol", "COMPOSITE"])
    return pd.DataFrame(rows).sort_values("COMPOSITE", ascending=False).head(controls.top_n_composite).reset_index(drop=True)


def build_model(path: str | Path = DEFAULT_WORKBOOK, controls: ModelControls = ModelControls()) -> PythonModel:
    model = PythonModel(Path(path), load_raw_workbook(path))
    curve_config = load_curve_config(path)
    for currency, clean in model.raw.items():
        model.cleaned[currency] = clean
        frequency, starting_months = curve_config.get(currency, (PAYMENT_FREQUENCY[currency], DEFAULT_STARTING_TENOR[currency]))
        carry_rows, roll_rows, final_rows = [], [], []
        for _, row in clean.iterrows():
            curve = row.drop(labels="Date")
            factors, grid_max = _discount_factors(curve, frequency, starting_months)
            carry_row, roll_row, final_row = {"Date": row["Date"]}, {"Date": row["Date"]}, {"Date": row["Date"]}
            level_values: dict[str, float] = {}
            carry_values: dict[str, float] = {}
            roll_values: dict[str, float] = {}
            for tenor in controls.levels:
                months = TENOR_MONTHS.get(_tenor(tenor))
                if months is None:
                    continue
                if _tenor(tenor) not in curve or pd.isna(curve.get(_tenor(tenor))):
                    continue
                spot = _outright_rate(curve, factors, months, frequency, starting_months, grid_max)
                final_row[tenor] = spot
                if months < 12:
                    final_row[f"{tenor} C&R"] = None
                level_values[_tenor(tenor)] = spot
                if months >= 12 and months > controls.carry_horizon_months:
                    forward = _swap_rate_from_factors(curve, factors, controls.carry_horizon_months, months, frequency, starting_months, grid_max) * 100
                    roll_rate = _aged_spot_rate(curve, factors, months - controls.carry_horizon_months, frequency, starting_months, grid_max)
                    carry = (forward - spot) * 100
                    roll = (spot - roll_rate) * 100
                    carry_row[tenor], roll_row[tenor] = carry, roll
                    final_row.update({f"{tenor} Carry": carry, f"{tenor} Roll": roll, f"{tenor} C&R": carry + roll if months >= 12 else None})
                    carry_values[_tenor(tenor)] = carry
                    roll_values[_tenor(tenor)] = roll
            for structure in list(controls.spreads) + list(controls.flies):
                level_structure = _combine_structure(level_values, structure) * 100
                carry_structure = _combine_structure(carry_values, structure)
                roll_structure = _combine_structure(roll_values, structure)
                if pd.notna(level_structure):
                    final_row.update({structure: level_structure, f"{structure} Carry": carry_structure, f"{structure} Roll": roll_structure, f"{structure} C&R": carry_structure + roll_structure})
            carry_rows.append(carry_row)
            roll_rows.append(roll_row)
            final_rows.append(final_row)
        model.carry[currency] = pd.DataFrame(carry_rows)
        model.rolldown[currency] = pd.DataFrame(roll_rows)
        model.final[currency] = pd.DataFrame(final_rows)
        model.analysis[currency] = build_analysis(model, currency, controls)
    model.cross = build_cross_analysis(model, controls)
    return model
