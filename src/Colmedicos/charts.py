from Colmedicos.registry import register

# ===========================
# Utilidades comunes
# ===========================
import matplotlib
matplotlib.use("Agg")  # Backend no interactivo (robusto para server/exports)
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype, is_datetime64_any_dtype
from typing import List, Dict, Any, Tuple, Optional, Union
import unicodedata
from pandas.api.types import is_numeric_dtype, is_datetime64_any_dtype

# ---- Operadores soportados ----
_OPS = {
    '>':  lambda a, b: a > b,
    '<':  lambda a, b: a < b,
    '==': lambda a, b: a == b,
    '!=': lambda a, b: a != b,
    '>=': lambda a, b: a >= b,
    '<=': lambda a, b: a <= b,
    'in':     lambda a, b: a.isin(b),
    'not in': lambda a, b: ~a.isin(b),
}

def _to_numeric_like(series: pd.Series) -> pd.Series:
    """Convierte a numérico donde aplique, preservando no-numéricos como NaN."""
    if np.issubdtype(series.dtype, np.number):
        return series
    return pd.to_numeric(series, errors="coerce")


def _mask_from_conditions(df: pd.DataFrame, conds: list[list]) -> pd.Series:
    """
    Versión tolerante:
    - Si la condición está mal formada → devuelve un mask FULL FALSE → conteo 0
    - Si la columna no existe → mask = False
    - Si el operador no existe → mask = False
    - Si el valor es None o string vacío → mask = False
    """
    if not conds:
        return pd.Series(True, index=df.index)

    m = pd.Series(True, index=df.index)

    for item in conds:
        # Condiciones mal formadas → devuelven CERO
        if not isinstance(item, (list, tuple)) or len(item) != 3:
            return pd.Series(False, index=df.index)

        col, op, val = item

        # Columnas inexistentes → CERO
        if col not in df.columns:
            return pd.Series(False, index=df.index)

        # Operador inválido → CERO
        if op not in _OPS:
            return pd.Series(False, index=df.index)

        s = df[col]

        # Valor vacío o None → CERO
        if val is None or (isinstance(val, str) and val.strip() == ""):
            return pd.Series(False, index=df.index)

        # Convertir valor según tipo
        if is_numeric_dtype(s):
            val = pd.to_numeric(val, errors="coerce")
        elif is_datetime64_any_dtype(s):
            val = pd.to_datetime(val, errors="coerce")

        # Si conversión falla → CERO
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return pd.Series(False, index=df.index)

        # Aplicar condición
        m &= _OPS[op](s, val)

    return m


def _mask_from_condition_blocks(
    df: pd.DataFrame,
    conditions_all: list[list] | None = None,
    conditions_any: list[list] | list[list[list]] | None = None
) -> pd.Series:
    """
    AND de 'conditions_all' + OR de 'conditions_any'.
    'conditions_any' puede ser:
      - lista de condiciones simples, o
      - lista de bloques AND (cada bloque es [[...],[...]])
    """
    m_all = _mask_from_conditions(df, conditions_all or [])
    if not conditions_any:
        return m_all
    or_mask = pd.Series(False, index=df.index)
    for item in conditions_any:
        if isinstance(item, (list, tuple)) and item and isinstance(item[0], (list, tuple)) and len(item[0]) == 3:
            # bloque AND
            or_mask |= _mask_from_conditions(df, item)
        else:
            # condición simple
            or_mask |= _mask_from_conditions(df, [item])
    return m_all & or_mask

def make_groups_from_conditions(
    df: pd.DataFrame,
    rules: list[dict],
    new_col: str,
    categories_order: list[str] | None = None,
    fill_unmatched: str | None = None
) -> str:
    """
    Crea una columna categórica a partir de reglas condicionadas.
    rules: [
      { "label": "total_0_y_5",  "conditions_all": [["edad", ">=", 0], ["edad", "<=", 5]] },
      { "label": "total_6_y_11", "conditions_all": [["edad", ">=", 6], ["edad", "<=", 11]] },
      ...
    ]
    Prioridad: primera regla que haga match.
    """
    out = pd.Series(np.nan, index=df.index, dtype="object")
    for r in rules:
        label = r["label"]
        m = _mask_from_condition_blocks(
            df,
            r.get("conditions_all"),
            r.get("conditions_any")
        )
        out[m & out.isna()] = label  # sólo asigna donde aún no se asignó (respeta prioridad)

    if fill_unmatched is not None:
        out = out.fillna(fill_unmatched)

    # convierte a categoría ordenada si se provee el orden
    if categories_order is None:
        cats = [r["label"] for r in rules]
    else:
        cats = categories_order
    df[new_col] = pd.Categorical(out, categories=cats, ordered=True)
    return new_col

def _prefilter_df(
    df: pd.DataFrame,
    unique_by: Optional[Union[str, List[str]]] = None,
    conditions_all: Optional[List[List[Any]]] = None,
    conditions_any: Optional[List[Union[List[Any], List[List[Any]]]]] = None,
) -> pd.DataFrame:
    """Aplica condiciones AND/OR y deduplicación por unique_by (si se pide)."""
    m = _mask_from_condition_blocks(df, conditions_all, conditions_any)
    out = df.loc[m].copy()
    if unique_by:
        out = out.drop_duplicates(subset=[unique_by] if isinstance(unique_by, str) else list(unique_by))
    return out

def _ensure_xlabel(df, xlabel, *, joiner=" | ", out_col="__xlabel_combo"):
    if xlabel is None or isinstance(xlabel, str):
        return df, xlabel, None
    missing = [c for c in xlabel if c not in df.columns]
    if missing:
        raise ValueError(f"Las siguientes columnas de 'xlabel' no existen: {missing}")
    df2 = df.copy()
    df2[out_col] = (
        df2[xlabel]
        .astype(object)
        .where(pd.notna(df2[xlabel]), "")
        .astype(str)
        .agg(joiner.join, axis=1)
    )
    return df2, out_col, list(xlabel)

def _aggregate_frame(
    df: pd.DataFrame,
    xlabel: Optional[Union[str, List[str]]],
    y_cols: List[str],
    agg: Union[str, Dict[str, str]],
    *,
    distinct_on: Optional[str] = None,
    drop_dupes_before_sum: bool = False,
    where: Optional[pd.Series] = None,
) -> pd.DataFrame:
    if where is not None:
        df = df.loc[where].copy()

    # Asegurar lista de claves de agrupación
    group_keys: Optional[List[str]] = None
    if xlabel is not None:
        if isinstance(xlabel, str):
            if xlabel not in df.columns:
                raise ValueError(f"La columna '{xlabel}' no existe en el DataFrame.")
            group_keys = [xlabel]
        else:
            # lista de columnas
            missing = [c for c in xlabel if c not in df.columns]
            if missing:
                raise ValueError(f"Columnas en xlabel no existen: {missing}")
            group_keys = list(xlabel)

    if not y_cols:
        y_cols = []

    def _maybe_dedup(_df: pd.DataFrame) -> pd.DataFrame:
        if drop_dupes_before_sum and distinct_on:
            if group_keys is None:
                return _df.drop_duplicates(subset=[distinct_on])
            else:
                return _df.drop_duplicates(subset=group_keys + [distinct_on])
        return _df

    def _count_distinct_on(_df: pd.DataFrame) -> pd.DataFrame:
        if distinct_on not in _df.columns:
            raise ValueError(f"distinct_on='{distinct_on}' no existe en el DataFrame.")
        if group_keys is None:
            val = int(_df[distinct_on].nunique(dropna=True))
            return pd.DataFrame({"Número trabajadores": [val]})
        else:
            s = _df.groupby(group_keys, dropna=False)[distinct_on].nunique(dropna=True)
            return s.to_frame(name="Número trabajadores")

    if isinstance(agg, str) and distinct_on and agg in ("distinct_count", "count"):
        return _count_distinct_on(df)

    if isinstance(agg, str):
        if agg in ("distinct_count", "sum_distinct"):
            base = df[y_cols].copy() if group_keys is None else df[group_keys + y_cols].copy()
            if group_keys is None:
                res = pd.DataFrame(index=[0])
            else:
                idx = base.groupby(group_keys, dropna=False).ngroup()
                # construimos un índice MultiIndex real
                res_idx = base[group_keys].drop_duplicates().set_index(group_keys)
                res = pd.DataFrame(index=res_idx.index)

            for c in y_cols:
                if agg == "distinct_count":
                    if group_keys is None:
                        res.loc[0, c] = int(base[c].nunique(dropna=True))
                    else:
                        res[c] = base.groupby(group_keys, dropna=False)[c].nunique(dropna=True)
                else:
                    if group_keys is None:
                        res.loc[0, c] = float(_to_numeric_like(base[c].drop_duplicates()).sum())
                    else:
                        tmp = (base[group_keys + [c]]
                               .drop_duplicates(subset=group_keys + [c])
                               .groupby(group_keys, dropna=False)[c]
                               .apply(lambda s: _to_numeric_like(s).sum()))
                        res[c] = tmp
            return res

        work = _maybe_dedup(df)
        if group_keys is None:
            return work[y_cols].agg(agg).to_frame().T
        return work.groupby(group_keys, dropna=False)[y_cols].agg(agg)

    # agg dict
    work = _maybe_dedup(df)
    if group_keys is None:
        parts = []
        for c in y_cols:
            a = agg.get(c, "sum")
            if a == "distinct_count":
                parts.append(pd.Series({c: int(work[c].nunique(dropna=True))}))
            elif a == "sum_distinct":
                parts.append(pd.Series({c: float(_to_numeric_like(work[c].drop_duplicates()).sum())}))
            elif distinct_on and a in ("count", "distinct_count"):
                parts.append(pd.Series({c: int(work[distinct_on].nunique(dropna=True))}))
            else:
                parts.append(pd.Series({c: work[c].agg(a)}))
        return pd.DataFrame([pd.concat(parts)])

    g = work.groupby(group_keys, dropna=False)
    frames = []
    for c in y_cols:
        a = agg.get(c, "sum")
        if a == "distinct_count":
            s = g[c].nunique(dropna=True).rename(c)
        elif a == "sum_distinct":
            s = g[c].apply(lambda s: _to_numeric_like(s.drop_duplicates()).sum()).rename(c)
        elif distinct_on and a in ("count", "distinct_count"):
            s = g[distinct_on].nunique(dropna=True).rename(c)
        else:
            s = g[c].agg(a)
        frames.append(s)
    out = pd.concat(frames, axis=1)
    return out

def _annotate_bars(ax: plt.Axes, fmt: str = "{:,.0f}"):
    for p in ax.patches:
        if not hasattr(p, "get_height"):
            continue
        val = p.get_height() if p.get_height() != 0 else p.get_width()
        if np.isnan(val):
            continue
        x = p.get_x() + p.get_width()/2
        y = p.get_y() + p.get_height()
        ax.annotate(fmt.format(val), (x, y), ha="center", va="bottom", fontsize=9, rotation=0)


# ===========================
# Gráficas
# ===========================
def graficar_barras(
    df: pd.DataFrame,
    xlabel: Optional[Union[str, List[str]]] = None,
    y: Optional[Union[str, List[str]]] = None,
    agg: Union[str, Dict[str, str]] = "sum",
    titulo: str = "Gráfica de Barras",
    color: Optional[Union[str, List[str]]] = None,
    *,
    # Compatibilidad completa con plot_from_params
    unique_by: Optional[Union[str, List[str]]] = None,
    conditions_all: Optional[List[List[Any]]] = None,
    conditions_any: Optional[List[Union[List[Any], List[List[Any]]]]] = None,
    distinct_on: Optional[str] = None,
    drop_dupes_before_sum: bool = False,
    sort: Optional[Dict[str, str]] = None,
    limit_categories: Optional[int] = None,
    show_legend: bool = True,
    show_values: bool = True,
    # NUEVO: columna de leyenda y colores por categoría
    legend_col: Optional[str] = None,
    colors_by_category: Optional[Dict[str, str]] = None,
    **kwargs,
) -> Tuple[plt.Figure, plt.Axes]:

    # --------------------------------------------
    # Validación base
    # --------------------------------------------
    if not isinstance(df, pd.DataFrame):
        raise TypeError("El parámetro 'df' debe ser un DataFrame de pandas.")

    # Unificar X
    df2, xlabel_final, _ = _ensure_xlabel(df, xlabel)

    # --------------------------------------------
    # Determinar Y
    # --------------------------------------------
    if y is None:
        num_cols = df2.select_dtypes(include=[np.number]).columns.tolist()
        if not num_cols and not (distinct_on and agg in ("count", "distinct_count")):
            raise ValueError("No se encontraron columnas numéricas ni 'distinct_on'.")
        y_cols = [num_cols[0]] if num_cols else []
    else:
        if isinstance(y, list):
            if not y:
                raise ValueError("Lista 'y' vacía.")
            y = y[0]
        if y not in df2.columns:
            raise ValueError(f"La columna '{y}' no existe en el DataFrame.")
        y_cols = [y]

    value_col = y_cols[0] if y_cols else None

    # --------------------------------------------
    # Filtro + deduplicación previa
    # (ya viene filtrado desde plot_from_params, pero esto es inocuo
    #  porque safe_kwargs manda conditions_* = None)
    # --------------------------------------------
    dff = _prefilter_df(
        df2,
        unique_by=unique_by,
        conditions_all=conditions_all,
        conditions_any=conditions_any,
    )

    # --------------------------------------------
    # Claves de agrupación (X + leyenda opcional)
    # --------------------------------------------
    group_x = xlabel_final

    if legend_col:
        if legend_col not in dff.columns:
            raise ValueError(f"La columna de leyenda '{legend_col}' no existe en el DataFrame.")
        if group_x is None:
            group_x = legend_col
        else:
            if isinstance(group_x, str):
                group_x = [group_x, legend_col]
            else:
                group_x = list(group_x) + [legend_col]

    # --------------------------------------------
    # Agregación
    # --------------------------------------------
    df_plot = _aggregate_frame(
        dff,
        xlabel=group_x,
        y_cols=y_cols,
        agg=agg,
        distinct_on=distinct_on,
        drop_dupes_before_sum=drop_dupes_before_sum,
    )

    if isinstance(df_plot, pd.Series):
        df_plot = df_plot.to_frame(name=value_col or "valor")

    # =====================================================
    # MODO 1: SIN LEYENDA (comportamiento clásico)
    # =====================================================
    if not legend_col:
        if sort:
            by = sort.get("by", "y")
            order = sort.get("order", "desc")
            ascending = (order == "asc")
            if by == "label":
                df_plot = df_plot.sort_index(ascending=ascending)
            else:
                df_plot = df_plot.sort_values(by=df_plot.columns[0], ascending=ascending)

        if limit_categories and limit_categories > 0:
            df_plot = df_plot.head(limit_categories)

        fig, ax = plt.subplots(figsize=(12, 7), facecolor='white')
        ax.set_facecolor('#f8f9fa')

        colname = df_plot.columns[0]

        if color is None or isinstance(color, str):
            base_color = color if isinstance(color, str) else '#5A6C7D'
            bar_colors = base_color
        else:
            bar_colors = color[0] if color else '#5A6C7D'

        bars = ax.bar(
            df_plot.index.astype(str),
            df_plot[colname],
            color=bar_colors,
            edgecolor='white',
            linewidth=1.2,
            alpha=0.9
        )

        for bar in bars:
            bar.set_zorder(3)

        if titulo:
            ax.set_title(titulo, fontsize=16, fontweight='bold', pad=20, color="#000000")
        ax.set_ylabel(colname, fontsize=11, fontweight='600', color='#34495e')
        ax.set_xlabel(xlabel_final if xlabel_final else '', fontsize=11, fontweight='600', color='#34495e')

        ax.grid(axis="y", linestyle="--", alpha=0.3, color='#bdc3c7', zorder=0)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#bdc3c7')
        ax.spines['bottom'].set_color('#bdc3c7')

        ax.tick_params(axis="x", rotation=45, labelsize=9, colors='#34495e')
        ax.tick_params(axis="y", labelsize=9, colors='#34495e')

        if show_values:
            for i, v in enumerate(df_plot[colname]):
                if pd.isna(v):
                    continue
                formatted_val = f"{v:,.0f}" if v >= 1 else f"{v:.2f}"
                ax.text(i, v, formatted_val, ha="center", va="bottom",
                        fontsize=9, fontweight='600', color='#2c3e50')

        if show_legend:
            ax.legend([colname], loc='upper right', framealpha=0.95,
                      edgecolor='#bdc3c7', fontsize=10)
        else:
            leg = ax.get_legend()
            if leg:
                leg.remove()

        fig.tight_layout()
        return fig, ax

    # =====================================================
    # MODO 2: CON LEYENDA (múltiples series agrupadas)
    # =====================================================

    if not isinstance(df_plot.index, pd.MultiIndex):
        raise ValueError("Se esperaba un índice MultiIndex al usar 'legend_col'.")

    val_col = df_plot.columns[0]
    df_wide = df_plot[val_col].unstack(level=-1)  # filas = X, columnas = leyenda

    if sort:
        by = sort.get("by", "y")
        order = sort.get("order", "desc")
        ascending = (order == "asc")
        if by == "label":
            df_wide = df_wide.sort_index(ascending=ascending)
        else:
            row_metric = df_wide.sum(axis=1)
            df_wide = df_wide.loc[row_metric.sort_values(ascending=ascending).index]

    if limit_categories and limit_categories > 0:
        df_wide = df_wide.head(limit_categories)

    fig, ax = plt.subplots(figsize=(12, 7), facecolor='white')
    ax.set_facecolor('#f8f9fa')

    x_labels = df_wide.index.astype(str).tolist()
    legend_values = df_wide.columns.tolist()
    n_series = len(legend_values)
    x = np.arange(len(x_labels))
    width = 0.8 / max(n_series, 1)

    def _color_for_series(name, idx):
        if colors_by_category and name in colors_by_category:
            return colors_by_category[name]
        if isinstance(color, list) and len(color) > idx:
            return color[idx]
        if isinstance(color, str):
            return color
        palette = ["#1A4F80", "#1a9422", "#e0830a", "#eff312",
                   '#8e44ad', '#16a085', '#c0392b', '#7f8c8d']
        return palette[idx % len(palette)]

    for i, serie in enumerate(legend_values):
        vals = df_wide[serie].values
        offset = (i - (n_series - 1) / 2) * width
        c = _color_for_series(str(serie), i)

        bars = ax.bar(
            x + offset,
            vals,
            width=width,
            label=str(serie),
            color=c,
            edgecolor='white',
            linewidth=1.2,
            alpha=0.9
        )

        if show_values:
            for bar in bars:
                v = bar.get_height()
                if pd.isna(v):
                    continue
                formatted_val = f"{v:,.0f}" if v >= 1 else f"{v:.2f}"
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    v,
                    formatted_val,
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    fontweight='600',
                    color='#2c3e50'
                )

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=9, color='#34495e')
    ax.tick_params(axis="y", labelsize=9, colors='#34495e')

    if titulo:
        ax.set_title(titulo, fontsize=16, fontweight='bold', pad=20, color="#000000")
    ax.set_ylabel(val_col, fontsize=11, fontweight='600', color='#34495e')
    ax.set_xlabel(xlabel_final if xlabel_final else '', fontsize=11, fontweight='600', color='#34495e')

    ax.grid(axis="y", linestyle="--", alpha=0.3, color='#bdc3c7', zorder=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#bdc3c7')
    ax.spines['bottom'].set_color('#bdc3c7')

    if show_legend:
        ax.legend(loc='upper right', framealpha=0.95,
                  edgecolor='#bdc3c7', fontsize=10, title=legend_col)
    else:
        leg = ax.get_legend()
        if leg:
            leg.remove()

    fig.tight_layout()
    return fig, ax

    # =====================================================
    # MODO 2: CON LEYENDA (múltiples series agrupadas)
    # =====================================================

    # En este punto, el índice es MultiIndex: [xlabel, legend_col]
    if not isinstance(df_plot.index, pd.MultiIndex):
        raise ValueError("Se esperaba un índice MultiIndex al usar 'legend_col'.")

    # Usamos solo la columna de valores
    val_col = df_plot.columns[0]
    # Pivot: filas = categorías X, columnas = categorías de leyenda
    df_wide = df_plot[val_col].unstack(level=-1)

    # Sort opcional: por suma de filas o por label
    if sort:
        by = sort.get("by", "y")
        order = sort.get("order", "desc")
        ascending = (order == "asc")

        if by == "label":
            df_wide = df_wide.sort_index(ascending=ascending)
        else:
            row_metric = df_wide.sum(axis=1)
            df_wide = df_wide.loc[row_metric.sort_values(ascending=ascending).index]

    # Top-N opcional
    if limit_categories and limit_categories > 0:
        df_wide = df_wide.head(limit_categories)

    # --------------------------------------------
    # Construir figura agrupada por leyenda
    # --------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 7), facecolor='white')
    ax.set_facecolor('#f8f9fa')

    x_labels = df_wide.index.astype(str).tolist()
    legend_values = df_wide.columns.tolist()
    n_series = len(legend_values)
    x = np.arange(len(x_labels))

    # ancho de cada barra
    width = 0.8 / max(n_series, 1)

    # Paleta de colores por serie
    def _color_for_series(name, idx):
        # prioridad: colors_by_category > lista color > un solo color base > paleta default
        if colors_by_category and name in colors_by_category:
            return colors_by_category[name]
        if isinstance(color, list) and len(color) > idx:
            return color[idx]
        if isinstance(color, str):
            return color
        # paleta por defecto
        base_palette = ['#5A6C7D', '#0e4a8f', '#58b12e', '#f39c12',
                        '#8e44ad', '#16a085', '#c0392b', '#7f8c8d']
        return base_palette[idx % len(base_palette)]

    bars_containers = []

    for i, serie in enumerate(legend_values):
        serie_vals = df_wide[serie].values
        offset = (i - (n_series - 1) / 2) * width
        c = _color_for_series(str(serie), i)

        bars = ax.bar(
            x + offset,
            serie_vals,
            width=width,
            label=str(serie),
            color=c,
            edgecolor='white',
            linewidth=1.2,
            alpha=0.9
        )
        bars_containers.append(bars)

        # Valores sobre las barras de cada serie (opcional)
        if show_values:
            for bar in bars:
                v = bar.get_height()
                if pd.isna(v):
                    continue
                formatted_val = f"{v:,.0f}" if v >= 1 else f"{v:.2f}"
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    v,
                    formatted_val,
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    fontweight='600',
                    color='#2c3e50'
                )

    # Ejes y estilo
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=9, color='#34495e')
    ax.tick_params(axis="y", labelsize=9, colors='#34495e')

    if titulo:
        ax.set_title(titulo, fontsize=16, fontweight='bold', pad=20, color="#000000")
    ax.set_ylabel(val_col, fontsize=11, fontweight='600', color='#34495e')
    ax.set_xlabel(xlabel_final if xlabel_final else '', fontsize=11, fontweight='600', color='#34495e')

    ax.grid(axis="y", linestyle="--", alpha=0.3, color='#bdc3c7', zorder=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#bdc3c7')
    ax.spines['bottom'].set_color('#bdc3c7')

    # Leyenda
    if show_legend:
        ax.legend(loc='upper right', framealpha=0.95, edgecolor='#bdc3c7', fontsize=10, title=legend_col)
    else:
        leg = ax.get_legend()
        if leg:
            leg.remove()

    fig.tight_layout()
    return fig, ax


def graficar_torta(
    df: pd.DataFrame,
    xlabel: Optional[Union[str, List[str]]] = None,
    y: Optional[str] = None,
    agg: Union[str, Dict[str, str]] = "sum",
    titulo: str = "Gráfico de Torta",
    color: Optional[Union[str, List[str]]] = None,
    *,
    unique_by: Optional[Union[str, List[str]]] = None,
    conditions_all: Optional[List[List[Any]]] = None,
    conditions_any: Optional[List[Union[List[Any], List[List[Any]]]]] = None,
    distinct_on: Optional[str] = None,
    drop_dupes_before_sum: bool = False,
    sort: Optional[Dict[str, str]] = None,
    limit_categories: Optional[int] = None,
) -> Tuple[plt.Figure, plt.Axes]:

    if not isinstance(df, pd.DataFrame):
        raise TypeError("El parámetro 'df' debe ser un DataFrame de pandas.")

    # Aceptar múltiples columnas en X → combinar
    df2, xlabel_final, _ = _ensure_xlabel(df, xlabel)

    # Determinar y
    if y is None:
        num_cols = df2.select_dtypes(include=[np.number]).columns.tolist()
        if not num_cols and not (distinct_on and agg in ("count", "distinct_count")):
            raise ValueError("No se encontraron columnas numéricas ni 'distinct_on' para contar.")
        y_cols = [num_cols[0]] if num_cols else []
    else:
        if y not in df2.columns:
            raise ValueError(f"La columna '{y}' no existe en el DataFrame.")
        y_cols = [y]

    # Filtro + deduplicación previa
    dff = _prefilter_df(
        df2,
        unique_by=unique_by,
        conditions_all=conditions_all,
        conditions_any=conditions_any,
    )

    # Agregar
    df_plot = _aggregate_frame(
        dff,
        xlabel=xlabel_final,
        y_cols=y_cols,
        agg=agg,
        distinct_on=distinct_on,
        drop_dupes_before_sum=drop_dupes_before_sum,
    )

    # --- Normalizar a DataFrame ---
    if isinstance(df_plot, pd.Series):
        df_plot = df_plot.to_frame(name="valor")

    if df_plot.shape[1] != 1:
        raise ValueError("El gráfico de torta requiere una única serie numérica después de agregar.")

    df_plot = df_plot.reset_index()

    col_val = df_plot.columns[-1]
    col_lab = df_plot.columns[0]

    serie = df_plot[col_val].replace([np.inf, -np.inf], np.nan).dropna()
    etiquetas = df_plot.loc[serie.index, col_lab].astype(str)

    if serie.empty:
        raise ValueError("No hay datos válidos para graficar (todo es NaN/inf o vacío).")

    # sort / top-N
    if sort:
        by = sort.get("by", "y")
        order = sort.get("order", "desc")
        ascending = (order == "asc")
        if by == "label":
            orden_idx = etiquetas.sort_values(ascending=ascending).index
        else:
            orden_idx = serie.sort_values(ascending=ascending).index
        serie = serie.loc[orden_idx]
        etiquetas = etiquetas.loc[orden_idx]

    if limit_categories and limit_categories > 0:
        serie = serie.head(limit_categories)
        etiquetas = etiquetas.head(limit_categories)

    total = serie.clip(lower=0).sum()
    if total <= 0:
        raise ValueError("La suma de valores es 0; no es posible construir la torta.")

    # --- COLORES ---
    if color is None:
        # Paleta multicolor por defecto
        cmap = plt.get_cmap("tab20")
        color = [cmap(i % cmap.N) for i in range(len(serie))]
    elif isinstance(color, str):
        # Un solo color
        color = [color] * len(serie)
    elif isinstance(color, (list, tuple, np.ndarray)):
        # Si pasan menos colores que categorías, repetimos la paleta
        color = list(color)
        if len(color) < len(serie):
            reps = int(np.ceil(len(serie) / len(color)))
            color = (color * reps)[:len(serie)]

    # Plot
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.pie(
        serie.values,
        labels=etiquetas.values,
        autopct='%1.1f%%',
        startangle=90,
        colors=color,
        shadow=False,
    )
    ax.set_title(titulo)
    ax.axis('equal')
    fig.tight_layout()
    return fig, ax

def graficar_barras_horizontal(
    df: pd.DataFrame,
    xlabel: Optional[Union[str, List[str]]] = None,
    y: Optional[Union[str, List[str]]] = None,
    agg: Union[str, Dict[str, str]] = "sum",
    titulo: str = "Gráfica de Barras Horizontal",
    color: Optional[Union[str, List[str]]] = None,
    *,
    # Controles adicionales (compatibilidad total)
    unique_by: Optional[Union[str, List[str]]] = None,
    conditions_all: Optional[List[List[Any]]] = None,
    conditions_any: Optional[List[Union[List[Any], List[List[Any]]]]] = None,
    distinct_on: Optional[str] = None,
    drop_dupes_before_sum: bool = False,
    sort: Optional[Dict[str, str]] = None,          # {"by": "y"|"label", "order": "asc"|"desc"}
    limit_categories: Optional[int] = None,
    show_legend: bool = True,
    show_values: bool = True,
    # 👇 NUEVO
    legend_col: Optional[str] = None,
    colors_by_category: Optional[Dict[str, str]] = None,
    **kwargs,   # absorbe parámetros extra
) -> Tuple[plt.Figure, plt.Axes]:

    # Validación base
    if not isinstance(df, pd.DataFrame):
        raise TypeError("El parámetro 'df' debe ser un DataFrame de pandas.")

    # Combinar X
    df2, xlabel_final, _ = _ensure_xlabel(df, xlabel)

    # Determinar columnas numéricas (una sola métrica)
    if y is None:
        num_cols = df2.select_dtypes(include=[np.number]).columns.tolist()
        if not num_cols and not (distinct_on and agg in ("count", "distinct_count")):
            raise ValueError("No se encontraron columnas numéricas ni 'distinct_on'.")
        y_cols = [num_cols[0]] if num_cols else []
    else:
        if isinstance(y, list):
            if not y:
                raise ValueError("Lista 'y' vacía.")
            y = y[0]
        if y not in df2.columns:
            raise ValueError(f"La columna '{y}' no existe en el DataFrame.")
        y_cols = [y]

    value_col = y_cols[0] if y_cols else None

    # Filtro + deduplicación previa
    dff = _prefilter_df(
        df2,
        unique_by=unique_by,
        conditions_all=conditions_all,
        conditions_any=conditions_any,
    )

    # Preparar claves de agrupación (X + leyenda opcional)
    group_x = xlabel_final

    if legend_col:
        if legend_col not in dff.columns:
            raise ValueError(f"La columna de leyenda '{legend_col}' no existe en el DataFrame.")

        if group_x is None:
            group_x = legend_col
        else:
            if isinstance(group_x, str):
                group_x = [group_x, legend_col]
            else:
                group_x = list(group_x) + [legend_col]

    # Agregación
    df_plot = _aggregate_frame(
        dff,
        xlabel=group_x,
        y_cols=y_cols,
        agg=agg,
        distinct_on=distinct_on,
        drop_dupes_before_sum=drop_dupes_before_sum,
    )

    # Asegurar objeto gráfico
    if isinstance(df_plot, pd.Series):
        df_plot = df_plot.to_frame(name=value_col or "valor")

    # =====================================================
    # MODO 1: SIN LEYENDA (comportamiento actual)
    # =====================================================
    if not legend_col:
        # Ordenamiento opcional
        if sort:
            by = sort.get("by", "y")
            order = sort.get("order", "desc")
            ascending = (order == "asc")

            if by == "label":
                df_plot = df_plot.sort_index(ascending=ascending)
            else:
                df_plot = df_plot.sort_values(by=df_plot.columns[0], ascending=ascending)

        # Top-N opcional
        if limit_categories and limit_categories > 0:
            df_plot = df_plot.head(limit_categories)

        # Crear figura con estilo profesional
        fig, ax = plt.subplots(figsize=(12, max(6, len(df_plot) * 0.4)), facecolor='white')
        ax.set_facecolor('#f8f9fa')

        colname = df_plot.columns[0]
        
        # Color profesional si no se especifica
        if color is None or isinstance(color, str):
            base_color = color if isinstance(color, str) else '#607D8B'
            bar_color = base_color
        else:
            bar_color = color[0] if color else '#607D8B'
        
        bars = ax.barh(
            df_plot.index.astype(str),
            df_plot[colname], 
            color=bar_color,
            edgecolor='white',
            linewidth=1.2,
            alpha=0.9
        )
        
        for bar in bars:
            bar.set_zorder(3)

        # Títulos y etiquetas mejoradas
        if titulo:
            ax.set_title(titulo, fontsize=16, fontweight='bold', pad=20, color='#2c3e50')
        ax.set_xlabel(colname, fontsize=11, fontweight='600', color='#34495e')
        ax.set_ylabel(xlabel_final if xlabel_final else '', fontsize=11, fontweight='600', color='#34495e')
        
        # Grid mejorado
        ax.grid(axis="x", linestyle="--", alpha=0.3, color='#bdc3c7', zorder=0)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#bdc3c7')
        ax.spines['bottom'].set_color('#bdc3c7')
        
        # Tick styling
        ax.tick_params(axis="both", labelsize=9, colors='#34495e')

        # Mostrar valores con formato mejorado
        if show_values:
            for bar in bars:
                v = bar.get_width()
                if pd.isna(v):
                    continue
                formatted_val = f"{v:,.0f}" if v >= 1 else f"{v:.2f}"
                ax.text(
                    v,
                    bar.get_y() + bar.get_height() / 2,
                    formatted_val,
                    va='center',
                    ha='left',
                    fontsize=9,
                    fontweight='600',
                    color='#2c3e50'
                )

        # Leyenda mejorada
        if show_legend:
            ax.legend([colname], loc='lower right', framealpha=0.95, 
                      edgecolor='#bdc3c7', fontsize=10)
        else:
            leg = ax.get_legend()
            if leg:
                leg.remove()

        fig.tight_layout()
        return fig, ax

    # =====================================================
    # MODO 2: CON LEYENDA (múltiples series agrupadas)
    # =====================================================

    # Esperamos índice MultiIndex: [xlabel, legend_col]
    if not isinstance(df_plot.index, pd.MultiIndex):
        raise ValueError("Se esperaba un índice MultiIndex al usar 'legend_col'.")

    val_col = df_plot.columns[0]
    df_wide = df_plot[val_col].unstack(level=-1)  # filas = categorías, columnas = leyenda

    # Ordenamiento opcional
    if sort:
        by = sort.get("by", "y")
        order = sort.get("order", "desc")
        ascending = (order == "asc")

        if by == "label":
            df_wide = df_wide.sort_index(ascending=ascending)
        else:
            row_metric = df_wide.sum(axis=1)
            df_wide = df_wide.loc[row_metric.sort_values(ascending=ascending).index]

    # Top-N opcional
    if limit_categories and limit_categories > 0:
        df_wide = df_wide.head(limit_categories)

    # Construir figura agrupada
    n_cats = len(df_wide.index)
    fig, ax = plt.subplots(figsize=(12, max(6, n_cats * 0.4)), facecolor='white')
    ax.set_facecolor('#f8f9fa')

    y_labels = df_wide.index.astype(str).tolist()
    legend_values = df_wide.columns.tolist()
    n_series = len(legend_values)
    y_pos = np.arange(len(y_labels))

    # altura de cada barra
    height = 0.8 / max(n_series, 1)

    # Paleta de colores por serie
    def _color_for_series(name, idx):
        if colors_by_category and name in colors_by_category:
            return colors_by_category[name]
        if isinstance(color, list) and len(color) > idx:
            return color[idx]
        if isinstance(color, str):
            return color
        base_palette = ['#607D8B', '#0e4a8f', '#58b12e', '#f39c12',
                        '#8e44ad', '#16a085', '#c0392b', '#7f8c8d']
        return base_palette[idx % len(base_palette)]

    for i, serie in enumerate(legend_values):
        serie_vals = df_wide[serie].values
        offset = (i - (n_series - 1) / 2) * height
        c = _color_for_series(str(serie), i)

        bars = ax.barh(
            y_pos + offset,
            serie_vals,
            height=height,
            label=str(serie),
            color=c,
            edgecolor='white',
            linewidth=1.2,
            alpha=0.9
        )

        if show_values:
            for bar in bars:
                v = bar.get_width()
                if pd.isna(v):
                    continue
                formatted_val = f"{v:,.0f}" if v >= 1 else f"{v:.2f}"
                ax.text(
                    v,
                    bar.get_y() + bar.get_height() / 2,
                    formatted_val,
                    va='center',
                    ha='left',
                    fontsize=9,
                    fontweight='600',
                    color='#2c3e50'
                )

    # Etiquetas de eje
    ax.set_yticks(y_pos)
    ax.set_yticklabels(y_labels, fontsize=9, color='#34495e')
    ax.tick_params(axis="x", labelsize=9, colors='#34495e')

    if titulo:
        ax.set_title(titulo, fontsize=16, fontweight='bold', pad=20, color='#2c3e50')
    ax.set_xlabel(val_col, fontsize=11, fontweight='600', color='#34495e')
    ax.set_ylabel(xlabel_final if xlabel_final else '', fontsize=11, fontweight='600', color='#34495e')

    ax.grid(axis="x", linestyle="--", alpha=0.3, color='#bdc3c7', zorder=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#bdc3c7')
    ax.spines['bottom'].set_color('#bdc3c7')

    # Leyenda
    if show_legend:
        ax.legend(loc='lower right', framealpha=0.95,
                  edgecolor='#bdc3c7', fontsize=10, title=legend_col)
    else:
        leg = ax.get_legend()
        if leg:
            leg.remove()

    fig.tight_layout()
    return fig, ax

def graficar_tabla(
    df: pd.DataFrame,
    xlabel: Optional[Union[str, List[str]]] = None,
    y: Optional[Union[str, List[str]]] = None,
    agg: Union[str, Dict[str, str]] = "sum",
    titulo: str = "Tabla de Datos",
    color: Optional[str] = None,
    *,
    # Filtros / deduplicación
    unique_by: Optional[Union[str, List[str]]] = None,
    conditions_all: Optional[List[List[Any]]] = None,
    conditions_any: Optional[List[Union[List[Any], List[List[Any]]]]] = None,
    # Agregación extendida
    distinct_on: Optional[str] = None,
    drop_dupes_before_sum: bool = False,
    where: Optional[pd.Series] = None,
    # Columna de porcentaje
    percentage_of: Optional[str] = None,
    percentage_colname: str = "porcentaje",
    # 👇 NUEVO: usar legend_col también en tablas
    legend_col: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:

    """
    Tabla con:
      - xlabel como str o list[str] (se combina con _ensure_xlabel)
      - filtros AND/OR, unique_by, distinct_on y agregaciones extendidas
      - opcionalmente, legend_col para pivotear columnas por categoría
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("El parámetro 'df' debe ser un DataFrame de pandas.")

    # Aceptar multi-X → combinar etiquetas
    df2, xlabel_final, _ = _ensure_xlabel(df, xlabel)

    # y_cols
    if y is None:
        y_cols = df2.select_dtypes(include=[np.number]).columns.tolist()
        # permitir tablas con conteo único vía distinct_on aun si no hay numéricas
        if not y_cols and not (distinct_on and isinstance(agg, str) and agg in ("count", "distinct_count")):
            raise ValueError("No se encontraron columnas numéricas para la tabla.")
    elif isinstance(y, str):
        if y not in df2.columns:
            raise ValueError(f"La columna '{y}' no existe en el DataFrame.")
        y_cols = [y]
    else:
        missing = [col for col in y if col not in df2.columns]
        if missing:
            raise ValueError(f"Las siguientes columnas no existen en el DataFrame: {missing}")
        y_cols = y

    # Filtro y deduplicación previa
    dff = _prefilter_df(
        df2,
        unique_by=unique_by,
        conditions_all=conditions_all,
        conditions_any=conditions_any
    )

    # --- Preparar claves de agrupación (X + legend_col opcional) ---
    group_x = xlabel_final
    if legend_col:
        if legend_col not in dff.columns:
            raise ValueError(f"La columna de leyenda '{legend_col}' no existe en el DataFrame.")
        if group_x is None:
            group_x = legend_col
        else:
            if isinstance(group_x, str):
                group_x = [group_x, legend_col]
            else:
                group_x = list(group_x) + [legend_col]

    # --- Agregar (respeta distinct_count / sum_distinct / distinct_on) ---
    df_plot = _aggregate_frame(
        dff,
        xlabel=group_x,
        y_cols=y_cols,
        agg=agg,
        distinct_on=distinct_on,
        drop_dupes_before_sum=drop_dupes_before_sum,
        where=where
    )

    # Si es Series → DataFrame
    if isinstance(df_plot, pd.Series):
        df_plot = df_plot.to_frame(name=y_cols[0] if y_cols else "valor")

    # --- Pivot por legend_col → columnas por categoría ---
    if legend_col and isinstance(df_plot.index, pd.MultiIndex) and len(df_plot.columns) == 1:
        # Por ahora solo soportamos 1 métrica para el pivot
        metric_col = df_plot.columns[0]
        # columnas = categorías de legend_col
        df_wide = df_plot[metric_col].unstack(level=-1)
        # index (xlabel) pasa a columnas normales
        df_plot = df_wide.reset_index()
        # IMPORTANTE: en este modo no usamos percentage_of (no es obvio a qué columna aplica)
        if percentage_of:
            raise ValueError(
                "Por ahora no se soporta 'percentage_of' cuando se usa 'legend_col' en tablas. "
                "Quita percentage_of del JSON o no uses legend_col."
            )
    else:
        # Reset index normal
        df_plot = df_plot.reset_index() if group_x is not None else df_plot.reset_index(drop=True)

    # --- Columna de porcentaje opcional (solo cuando NO hay pivot por legend_col) ---
    if percentage_of and not legend_col:
        if percentage_of not in df_plot.columns:
            raise ValueError(f"percentage_of='{percentage_of}' no existe en la tabla resultante.")
        col_num = pd.to_numeric(df_plot[percentage_of], errors="coerce")
        total = col_num.sum()
        if total and not np.isnan(total):
            df_plot[percentage_colname] = np.where(
                col_num.notna(),
                col_num / total,   # proporción (0.45 → 45%)
                np.nan,
            )
        else:
            df_plot[percentage_colname] = np.nan

    # ---------- Formateo base ----------
    df_display = df_plot.copy()
    for col in df_display.columns:
        if is_numeric_dtype(df_display[col]):
            if col != percentage_colname:
                df_display[col] = df_display[col].round(2)
        elif is_datetime64_any_dtype(df_display[col]):
            df_display[col] = df_display[col].dt.strftime("%Y-%m-%d")
    df_display = df_display.fillna("").astype(str)

    # Formatear números con separador de miles y porcentajes
    df_formatted = df_display.copy()
    for col in df_display.columns:
        try:
            numeric_col = pd.to_numeric(df_display[col], errors='coerce')
            if numeric_col.notna().any():
                if col == percentage_colname:
                    df_formatted[col] = numeric_col.apply(
                        lambda x: f"{x:.0%}" if pd.notna(x) else ""
                    )
                else:
                    df_formatted[col] = numeric_col.apply(
                        lambda x: f"{x:,.0f}" if pd.notna(x) and x >= 1 else 
                                 (f"{x:.2f}" if pd.notna(x) else "")
                    )
        except Exception:
            pass

    # Figura con diseño profesional y más grande
    n_cols = len(df_formatted.columns)
    n_rows = len(df_formatted)
    fig_w = max(18, n_cols * 1)
    fig_h = max(9, n_rows * 1)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), facecolor='white')
    ax.axis("off")

    tabla = ax.table(
        cellText=df_formatted.values.tolist(),
        colLabels=df_formatted.columns.tolist(),
        cellLoc="center",
        loc="center",
        colWidths=[0.28] * n_cols
    )
    tabla.auto_set_font_size(False)
    tabla.set_fontsize(28)
    tabla.scale(2.2, 4.2)

    for col_idx in range(n_cols):
        tabla.auto_set_column_width(col=col_idx)



# Figura con diseño profesional y más grande
    n_cols = len(df_formatted.columns)
    n_rows = len(df_formatted)
    fig_w = max(18, n_cols * 1)
    fig_h = max(9, n_rows * 1)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), facecolor='white')
    ax.axis("off")

    tabla = ax.table(
        cellText=df_formatted.values.tolist(),
        colLabels=df_formatted.columns.tolist(),
        cellLoc="center",
        loc="center",
        colWidths=[0.28] * n_cols
    )
    tabla.auto_set_font_size(False)
    tabla.set_fontsize(28)
    tabla.scale(2.2, 4.2)

    for col_idx in range(n_cols):
        tabla.auto_set_column_width(col=col_idx)

    # ===== Estilos de colores: encabezado azul, filas blancas (o color custom) =====
    # Azul corporativo para encabezado
    header_bg = "#0e4a8f"

    # Fondo de filas de datos:
    # - si se pasa 'color' → se usa para las celdas de datos
    # - si no se pasa → filas blancas
    data_bg = color if color else "#ffffff"

    n_rows = len(df_formatted)
    for (row, col_idx), cell in tabla.get_celld().items():
        if row == 0:
            # Encabezado
            text_color = "#212121" if header_bg.lower() in ("#fff", "#ffffff", "white") else "white"
            cell.set_text_props(weight="bold", color=text_color, fontsize=36)
            cell.set_facecolor(header_bg)
        else:
            # Filas de datos
            valor = cell.get_text().get_text()
            try:
                float(valor.replace(',', ''))
                cell.set_text_props(color="#212121", fontsize=32, weight="bold")
            except Exception:
                cell.set_text_props(color="#2c3e50", fontsize=28, weight="normal")

            cell.set_facecolor(data_bg)

        cell.set_edgecolor("#FFFFFF")
        cell.set_linewidth(1.5)
        cell.PAD = 0.42
 
    # Título (recuerda que en plot_from_params ya lo estamos vaciando)
    if titulo:
        ax.set_title(titulo, fontsize=38, fontweight="bold", pad=60, color="#000000")

    fig.tight_layout(pad=6.5)
    return fig, ax

# ===========================
# Router desde params
# ===========================
def _as_float_or_inf(x):
    if isinstance(x, str):
        s = x.strip().lower()
        if s in ("-inf", "-infty", "-infinite"):
            return -np.inf
        if s in ("+inf", "inf", "infty", "infinite"):
            return  np.inf
    return float(x)

def _apply_binning(df: pd.DataFrame, binning: dict) -> tuple[pd.DataFrame, str]:
    """
    Crea una columna de agrupación por rangos con pd.cut y la devuelve junto con su nombre.
    binning = { "column": "edad", "bins": ["-inf",5,11,18,26,59,"+inf"], "labels": [...], "output_col": "grupo_edad" }
    """
    if not binning:
        return df, None

    src_col   = binning.get("column")
    if src_col not in df.columns:
        raise ValueError(f"Binning: la columna fuente '{src_col}' no existe en el DataFrame.")

    # nombre de salida
    out_col   = binning.get("output_col") or f"{src_col}_bucket"

    # normaliza bordes
    raw_bins  = binning.get("bins")
    if not raw_bins or len(raw_bins) < 2:
        raise ValueError("Binning: 'bins' debe tener al menos 2 límites.")
    bins = [_as_float_or_inf(b) for b in raw_bins]

    labels = binning.get("labels")
    if labels is not None and len(labels) != (len(bins) - 1):
        raise ValueError("Binning: la cantidad de 'labels' debe ser igual a len(bins)-1.")

    # coerciona a numérico la columna fuente
    s = pd.to_numeric(df[src_col], errors="coerce")

    # crea el bucket
    df2 = df.copy()
    df2[out_col] = pd.cut(
        s, bins=bins, labels=labels, include_lowest=True, right=True
    )
    # pd.cut puede dar Categorical; conviértelo a string para labels uniformes
    df2[out_col] = df2[out_col].astype(str)

    return df2, out_col

def _stack_columns(
    df: pd.DataFrame,
    columns: List[str],
    *,
    output_col: str = "tipo_riesgo",
    value_col: str = "valor",
    label_map: Optional[Dict[str, str]] = None,
    id_cols: Optional[List[str]] = None,     # <<< opcional: claves a conservar explícitamente
    dropna: bool = True                      # <<< opcional: limpiar filas sin valor
) -> pd.DataFrame:
    df2 = df

    # 1) Si 'documento' (u otras claves) está en el índice, súbela a columnas
    index_names = list(df2.index.names) if df2.index.names is not None else []
    must_reset = False
    keys_to_preserve = set(id_cols or [])
    for k in ["documento"]:                     # <<< agrega aquí otras claves si aplica
        if k in index_names:
            must_reset = True
            keys_to_preserve.add(k)

    if must_reset:
        df2 = df2.reset_index()

    # 2) Construir id_vars de forma explícita para garantizar que sobrevivan
    if id_cols is not None:
        id_vars = list(dict.fromkeys(id_cols))   # sin duplicados, respetando orden
    else:
        # por defecto: todas las columnas menos las que apilamos
        id_vars = [c for c in df2.columns if c not in columns]

    # Asegurar que 'documento' esté en id_vars si existe como columna
    for k in ["documento"]:
        if k in df2.columns and k not in id_vars:
            id_vars.append(k)

    # 3) Melt preservando id_vars
    m = df2.melt(id_vars=id_vars, value_vars=columns, var_name=output_col, value_name=value_col)

    # 4) Map de etiquetas (si se pide)
    if label_map:
        m[output_col] = m[output_col].map(lambda x: label_map.get(x, x))

    # 5) Opcional: eliminar filas sin valor
    if dropna:
        m = m.dropna(subset=[value_col])

    return m



from textwrap import fill, shorten
import matplotlib.pyplot as plt

# ======================
# Helpers de formateo
# ======================

def _wrap_shorten(text: str, *, max_chars: int | None, wrap_width: int) -> str:
    if text is None:
        return ""
    s = str(text)
    if max_chars and len(s) > max_chars:
        s = shorten(s, width=max_chars, placeholder="…")
    return fill(s, width=wrap_width)

def _wrap_shorten_ticks(ax, *, axis="x", wrap_width=22, max_chars=90, fontsize=9, rotation=None):
    """Abrevia (ellipsis) y envuelve (\n) las etiquetas del eje indicado, sin cambiar orientación."""
    if axis == "x":
        ticks = ax.get_xticks()
        labels = [lab.get_text() for lab in ax.get_xticklabels()]
        new_labels = [_wrap_shorten(lbl, max_chars=max_chars, wrap_width=wrap_width) for lbl in labels]
        ax.set_xticks(ticks, new_labels)
        ax.tick_params(axis="x", labelsize=fontsize)
        if rotation is not None:
            for lab in ax.get_xticklabels():
                lab.set_rotation(rotation)
                lab.set_ha("right")
    else:
        ticks = ax.get_yticks()
        labels = [lab.get_text() for lab in ax.get_yticklabels()]
        new_labels = [_wrap_shorten(lbl, max_chars=max_chars, wrap_width=wrap_width) for lbl in labels]
        ax.set_yticks(ticks, new_labels)
        ax.tick_params(axis="y", labelsize=fontsize)

def _format_pie_labels(labels, *, max_chars=60, wrap_width=25):
    """Devuelve las etiquetas formateadas para tortas (abrevia + envuelve)."""
    return [_wrap_shorten(str(lbl), max_chars=max_chars, wrap_width=wrap_width) for lbl in labels]

def _apply_pie_texts(ax, formatted_labels):
    """
    Reemplaza textos de labels en una torta ya dibujada.
    - Intenta sobre los Text de ax (labels) evitando los que son % (autopct).
    - También actualiza la leyenda si existe.
    """
    # 1) Textos alrededor de la torta
    i = 0
    for txt in ax.texts:
        s = txt.get_text()
        # Heurística simple: si contiene '%', lo consideramos autopct y no lo tocamos
        if "%" in s:
            continue
        if i < len(formatted_labels):
            txt.set_text(formatted_labels[i])
            i += 1

    # 2) Leyenda (si existe)
    leg = ax.get_legend()
    if leg is not None:
        leg_texts = leg.get_texts()
        for j, t in enumerate(leg_texts):
            if j < len(formatted_labels):
                t.set_text(formatted_labels[j])

def _finalize_layout(fig, ax, *, legend_outside=True):
    """Ajuste seguro de layout sin cambiar el estilo del gráfico."""
    if legend_outside and ax.get_legend() is not None:
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.)
    plt.tight_layout()


@register("plt_from_params")
def plot_from_params(df, params, *, show: bool=False):
    import copy
    import matplotlib.pyplot as plt

    p = copy.deepcopy(params)

    # --- Normalización del selector ---
    fn_raw = p.get("function_name") or p.get("chart_type")
    if fn_raw is None:
        raise ValueError("Falta 'function_name' o 'chart_type' en params.")

    fn_key = str(fn_raw).strip().lower()

    DISPATCH = {
        "graficar_barras": graficar_barras,
        "graficar_barras_horizontal": graficar_barras_horizontal,
        "graficar_torta": graficar_torta,
        "graficar_tabla": graficar_tabla,

        "barras": graficar_barras,
        "bar": graficar_barras,
        "barras_horizontal": graficar_barras_horizontal,
        "horizontal": graficar_barras_horizontal,

        "torta": graficar_torta,
        "pie": graficar_torta,

        "tabla": graficar_tabla,
        "table": graficar_tabla,
    }

    func = DISPATCH.get(fn_key)
    if func is None:
        raise ValueError(f"Función de gráfico no reconocida: {fn_raw!r} (normalizado: {fn_key!r})")

    # --- Parámetros comunes ---
    xlabel  = p.get("xlabel")
    y       = p.get("y")
    agg     = p.get("agg", "sum")
    titulo  = ""
    #titulo  = p.get("title", "Gráfico")
    color   = p.get("color")
    tipo    = p.get("tipo")

    # --- Formateo general ---
    tick_fontsize = int(p.get("tick_fontsize", 9))
    rotation      = p.get("rotation", 35)
    wrap_width_x  = int(p.get("wrap_width_x", 20))
    wrap_width_y  = int(p.get("wrap_width_y", 30))
    max_chars_x   = int(p.get("max_chars_x", 85))
    max_chars_y   = int(p.get("max_chars_y", 120))
    legend_out    = bool(p.get("legend_outside", True))

    # --- Formateo torta ---
    pie_max_chars = int(p.get("pie_max_chars", 60))
    pie_wrap_w    = int(p.get("pie_wrap_width", 25))

    # --- Limpieza de condiciones ---
    def _clean_conditions(cond):
        if not cond:
            return []
        cleaned = []
        for c in cond:
            if not c:
                continue
            if not isinstance(c, (list, tuple)):
                continue
            if len(c) != 3:
                continue
            col, op, val = c
            if col is None or col == "" or str(col).strip() == "":
                continue
            cleaned.append([col, op, val])
        return cleaned

    # --- Extra: parámetros faltantes ---
    unique_by             = p.get("unique_by")
    conditions_all        = _clean_conditions(p.get("conditions_all"))
    conditions_any        = _clean_conditions(p.get("conditions_any"))
    distinct_on           = p.get("distinct_on")
    drop_dupes_before_sum = p.get("drop_dupes_before_sum", False)

    # --- BINNING ---
    binning = p.get("binning")
    if binning:
        df, bucket_col = _apply_binning(df, binning)
        xlabel = bucket_col

    # --- PRE-FILTRADO ANTES DEL STACK ---
    df_filtered = _prefilter_df(
        df,
        unique_by=unique_by,
        conditions_all=conditions_all,
        conditions_any=conditions_any,
    )

    # --- STACK ---
    stack = p.get("stack_columns")
    if stack:
        cols      = stack["columns"]
        out_col   = stack.get("output_col", "tipo_riesgo")
        val_col   = stack.get("value_col", "valor")
        label_map = stack.get("label_map")
        keep_val  = stack.get("keep_value")

        df_filtered = _stack_columns(df_filtered, cols, output_col=out_col, value_col=val_col, label_map=label_map)

        if keep_val not in (None, "any", "", []):
            df_filtered = df_filtered[df_filtered[val_col] == keep_val]

        xlabel = out_col


    # --- Preparar kwargs comunes ---
    common_kwargs = dict(
        xlabel=xlabel,
        y=y,
        agg=agg,
        titulo=titulo,
        color=color,
        unique_by=unique_by,
        conditions_all=conditions_all,
        conditions_any=conditions_any,
        distinct_on=distinct_on,
        drop_dupes_before_sum=drop_dupes_before_sum,
    )
        # Config específica de tablas: columna de porcentaje
    if func is graficar_tabla:
        percentage_of = p.get("percentage_of")
        percentage_colname = p.get("percentage_colname", "porcentaje")
        common_kwargs["percentage_of"] = percentage_of
        common_kwargs["percentage_colname"] = percentage_colname

    # legend_col también usable en tablas (para pivotear columnas)
    if func is graficar_tabla:
        legend_col = p.get("legend_col")
        if legend_col is not None:
            common_kwargs["legend_col"] = legend_col

            
    # --- Normalización mínima de torta ---
    if fn_key in ("graficar_torta", "torta", "pie") and isinstance(y, list):
        y = y[0] if y else None
        common_kwargs["y"] = y
    
    # --- Config específica de barras: leyendas / colores por categoría ---
    if func in (graficar_barras, graficar_barras_horizontal):
        legend_col = p.get("legend_col")
        colors_by_category = p.get("colors_by_category")

        if legend_col is not None:
            common_kwargs["legend_col"] = legend_col
        if colors_by_category is not None:
            common_kwargs["colors_by_category"] = colors_by_category

    # === Llamado central a la función sin repetir filtros internos ===
    safe_kwargs = common_kwargs.copy()
    safe_kwargs["conditions_all"] = None
    safe_kwargs["conditions_any"] = None
    safe_kwargs["unique_by"] = None

    fig, ax = func(df_filtered, **safe_kwargs)


    # --- Post-formateo según el tipo ---
    t = (tipo or "").lower()

    if not t:
        ct = (p.get("chart_type") or "").lower()
        if ct in ("barras", "bar", "column", "col"):
            t = "barras"
        elif ct in ("horizontal", "barh", "barras_horizontal", "graficar_barras_horizontal"):
            t = "barh"
        elif ct in ("torta", "pie", "pastel", "graficar_torta"):
            t = "torta"
        elif ct in ("tabla", "table", "graficar_tabla"):
            t = "tabla"

    if t in ("barras", "bar", "column", "col"):
        _wrap_shorten_ticks(ax, axis="x", wrap_width=wrap_width_x, max_chars=max_chars_x, fontsize=tick_fontsize, rotation=rotation)

    elif t in ("barh", "barras_h", "horizontal", "barra_horizontal"):
        _wrap_shorten_ticks(ax, axis="y", wrap_width=wrap_width_y, max_chars=max_chars_y, fontsize=tick_fontsize)

    elif t in ("torta", "pie", "pastel"):
        # Tomar SOLO los textos que no son porcentajes (autopct)
        labels_raw = [txt.get_text() for txt in ax.texts if "%" not in txt.get_text()]

    # Aplicar abreviación / wrap
        labels_fmt = _format_pie_labels(
            labels_raw,
            max_chars=pie_max_chars,
            wrap_width=pie_wrap_w
        )
        _apply_pie_texts(ax, labels_fmt)


    _finalize_layout(fig, ax, legend_outside=legend_out)

    if show:
        plt.tight_layout()
        plt.show()

    return fig, ax

