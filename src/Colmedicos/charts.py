#from curses import raw
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



def _normalize(text: str) -> str:
    if text is None:
        return ""
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower()


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

    # 🔥 NUEVOS OPERADORES PARA TEXTO
    "contains": lambda a, b: a.astype(str).str.contains(b, case=False, na=False),
    "startswith": lambda a, b: (a.astype(str).str.lower().str.startswith(str(b).lower(), na=False)),
    "endswith": lambda a, b: a.astype(str).str.endswith(b, na=False),

    # LIKE estilo SQL
    "like": lambda a, b: a.astype(str).str.contains(b.replace("%", ""), case=False, na=False),

    # Versión insensible a tildes
    "icontains": lambda a, b: a.astype(str).apply(lambda x: _normalize(x)).str.contains(_normalize(b), case=False, na=False),
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
def _aggregate_frame_chart(
    *,
    df: pd.DataFrame,
    x: str,
    y: str,
    agg: str,
    legend_col: str,
    distinct_on=None,
    drop_dupes_before_sum=False,
    unique_by=None,
):
    """
    Wrapper universal para adaptar _aggregate_frame
    al ecosistema agentico (barras, torta, pirámide).
    Siempre retorna:
    - columna X
    - columna y agregada
    - columna legend_col (sin agregación)
    """

    # ============================
    # Validación básica
    # ============================
    if x not in df.columns:
        raise ValueError(f"La columna X '{x}' no existe en el dataframe")

    if y not in df.columns:
        raise ValueError(f"La columna y '{y}' no existe en el dataframe")

    if legend_col not in df.columns:
        raise ValueError(f"La columna legend_col '{legend_col}' no existe en el dataframe")

    # ============================
    # Ejecutar la agregación
    # ============================
    df_agg = _aggregate_frame(
        df=df,
        xlabel=x,
        y_cols=[y],
        agg=agg,
        distinct_on=distinct_on,
        drop_dupes_before_sum=drop_dupes_before_sum,
    )

    # _aggregate_frame devuelve index = x
    df_agg = df_agg.reset_index()

    # ============================
    # Agregar la columna de categoría (legend_col)
    # ============================
    df_cat = df[[x, legend_col]].drop_duplicates()

    df_agg = df_agg.merge(df_cat, on=x, how="left")

    return df_agg


def _apply_top_n_general(df: pd.DataFrame, label_col: str, top_n: int) -> pd.DataFrame:
    """
    Aplica Top-N + 'Otros' de forma generalizada para:
      - tablas simples (1 métrica)
      - tablas pivoteadas (múltiples métricas)
      - gráficos de barras / torta

    df: DataFrame donde una columna es la etiqueta (label_col)
        y las demás columnas son valores numéricos.
    """
    # Identificar columnas de valores
    value_cols = [c for c in df.columns if c != label_col]

    # Si solo hay una columna de valores, usarla directamente
    if len(value_cols) == 1:
        value_col = value_cols[0]
        df_sorted = df.sort_values(value_col, ascending=False)

        if len(df_sorted) <= top_n:
            return df_sorted

        top_df = df_sorted.head(top_n)
        others_df = df_sorted.iloc[top_n:]

        others_value = pd.to_numeric(
            others_df[value_col], errors="coerce"
        ).fillna(0).sum()

        others_row = {label_col: "Otros", value_col: others_value}
        return pd.concat([top_df, pd.DataFrame([others_row])], ignore_index=True)

    # Si hay múltiples columnas (caso pivot), sumar para ordenar
    df["_row_total"] = df[value_cols].apply(
        lambda r: pd.to_numeric(r, errors="coerce").fillna(0).sum(), axis=1
    )
    df_sorted = df.sort_values("_row_total", ascending=False)

    if len(df_sorted) <= top_n:
        return df_sorted.drop(columns=["_row_total"])

    top_df = df_sorted.head(top_n).drop(columns=["_row_total"])
    others_df = df_sorted.iloc[top_n:]

    # Crear fila de agregación 'Otros'
    others_row = {label_col: "Otros"}
    for col in value_cols:
        others_row[col] = pd.to_numeric(
            others_df[col], errors="coerce"
        ).fillna(0).sum()

    top_df = pd.concat([top_df, pd.DataFrame([others_row])], ignore_index=True)
    return top_df

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

    # === TOP-N + Otros (solo sobre df_plot, antes de ordenar/limitar) ===
    if limit_categories and limit_categories > 0:
        label_col = df_plot.index.names[0] if isinstance(df_plot.index, pd.MultiIndex) else None

        # Si el índice es simple, convertirlo a columna
        if not label_col:
            df_plot = df_plot.reset_index()
            label_col = df_plot.columns[0]

        else:
            # MultiIndex → solo aplicamos si legend_col NO está activo
            # (Si legend_col está activo, el pivot lo maneja la función más adelante)
            df_plot = df_plot.reset_index()

        df_plot = _apply_top_n_general(df_plot, label_col, limit_categories)

        # Asegurar formato DataFrame
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
            base_color = color if isinstance(color, str) else "#2238FF"
            bar_colors = base_color
        else:
            bar_colors = color[0] if color else "#3B2DFF"

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
        palette = ["#1F2EB8", "#1a9422", "#e0830a", "#eff312",
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

def graficar_torta(
    df: pd.DataFrame,
    xlabel: Optional[Union[str, List[str]]] = None,
    y: Optional[str] = None,
    agg: Union[str, Dict[str, str]] = "sum",
    titulo: str = "Gráfico de Torta",
    color: Optional[Union[str, List[str], Dict[str, str]]] = None,
    *,
    unique_by: Optional[Union[str, List[str]]] = None,
    conditions_all: Optional[List[List[Any]]] = None,
    conditions_any: Optional[List[Union[List[Any], List[List[Any]]]]] = None,
    distinct_on: Optional[str] = None,
    drop_dupes_before_sum: bool = False,
    sort: Optional[Dict[str, str]] = None,
    limit_categories: Optional[int] = None,
    # 👇 NUEVO: mapa etiqueta → color, ej:
    # {"Riesgo Alto":"#FF0000", "Riesgo Moderado":"#FFFF00", "Riesgo Bajo":"#00AA00"}
    colors_by_category: Optional[Dict[str, str]] = None,
) -> Tuple[plt.Figure, plt.Axes]:

    if not isinstance(df, pd.DataFrame):
        raise TypeError("El parámetro 'df' debe ser un DataFrame de pandas.")

    # Aceptar múltiples columnas en X → combinar
    df2, xlabel_final, _ = _ensure_xlabel(df, xlabel)

    # Determinar y
    if y is None:
        num_cols = df2.select_dtypes(include=[np.number]).columns.tolist()
        if not num_cols and not (distinct_on and agg in ("count", "distinct_count")):
            raise ValueError("No se encontraron columnas numéricas ni 'distinct_on'.")
        y_cols = [num_cols[0]] if num_cols else []
    else:
        if y not in df2.columns:
            raise ValueError(f"La columna '{y}' no existe en el DataFrame.")
        y_cols = [y]

    col_val = y_cols[0] if y_cols else None

    # --- Filtro + deduplicación previa ---
    dff = _prefilter_df(
        df2,
        unique_by=unique_by,
        conditions_all=conditions_all,
        conditions_any=conditions_any,
    )

    # --- Agregación única ---
    df_plot = _aggregate_frame(
        dff,
        xlabel=xlabel_final,
        y_cols=y_cols,
        agg=agg,
        distinct_on=distinct_on,
        drop_dupes_before_sum=drop_dupes_before_sum,
    )

    # Normalizar a DataFrame con columna de etiqueta + valores
    if isinstance(df_plot, pd.Series):
        df_plot = df_plot.to_frame(name=col_val or "valor")

    df_plot = df_plot.reset_index()
    col_lab = df_plot.columns[0]
    col_val = df_plot.columns[-1]

    # ------------------------------
    # TOP-N + Otros (antes del sort)
    # ------------------------------
    if limit_categories and limit_categories > 0:
        df_plot = _apply_top_n_general(df_plot, col_lab, limit_categories)

    # ------------------------------
    # Eliminar NaN/inf y filtrar
    # ------------------------------
    serie = pd.to_numeric(df_plot[col_val], errors="coerce").replace([np.inf, -np.inf], np.nan)
    etiquetas = df_plot[col_lab].astype(str)

    # Quitamos filas inválidas
    valid = serie.notna()
    serie = serie[valid]
    etiquetas = etiquetas[valid]

    if serie.empty:
        raise ValueError("No hay datos válidos para graficar la torta.")

    # ------------------------------
    # sort opcional
    # ------------------------------
    if sort:
        by = sort.get("by", "y")
        asc = sort.get("order", "desc") == "asc"

        if by == "label":
            orden = etiquetas.sort_values(ascending=asc).index
        else:
            orden = serie.sort_values(ascending=asc).index

        serie = serie.loc[orden]
        etiquetas = etiquetas.loc[orden]

    # ------------------------------
    # Aplicar limit_categories después del sort (si aplica)
    # ------------------------------
    if limit_categories and limit_categories > 0:
        serie = serie.head(limit_categories)
        etiquetas = etiquetas.head(limit_categories)

    # ------------------------------
    # Validación final
    # ------------------------------
    total = serie.clip(lower=0).sum()
    if total <= 0:
        raise ValueError("La suma de valores es 0; no es posible construir la torta.")

    # ------------------------------
    # Determinar colores
    # ------------------------------
    # 1) Caso: color es un dict etiqueta → color (lo usamos directo)
    cmap_dict: Optional[Dict[str, str]] = None
    if isinstance(color, dict):
        cmap_dict = color
    elif colors_by_category:
        # 2) Caso: viene en parámetro separado
        cmap_dict = colors_by_category

    if cmap_dict is not None:
        # Mapeamos cada etiqueta a su color; si no está en el dict, usamos un fallback
        default_color = "#1A4F80"
        color = [cmap_dict.get(lbl, default_color) for lbl in etiquetas]
    else:
        # 3) Caso clásico: color = None / str / lista
        if color is None:
            cmap = plt.get_cmap("tab20")
            color = [cmap(i % cmap.N) for i in range(len(serie))]
        elif isinstance(color, str):
            color = [color] * len(serie)
        elif isinstance(color, (list, tuple)):
            color = list(color)
            if len(color) < len(serie):
                reps = int(np.ceil(len(serie) / len(color)))
                color = (color * reps)[:len(serie)]

    # ------------------------------
    # Crear figura
    # ------------------------------
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_facecolor("white")

    wedges, texts, autotexts = ax.pie(
        serie.values,
        labels=etiquetas.values,
        autopct='%1.1f%%',
        startangle=90,
        colors=color,
        shadow=False,
    )

    ax.axis('equal')
    if titulo:
        ax.set_title(titulo, fontsize=16, fontweight="bold", pad=20)

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
    sort: Optional[Dict[str, str]] = None, 
    limit_categories: Optional[int] = None,
    show_legend: bool = True,
    show_values: bool = True,
    legend_col: Optional[str] = None,
    colors_by_category: Optional[Dict[str, str]] = None,
    **kwargs,
) -> Tuple[plt.Figure, plt.Axes]:

    if not isinstance(df, pd.DataFrame):
        raise TypeError("El parámetro 'df' debe ser un DataFrame de pandas.")

    # Combinar X
    df2, xlabel_final, _ = _ensure_xlabel(df, xlabel)

    # Determinar la métrica y_cols
    if y is None:
        num_cols = df2.select_dtypes(include=[np.number]).columns.tolist()
        if not num_cols and not (distinct_on and agg in ("count", "distinct_count")):
            raise ValueError("No se encontraron columnas numéricas ni 'distinct_on'.")
        y_cols = [num_cols[0]] if num_cols else []
    else:
        if isinstance(y, list):
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

    # Claves de agrupación
    group_x = xlabel_final
    if legend_col:
        if legend_col not in dff.columns:
            raise ValueError(f"La columna de leyenda '{legend_col}' no existe.")
        if group_x is None:
            group_x = legend_col
        else:
            group_x = [group_x] if isinstance(group_x, str) else list(group_x)
            group_x.append(legend_col)

    # Agregar
    df_plot = _aggregate_frame(
        dff,
        xlabel=group_x,
        y_cols=y_cols,
        agg=agg,
        distinct_on=distinct_on,
        drop_dupes_before_sum=drop_dupes_before_sum,
    )

    # === TOP-N + Otros ===
    if limit_categories and limit_categories > 0:
        df_plot = df_plot.reset_index()
        label_col = df_plot.columns[0]
        df_plot = _apply_top_n_general(df_plot, label_col, limit_categories)
        df_plot = df_plot.set_index(label_col)

    if isinstance(df_plot, pd.Series):
        df_plot = df_plot.to_frame(name=value_col or "valor")

    # =====================================================
    # MODO 1 — SIN LEYENDA
    # =====================================================
    if not legend_col:

        if sort:
            col = df_plot.columns[0]
            by = sort.get("by", "y")
            ascending = sort.get("order", "desc") == "asc"

            if by == "label":
                df_plot = df_plot.sort_index(ascending=ascending)
            else:
                df_plot = df_plot.sort_values(by=col, ascending=ascending)

        if limit_categories:
            df_plot = df_plot.head(limit_categories)

        fig, ax = plt.subplots(figsize=(12, max(6, len(df_plot) * 0.4)), facecolor='white')
        ax.set_facecolor('#f8f9fa')

        colname = df_plot.columns[0]

        if color is None or isinstance(color, str):
            bar_color = color if isinstance(color, str) else "#332FF7"
        else:
            bar_color = color[0]

        bars = ax.barh(
            df_plot.index.astype(str),
            df_plot[colname],
            color=bar_color,
            edgecolor='white',
            linewidth=1.2,
            alpha=0.9
        )

        if titulo:
            ax.set_title(titulo, fontsize=16, fontweight='bold', pad=20)

        ax.set_xlabel(colname)
        ax.set_ylabel(xlabel_final or '')

        ax.grid(axis="x", linestyle="--", alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        if show_values:
            for bar in bars:
                v = bar.get_width()
                if pd.isna(v): 
                    continue
                ax.text(
                    v,
                    bar.get_y() + bar.get_height()/2,
                    f"{v:,.0f}" if v >= 1 else f"{v:.2f}",
                    va="center", ha="left", fontsize=9
                )

        if show_legend:
            ax.legend([colname])

        fig.tight_layout()
        return fig, ax

    # =====================================================
    # MODO 2 — CON LEYENDA (única versión limpia)
    # =====================================================

    if not isinstance(df_plot.index, pd.MultiIndex):
        raise ValueError("Se esperaba un índice MultiIndex al usar legend_col.")

    val_col = df_plot.columns[0]
    df_wide = df_plot[val_col].unstack(level=-1)

    if sort:
        ascending = sort.get("order", "desc") == "asc"
        by = sort.get("by", "y")
        if by == "label":
            df_wide = df_wide.sort_index(ascending=ascending)
        else:
            df_wide = df_wide.loc[df_wide.sum(axis=1).sort_values(ascending=ascending).index]

    if limit_categories:
        df_wide = df_wide.head(limit_categories)

    fig, ax = plt.subplots(figsize=(12, max(6, len(df_wide) * 0.4)), facecolor='white')
    ax.set_facecolor('#f8f9fa')

    y_labels = df_wide.index.astype(str).tolist()
    legend_values = df_wide.columns.tolist()
    n_series = len(legend_values)
    y_pos = np.arange(len(y_labels))
    height = 0.8 / max(n_series, 1)

    def _color_for_series(name, idx):
        if colors_by_category and name in colors_by_category:
            return colors_by_category[name]
        if isinstance(color, list) and len(color) > idx:
            return color[idx]
        if isinstance(color, str):
            return color
        palette = ["#3327E7", "#dd6666", '#58b12e', '#f39c12',
                   '#8e44ad', '#16a085', "#fffc39"]
        return palette[idx % len(palette)]

    for i, serie in enumerate(legend_values):
        vals = df_wide[serie].values
        offset = (i - (n_series - 1)/2) * height
        c = _color_for_series(str(serie), i)

        bars = ax.barh(
            y_pos + offset,
            vals,
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
                ax.text(
                    v,
                    bar.get_y() + bar.get_height()/2,
                    f"{v:,.0f}" if v >= 1 else f"{v:.2f}",
                    ha="left", va="center", fontsize=9
                )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(y_labels)
    ax.grid(axis="x", linestyle="--", alpha=0.3)

    if titulo:
        ax.set_title(titulo, fontsize=16, fontweight='bold')

    ax.set_xlabel(val_col)
    ax.set_ylabel(xlabel_final or '')

    if show_legend:
        ax.legend(title=legend_col)

    fig.tight_layout()
    return fig, ax

def _wrap_shorten_table(
    df: pd.DataFrame,
    *,
    wrap_width: int = 25,
    max_chars: int = 120,
    exclude_cols: list[str] | None = None
) -> pd.DataFrame:
    """
    Envuelve texto largo en celdas de una tabla usando saltos de línea.
    Aplica SOLO a columnas no numéricas.
    """
    exclude_cols = set(exclude_cols or [])
    out = df.copy()

    for col in out.columns:
        if col in exclude_cols:
            continue

        # Solo texto
        if not is_numeric_dtype(out[col]):
            out[col] = out[col].apply(
                lambda x: _wrap_shorten(x, max_chars=max_chars, wrap_width=wrap_width)
            )

    return out

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
    # Top-N categorías
    limit_categories: Optional[int] = None,
    # 👇 NUEVO: usar legend_col también en tablas
    legend_col: Optional[str] = None,
    # 👇 NUEVO PARAMETRO multiples métricas
    extra_measures: Optional[List[Dict[str, Any]]] = None,
    hide_main_measure: bool = False,
    add_total_row: bool = False,
    add_total_column: bool = False,
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
        conditions_any=conditions_any,
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
        where=where,
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

    # ============================================================
    # Procesar extra_measures (múltiples agregaciones)
    # ============================================================
    if extra_measures:
        for measure in extra_measures:
            col_name = measure.get("name")
            if not col_name:
                raise ValueError("Cada 'extra_measure' debe incluir 'name'.")

            # Condiciones específicas para esta medida
            m_all = measure.get("conditions_all")
            m_any = measure.get("conditions_any")

            # Agregación específica
            agg_m = measure.get("agg", agg)

            # distinct_on propio (o hereda si no está)
            distinct_m = measure.get("distinct_on", distinct_on)

            # Filtro independiente
            dff_m = _prefilter_df(
                dff,
                unique_by=unique_by,
                conditions_all=m_all,
                conditions_any=m_any,
            )

            # Agregación independiente
            df_extra = _aggregate_frame(
                dff_m,
                xlabel=group_x,
                y_cols=y_cols,
                agg=agg_m,
                distinct_on=distinct_m,
                drop_dupes_before_sum=drop_dupes_before_sum,
                where=where,
            )

            # Normalizar a DataFrame
            if isinstance(df_extra, pd.Series):
                df_extra = df_extra.to_frame(name=col_name)
            else:
                df_extra.columns = [col_name]

            # Reset index para unir
            df_extra = df_extra.reset_index()

            # Unir por la primera columna (etiqueta)
            label_col = df_plot.columns[0]
            df_plot = df_plot.merge(df_extra, on=label_col, how="left")

    # ===========================================================
    # OCULTAR LA MEDIDA PRINCIPAL (si el usuario lo solicita)
    # ===========================================================
    if hide_main_measure:
        # La primera columna SIEMPRE es el xlabel ("habito")
        first_column = df_plot.columns[0]

        # Las columnas extra son las declaradas en el JSON
        extra_cols = []
        if extra_measures:
            for m in extra_measures:
                col = m.get("name")
                if col in df_plot.columns:
                    extra_cols.append(col)

        # Construimos la tabla final SIN la medida base
        keep_cols = [first_column] + extra_cols

        # Filtrar
        df_plot = df_plot[keep_cols]

    # ===========================================================
    # TOP-N + Otros
    # ===========================================================
    if limit_categories and limit_categories > 0:
        # La primera columna SIEMPRE es la etiqueta (tras pivot/reset)
        label_col = df_plot.columns[0]
        df_plot = _apply_top_n_general(df_plot, label_col, limit_categories)

    # --- Columna de porcentaje opcional (solo cuando NO hay pivot por legend_col) ---
    if percentage_of and not legend_col:
        if percentage_of not in df_plot.columns:
            raise ValueError(f"percentage_of='{percentage_of}' no existe en la tabla resultante.")
        col_num = pd.to_numeric(df_plot[percentage_of], errors="coerce")
        total = col_num.sum()
        if total and not np.isnan(total):
            df_plot[percentage_colname] = np.where(
                col_num.notna(),
                col_num / total,  # proporción (0.45 → 45%)
                np.nan,
            )
        else:
            df_plot[percentage_colname] = np.nan

    if add_total_row:
        numeric_cols = df_plot.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            total_row = df_plot[numeric_cols].sum()
            first_col = df_plot.columns[0]
            total_row[first_col] = "TOTAL"
            total_df = pd.DataFrame([total_row])
            df_plot = pd.concat([df_plot, total_df], ignore_index=True)

    if add_total_column:
        numeric_cols = df_plot.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            df_plot["TOTAL"] = df_plot[numeric_cols].sum(axis=1)

    # ---------- Formateo base ----------
    df_display = df_plot.copy()
    for col in df_display.columns:
        if is_numeric_dtype(df_display[col]):
            if col != percentage_colname:
                df_display[col] = df_display[col].round(2)
        elif is_datetime64_any_dtype(df_display[col]):
            df_display[col] = df_display[col].dt.strftime("%Y-%m-%d")
    # Llenar NA numéricos con 0 y convertir todo a string después
    df_numeric_filled = df_plot.copy()

    for col in df_numeric_filled.columns:
        # si la columna es numérica → llenar NA con 0
        if pd.api.types.is_numeric_dtype(df_numeric_filled[col]):
            df_numeric_filled[col] = df_numeric_filled[col].fillna(0)
        else:
            # si no es numérica, llenar NA con "" (texto vacío)
            df_numeric_filled[col] = df_numeric_filled[col].fillna("")

    # Convertir todo a string para el render final
    df_display = df_numeric_filled.astype(str)

    def _format_number(x):
        if pd.isna(x):
            return ""
        if x == 0:
            return "0"           # 👈 NUEVA REGLA
        if x >= 1:
            return f"{x:,.0f}"
        return f"{x:.2f}"

    # Formatear números con separador de miles y porcentajes
    df_formatted = df_display.copy()
    for col in df_display.columns:
        try:
            numeric_col = pd.to_numeric(df_display[col], errors="coerce")
            if numeric_col.notna().any():
                if col == percentage_colname:
                    df_formatted[col] = numeric_col.apply(
                        lambda x: f"{x:.0%}" if pd.notna(x) else ""
                    )
                else:
                    df_formatted[col] = numeric_col.apply(_format_number)
        except Exception:
            pass

    # ========= Figura y tabla =========
    n_cols = len(df_formatted.columns)
    n_rows = len(df_formatted)

    # Ancho/alto base de la figura (garantiza legibilidad)
    fig_w = max(18, n_cols * 1.0)
    fig_h = max(9, n_rows * 1.0)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), facecolor="white")
    ax.axis("off")
    fig_h_mm = fig.get_size_inches()[1] * 25.4

    # === PREVENCIÓN DE TABLA VACÍA ===
    if df_formatted.empty:
        fig, ax = plt.subplots(figsize=(10, 2))
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            "No hay datos para mostrar en la tabla",
            ha="center",
            va="center",
            fontsize=18,
            fontweight="bold",
        )
        return fig, ax

    from textwrap import fill

    col_labels = [
        fill(str(c), width=18) for c in df_formatted.columns.tolist()
    ]


    # 🔹 Anchos robustos
    TEXT_COL_RATIO = 0.7
    REST_RATIO = 1 - TEXT_COL_RATIO
    col_widths = (
        [TEXT_COL_RATIO] + [REST_RATIO / (n_cols - 1)] * (n_cols - 1)
        if n_cols > 1 else [1.0]
    )
    # =========================
    # TABLA
    # =========================
    tabla = ax.table(
        cellText=df_formatted.values.tolist(),
        colLabels=col_labels,
        cellLoc="center",
        bbox=[0, 0, 1, 1],
        colWidths=col_widths,
    )

    tabla.auto_set_font_size(False)
    tabla.set_fontsize(28)
    tabla.scale(1.4, 1.8)

    for cell in tabla.get_celld().values():
        cell.get_text().set_wrap(True)
        cell.PAD = 0.3


    # =========================
    # ALTURA POR FILA (ESTABLE)
    # =========================

    def _row_height_from_text(
        text: str,
        *,
        fontsize: int,
        base: float,
        line_factor: float,
        fig_h_mmm: float,         # <-- alto de la figura en mm
        extra_mm: float = 5.0    # <-- los 5 mm extra
    ) -> float:
        n_lines = text.count("\n") + 1
        extra = extra_mm / fig_h_mmm 
        tamano = base + (n_lines * fontsize * line_factor)
        return tamano # <-- convierte mm a fracción del alto del axes

    BASE_HEIGHT = 0.02
    LINE_FACTOR = 0.0015
    TEXT_COL_IDX = 0
    FONT_SIZE_BODY = 28
    FONT_SIZE_HEADER = 36
    TEXT_COL_IDX = 0

    rows = {}
    for (row, col), cell in tabla.get_celld().items():
        rows.setdefault(row, {})[col] = cell

    for row, cols in rows.items():
        if row == 0:
            # header: medir todas las columnas
            max_text = max(
                cell.get_text().get_text() for cell in cols.values()
            )
            height = _row_height_from_text(
                max_text,
                fontsize=FONT_SIZE_HEADER,
                base=BASE_HEIGHT,
                line_factor=LINE_FACTOR,
                fig_h_mmm=fig_h_mm,
                extra_mm=5.0,
            )
        else:
            text = cols[TEXT_COL_IDX].get_text().get_text()
            height = _row_height_from_text(
                text,
                fontsize=FONT_SIZE_BODY,
                base=BASE_HEIGHT,
                line_factor=LINE_FACTOR,
                fig_h_mmm=fig_h_mm,
                extra_mm=5.0,
            )

        for cell in cols.values():
            cell.set_height(height)

    # Ajustar automáticamente el ancho de las columnas según contenido
    #for col_idx in range(n_cols):
    #    tabla.auto_set_column_width(col=col_idx)

    # ===== Estilos de colores: encabezado azul, filas blancas (o color custom) =====
    header_bg = "#0e4a8f"
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
                float(valor.replace(",", ""))
                cell.set_text_props(color="#212121", fontsize=32, weight="bold")
            except Exception:
                cell.set_text_props(color="#2c3e50", fontsize=28, weight="normal")

            cell.set_facecolor(data_bg)

        # Borde interno gris oscuro
        cell.set_edgecolor("#292929")
        cell.set_linewidth(1.2)
        cell.PAD = 0.3

    # Título (en tu flujo general se suele vaciar desde plot_from_params,
    # pero aquí mantenemos la lógica por si se usa directo)
    if titulo:
        ax.set_title(titulo, fontsize=38, fontweight="bold", pad=60, color="#141414")

    # Ajustar layout para minimizar espacios en blanco sin deformar la tabla
    #fig.tight_layout(pad=0.5)

    return fig, ax
# def _wrap_shorten_table(
#     df: pd.DataFrame,
#     *,
#     wrap_width: int = 25,
#     max_chars: int = 120,
#     exclude_cols: list[str] | None = None
# ) -> pd.DataFrame:
#     """
#     Envuelve texto largo en celdas de una tabla usando saltos de línea.
#     Aplica SOLO a columnas no numéricas.
#     """
#     exclude_cols = set(exclude_cols or [])
#     out = df.copy()

#     for col in out.columns:
#         if col in exclude_cols:
#             continue
#         if not is_numeric_dtype(out[col]):
#             out[col] = out[col].apply(
#                 lambda x: _wrap_shorten(x, max_chars=max_chars, wrap_width=wrap_width)
#             )

#     return out


# def graficar_tabla(
#     df: pd.DataFrame,
#     xlabel: Optional[Union[str, List[str]]] = None,
#     y: Optional[Union[str, List[str]]] = None,
#     agg: Union[str, Dict[str, str]] = "sum",
#     titulo: str = "Tabla de Datos",
#     color: Optional[str] = None,
#     *,
#     unique_by: Optional[Union[str, List[str]]] = None,
#     conditions_all: Optional[List[List[Any]]] = None,
#     conditions_any: Optional[List[Union[List[Any], List[List[Any]]]]] = None,
#     distinct_on: Optional[str] = None,
#     drop_dupes_before_sum: bool = False,
#     where: Optional[pd.Series] = None,
#     percentage_of: Optional[str] = None,
#     percentage_colname: str = "porcentaje",
#     limit_categories: Optional[int] = None,
#     legend_col: Optional[str] = None,
#     extra_measures: Optional[List[Dict[str, Any]]] = None,
#     hide_main_measure: bool = False,
#     add_total_row: bool = False,
#     add_total_column: bool = False,
# ) -> Tuple[plt.Figure, plt.Axes]:

#     # =========================
#     # VALIDACIÓN
#     # =========================
#     if not isinstance(df, pd.DataFrame):
#         raise TypeError("El parámetro 'df' debe ser un DataFrame de pandas.")

#     # =========================
#     # PREPARACIÓN DE DATOS
#     # =========================
#     df2, xlabel_final, _ = _ensure_xlabel(df, xlabel)

#     if y is None:
#         y_cols = df2.select_dtypes(include=[np.number]).columns.tolist()
#         if not y_cols and not (distinct_on and agg in ("count", "distinct_count")):
#             raise ValueError("No se encontraron columnas numéricas para la tabla.")
#     elif isinstance(y, str):
#         y_cols = [y]
#     else:
#         y_cols = y

#     dff = _prefilter_df(
#         df2,
#         unique_by=unique_by,
#         conditions_all=conditions_all,
#         conditions_any=conditions_any,
#     )

#     group_x = xlabel_final
#     if legend_col:
#         group_x = [group_x, legend_col] if group_x else legend_col

#     df_plot = _aggregate_frame(
#         dff,
#         xlabel=group_x,
#         y_cols=y_cols,
#         agg=agg,
#         distinct_on=distinct_on,
#         drop_dupes_before_sum=drop_dupes_before_sum,
#         where=where,
#     )

#     if isinstance(df_plot, pd.Series):
#         df_plot = df_plot.to_frame(name=y_cols[0] if y_cols else "valor")

#     df_plot = df_plot.reset_index()

#     if hide_main_measure and extra_measures:
#         keep = [df_plot.columns[0]] + [
#             m["name"] for m in extra_measures
#             if isinstance(m, dict) and m.get("name") in df_plot.columns
#         ]
#         df_plot = df_plot[keep]

#     if limit_categories:
#         df_plot = _apply_top_n_general(df_plot, df_plot.columns[0], limit_categories)

#     if percentage_of and percentage_of in df_plot.columns:
#         total = pd.to_numeric(df_plot[percentage_of], errors="coerce").sum()
#         df_plot[percentage_colname] = (
#             pd.to_numeric(df_plot[percentage_of], errors="coerce") / total
#             if total else 0
#         )

#     if add_total_row:
#         total_row = df_plot.select_dtypes(include=[np.number]).sum(numeric_only=True)
#         total_row[df_plot.columns[0]] = "TOTAL"
#         df_plot = pd.concat([df_plot, pd.DataFrame([total_row])], ignore_index=True)

#     if add_total_column:
#         num_cols = df_plot.select_dtypes(include=[np.number]).columns
#         if len(num_cols) > 0:
#             df_plot["TOTAL"] = df_plot[num_cols].sum(axis=1)

#     # =========================
#     # FORMATEO + WRAP ADAPTATIVO
#     # =========================
#     df_plot = df_plot.fillna("")
#     numeric_cols = df_plot.select_dtypes(include=[np.number]).columns.tolist()
#     df_formatted = df_plot.astype(str)

#     first_col = df_formatted.columns[0]
#     max_len_first = df_formatted[first_col].str.len().max() if len(df_formatted) else 0

#     if max_len_first >= 140:
#         wrap_width, max_chars = 52, 240
#     elif max_len_first >= 90:
#         wrap_width, max_chars = 44, 210
#     else:
#         wrap_width, max_chars = 30, 150

#     df_formatted = _wrap_shorten_table(
#         df_formatted,
#         wrap_width=wrap_width,
#         max_chars=max_chars,
#         exclude_cols=numeric_cols,
#     )

#     # =========================
#     # FIGURA
#     # =========================
#     n_cols = len(df_formatted.columns)
#     n_rows = len(df_formatted)

#     fig_w = max(12, n_cols * 3.0)
#     fig_h = max(4.8, (n_rows + 1) * 0.9)

#     fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")
#     ax = fig.add_axes([0, 0, 1, 1])
#     ax.axis("off")

#     # =========================
#     # HEADERS
#     # =========================
#     from textwrap import fill
#     header_wrap = 18 if n_cols <= 3 else 14
#     col_labels = [fill(str(c), width=header_wrap) for c in df_formatted.columns]

#     # =========================
#     # ANCHOS DE COLUMNA
#     # =========================
#     if n_cols == 1:
#         col_widths = [1.0]
#     else:
#         if n_cols == 2:
#             text_ratio = 0.80 if max_len_first >= 120 else 0.76 if max_len_first >= 80 else 0.70
#         else:
#             text_ratio = 0.68 if max_len_first >= 120 else 0.64 if max_len_first >= 80 else 0.60

#         rest_ratio = 1.0 - text_ratio
#         col_widths = [text_ratio] + [rest_ratio / (n_cols - 1)] * (n_cols - 1)

#     # =========================
#     # TABLA
#     # =========================
#     tabla = ax.table(
#         cellText=df_formatted.values.tolist(),
#         colLabels=col_labels,
#         cellLoc="center",
#         colWidths=col_widths,
#         bbox=[0, 0, 1, 1],
#     )

#     tabla.auto_set_font_size(False)
#     tabla.set_fontsize(28)

#     for cell in tabla.get_celld().values():
#         cell.get_text().set_wrap(True)
#         cell.PAD = 0.22

#     # =========================
#     # ALTURAS POR FILA + BBOX DINÁMICO
#     # =========================
#     HEADER_BASE, HEADER_LINE = 0.11, 0.06
#     BODY_BASE, BODY_LINE = 0.085, 0.04

#     rows = {}
#     for (r, c), cell in tabla.get_celld().items():
#         rows.setdefault(r, []).append(cell)

#     desired_heights = {}
#     for r, cells in rows.items():
#         max_lines = max(cell.get_text().get_text().count("\n") + 1 for cell in cells)
#         desired_heights[r] = (
#             HEADER_BASE + (max_lines - 1) * HEADER_LINE
#             if r == 0 else
#             BODY_BASE + (max_lines - 1) * BODY_LINE
#         )

#     total_h = sum(desired_heights.values())
#     bbox_h = min(0.98, max(0.35, total_h * 1.05))
#     y0 = (1 - bbox_h) / 2
#     tabla._bbox = [0.01, y0, 0.98, bbox_h]

#     scale = bbox_h / total_h if total_h else 1
#     for r, cells in rows.items():
#         for cell in cells:
#             cell.set_height(desired_heights[r] * scale)

#     # =========================
#     # ESTILOS
#     # =========================
#     header_bg = "#0e4a8f"
#     data_bg = color if color else "#ffffff"

#     for (r, c), cell in tabla.get_celld().items():
#         if r == 0:
#             cell.set_facecolor(header_bg)
#             cell.set_text_props(weight="bold", fontsize=34, color="white")
#         else:
#             if c == 0:
#                 cell.set_text_props(fontsize=25, color="#2c3e50")
#             else:
#                 cell.set_text_props(fontsize=25, weight="bold", color="#212121")
#             cell.set_facecolor(data_bg)

#         cell.set_edgecolor("#292929")
#         cell.set_linewidth(1.2)

#     if titulo:
#         ax.set_title(titulo, fontsize=30, fontweight="bold", pad=20, color="#141414")

#     return fig, ax



def graficar_piramide(
    df: pd.DataFrame,
    xlabel: Optional[Union[str, List[str]]] = None,
    y: Optional[Union[str, List[str]]] = None,
    agg: Union[str, Dict[str, str]] = "count",
    titulo: str = "Gráfico de Pirámide",
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
    legend_col: Optional[str] = None,
    colors_by_category: Optional[Dict[str, str]] = None,
    show_legend: bool = True,
    show_values: bool = True,
    figsize: Tuple[float, float] = (9, 6),
    **kwargs,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Gráfico de pirámide poblacional compatible con plot_from_params.

    Típico JSON desde plot_from_params:
      {
        "function_name": "graficar_piramide",
        "xlabel": "grupo_edad",
        "legend_col": "genero",
        "y": "documento",
        "agg": "distinct_count"   # o "count"
      }
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("El parámetro 'df' debe ser un DataFrame de pandas.")

    # -----------------------------
    # 1. Normalizar X (xlabel)
    # -----------------------------
    df2, xlabel_final, _ = _ensure_xlabel(df, xlabel)

    if xlabel_final is None:
        raise ValueError("Se requiere 'xlabel' para graficar la pirámide.")

    # -----------------------------
    # 2. Validar legend_col
    # -----------------------------
    if legend_col is None:
        raise ValueError("Se requiere 'legend_col' para la pirámide poblacional.")

    if legend_col not in df2.columns:
        raise ValueError(f"La columna de leyenda '{legend_col}' no existe en el DataFrame.")

    # -----------------------------
    # 3. Determinar columna métrica (y_col)
    # -----------------------------
    if y is None:
        # Si no especifican Y:
        # - si hay numeric, usamos la primera
        # - si hay distinct_on, usamos distinct_on
        # - si no, creamos una columna de 1s para contar
        num_cols = df2.select_dtypes(include=[np.number]).columns.tolist()
        if num_cols:
            y_col = num_cols[0]
        elif distinct_on and distinct_on in df2.columns:
            y_col = distinct_on
        else:
            y_col = "__ones__"
            df2 = df2.copy()
            df2[y_col] = 1
    else:
        if isinstance(y, list):
            if not y:
                raise ValueError("Lista 'y' vacía.")
            y_col = y[0]
        else:
            y_col = y

        if y_col not in df2.columns:
            # Permitimos que _aggregate_frame use distinct_on si el agg es count/distinct_count
            if not (isinstance(agg, str) and agg in ("count", "distinct_count") and distinct_on in df2.columns):
                raise ValueError(f"La columna '{y_col}' no existe en el DataFrame.")

    # -----------------------------
    # 4. Filtro/deduplicación previa
    # -----------------------------
    dff = _prefilter_df(
        df2,
        unique_by=unique_by,
        conditions_all=conditions_all,
        conditions_any=conditions_any,
    )

    if dff.empty:
        fig, ax = plt.subplots(figsize=figsize)
        ax.axis("off")
        msg = "No hay datos para graficar la pirámide (dataset vacío tras filtros)."
        ax.text(0.5, 0.5, msg, ha="center", va="center", fontsize=14, weight="bold")
        if titulo:
            ax.set_title(titulo, fontsize=16, weight="bold", pad=20)
        fig.tight_layout()
        return fig, ax

    # -----------------------------
    # 5. Agregación: X + legend_col
    # -----------------------------
    df_agg = _aggregate_frame(
        dff,
        xlabel=[xlabel_final, legend_col],
        y_cols=[y_col],
        agg=agg,
        distinct_on=distinct_on,
        drop_dupes_before_sum=drop_dupes_before_sum,
    )

    if isinstance(df_agg, pd.Series):
        df_agg = df_agg.to_frame(name=y_col)

    df_agg_reset = df_agg.reset_index()

    if df_agg_reset.empty:
        fig, ax = plt.subplots(figsize=figsize)
        ax.axis("off")
        msg = "No hay datos agregados para construir la pirámide."
        ax.text(0.5, 0.5, msg, ha="center", va="center", fontsize=14, weight="bold")
        if titulo:
            ax.set_title(titulo, fontsize=16, weight="bold", pad=20)
        fig.tight_layout()
        return fig, ax

    # -----------------------------
    # 6. Identificar columna de valores
    # -----------------------------
    id_cols = [xlabel_final, legend_col]
    value_cols = [c for c in df_agg_reset.columns if c not in id_cols]
    if not value_cols:
        fig, ax = plt.subplots(figsize=figsize)
        ax.axis("off")
        msg = "No se encontró ninguna columna de valores para la pirámide."
        ax.text(0.5, 0.5, msg, ha="center", va="center", fontsize=14, weight="bold")
        if titulo:
            ax.set_title(titulo, fontsize=16, weight="bold", pad=20)
        fig.tight_layout()
        return fig, ax

    value_col = value_cols[0]

    # -----------------------------
    # 7. Pivot: filas = grupos, columnas = categorías (2)
    # -----------------------------
    pivot = df_agg_reset.pivot(
        index=xlabel_final,
        columns=legend_col,
        values=value_col
    ).fillna(0)

    # Si el pivot quedó completamente vacío
    if pivot.empty or pivot.shape[1] == 0:
        fig, ax = plt.subplots(figsize=figsize)
        ax.axis("off")
        msg = (
            f"No hay datos para construir la pirámide "
            f"(pivot vacío para xlabel='{xlabel_final}' y legend_col='{legend_col}')."
        )
        ax.text(0.5, 0.5, msg, ha="center", va="center", fontsize=14, weight="bold")
        if titulo:
            ax.set_title(titulo, fontsize=16, weight="bold", pad=20)
        fig.tight_layout()
        return fig, ax

    # -----------------------------
    # 8. SORT y TOP-N
    #   - Si el usuario NO manda sort, ordenamos por total de mayor a menor
    # -----------------------------
    if sort:
        by = sort.get("by", "y")
        order = sort.get("order", "asc")
        ascending = (order == "asc")

        if by == "label":
            pivot = pivot.sort_index(ascending=ascending)
        else:
            row_sum = pivot.sum(axis=1)
            pivot = pivot.loc[row_sum.sort_values(ascending=ascending).index]
    else:
        # Orden por total de cada grupo (mayor a menor) por defecto
        row_sum = pivot.sum(axis=1)
        pivot = pivot.loc[row_sum.sort_values(ascending=False).index]

    if limit_categories and limit_categories > 0:
        pivot = pivot.head(limit_categories)

    categorias = pivot.index.tolist()
    columnas = list(pivot.columns)

    if len(columnas) != 2:
        raise ValueError(
            f"La pirámide requiere EXACTAMENTE 2 categorías en '{legend_col}'. "
            f"Encontradas: {columnas}"
        )

    izquierda, derecha = columnas

    vals_izq = -pivot[izquierda].astype(float).values  # lado izquierdo (negativos)
    vals_der = pivot[derecha].astype(float).values     # lado derecho (positivos)

    # -----------------------------
    # 9. Colores
    # -----------------------------
    DEFAULT = ["#4A90E2", "#E94F37"]

    if colors_by_category:
        color_izq = colors_by_category.get(str(izquierda), DEFAULT[0])
        color_der = colors_by_category.get(str(derecha), DEFAULT[1])
    elif isinstance(color, list) and len(color) >= 2:
        color_izq, color_der = color[0], color[1]
    elif isinstance(color, str):
        color_izq, color_der = color, DEFAULT[1]
    else:
        color_izq, color_der = DEFAULT

    # -----------------------------
    # 10. Figura
    # -----------------------------
    fig, ax = plt.subplots(figsize=figsize)

    ax.barh(categorias, vals_izq, label=str(izquierda), color=color_izq)
    ax.barh(categorias, vals_der, label=str(derecha), color=color_der)

    if show_values:
        for i, v in enumerate(vals_izq):
            ax.text(v, i, f"{abs(v):,.0f}", va="center", ha="right", fontsize=8)
        for i, v in enumerate(vals_der):
            ax.text(v, i, f"{v:,.0f}", va="center", ha="left", fontsize=8)

    ax.axvline(0, color="gray", linewidth=1.2)
    ax.grid(axis="x", linestyle="--", color="#CCCCCC", alpha=0.6)

    ax.set_xlabel("Población")
    ax.set_ylabel(xlabel_final)

    if titulo:
        ax.set_title(titulo, fontsize=16, weight="bold")

    # -----------------------------
    # 11. Leyenda siempre que show_legend=True
    # -----------------------------
    if show_legend:
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)

    fig.tight_layout()
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
def plot_from_params(df: pd.DataFrame, params: Dict[str, Any], *, show: bool = False) -> Tuple[plt.Figure, plt.Axes]:
    """
    Router unificado para graficar según JSON de parámetros.
    - Soporta: graficar_barras, graficar_barras_horizontal, graficar_torta, graficar_tabla
    - Respeta binning, stack_columns, Top-N + 'Otros', legend_col, percentage_of, etc.
    """

    import copy

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df debe ser un DataFrame de pandas")

    p = copy.deepcopy(params) or {}

    # -------------------------------------------------
    # 1. Normalización del selector de función
    # -------------------------------------------------
    fn_raw = p.get("function_name") or p.get("chart_type")
    if fn_raw is None:
        raise ValueError("Falta 'function_name' o 'chart_type' en params.")

    fn_key = str(fn_raw).strip().lower()

    DISPATCH = {
        # nombres canónicos
        "graficar_barras": graficar_barras,
        "graficar_barras_horizontal": graficar_barras_horizontal,
        "graficar_torta": graficar_torta,
        "graficar_tabla": graficar_tabla,
        "graficar_piramide": graficar_piramide,

        # alias
        "barras": graficar_barras,
        "bar": graficar_barras,
        "column": graficar_barras,
        "col": graficar_barras,

        "barras_horizontal": graficar_barras_horizontal,
        "horizontal": graficar_barras_horizontal,
        "barh": graficar_barras_horizontal,

        "torta": graficar_torta,
        "pie": graficar_torta,
        "pastel": graficar_torta,

        "tabla": graficar_tabla,
        "table": graficar_tabla,

        "piramide": graficar_piramide,
        "piram": graficar_piramide,
    }

    func = DISPATCH.get(fn_key)
    if func is None:
        raise ValueError(f"Función de gráfico no reconocida: {fn_raw!r} (normalizado: {fn_key!r})")

    # -------------------------------------------------
    # 2. Parámetros base (comunes)
    # -------------------------------------------------
    xlabel = p.get("xlabel")
    y      = p.get("y")
    agg    = p.get("agg", "sum")

    # En tu flujo el título final viene del HTML, así que lo dejamos vacío
    titulo = ""
    # Si quisieras volver a usarlo desde params:
    # titulo = p.get("title", "Gráfico")

    color  = p.get("color")
    tipo   = p.get("tipo")  # usado solo para formateo final

    # -------------------------------------------------
    # 3. Formateo general (ticks, wraps, etc.)
    # -------------------------------------------------
    tick_fontsize = int(p.get("tick_fontsize", 9))
    rotation      = p.get("rotation", 35)
    wrap_width_x  = int(p.get("wrap_width_x", 20))
    wrap_width_y  = int(p.get("wrap_width_y", 30))
    max_chars_x   = int(p.get("max_chars_x", 85))
    max_chars_y   = int(p.get("max_chars_y", 120))
    legend_out    = bool(p.get("legend_outside", True))

    # Tortas (labels)
    pie_max_chars = int(p.get("pie_max_chars", 60))
    pie_wrap_w    = int(p.get("pie_wrap_width", 25))

    # Sort general (solo lo usaremos en barras / horizontal / torta)
    sort = p.get("sort")
    if sort is not None and not isinstance(sort, dict):
        sort = None

    # -------------------------------------------------
    # 4. Limpieza de condiciones (sin romper bloques)
    # -------------------------------------------------
    def _clean_conditions(cond_raw):
        """
        Acepta:
          - None / [] → []
          - [[col,op,val], ...]
          - [[[c1],[c2]], [c3], ...]  (bloques AND dentro de OR)
        Devuelve una estructura equivalente pero sin filas vacías/mal formadas.
        """
        if not cond_raw:
            return []

        cleaned = []

        for item in cond_raw:
            if not item:
                continue

            # Caso simple: ["col","==",5]
            if isinstance(item, (list, tuple)) and len(item) == 3 and not isinstance(item[0], (list, tuple)):
                col, op, val = item
                if col is None or str(col).strip() == "":
                    continue
                cleaned.append([col, op, val])
                continue

            # Caso bloque: [[c1],[c2],...]
            if isinstance(item, (list, tuple)) and item and isinstance(item[0], (list, tuple)):
                block = []
                for c in item:
                    if not isinstance(c, (list, tuple)) or len(c) != 3:
                        continue
                    col, op, val = c
                    if col is None or str(col).strip() == "":
                        continue
                    block.append([col, op, val])
                if block:
                    cleaned.append(block)
                continue

            # Otros formatos se ignoran
        return cleaned

    unique_by      = p.get("unique_by")
    conditions_all = _clean_conditions(p.get("conditions_all"))
    conditions_any = _clean_conditions(p.get("conditions_any"))
    distinct_on    = p.get("distinct_on")
    drop_dupes_before_sum = bool(p.get("drop_dupes_before_sum", False))

    # -------------------------------------------------
    # 5. Binning (agrupación por rangos)
    # -------------------------------------------------
    binning = p.get("binning")
    df_work = df
    if binning:
        df_work, bucket_col = _apply_binning(df_work, binning)
        xlabel = bucket_col  # la nueva X es el bucket

    # -------------------------------------------------
    # 6. PREFILTRO antes del stack (para stack_columns)
    # -------------------------------------------------
    df_filtered = _prefilter_df(
        df_work,
        unique_by=unique_by,
        conditions_all=conditions_all,
        conditions_any=conditions_any,
    )

    # -------------------------------------------------
    # 7. STACK opcional (apilar columnas)
    # -------------------------------------------------
    stack = p.get("stack_columns")
    if stack:
        cols      = stack["columns"]
        out_col   = stack.get("output_col", "tipo_riesgo")
        val_col   = stack.get("value_col", "valor")
        label_map = stack.get("label_map")
        keep_val  = stack.get("keep_value")

        df_filtered = _stack_columns(
            df_filtered,
            cols,
            output_col=out_col,
            value_col=val_col,
            label_map=label_map,
        )

        if keep_val not in (None, "any", "", []):
            df_filtered = df_filtered[df_filtered[val_col] == keep_val]
        if xlabel is None:
            xlabel = out_col  # ahora la X es el nombre del stack

    # -------------------------------------------------
    # 8. Top-N (solo se pasa, la lógica vive en las funciones graficar_*)
    # -------------------------------------------------
    limit_categories = p.get("limit_categories")
    if isinstance(limit_categories, str) and limit_categories.isdigit():
        limit_categories = int(limit_categories)

    # -------------------------------------------------
    # 9. Construir kwargs comunes para TODAS las funciones
    # -------------------------------------------------
    common_kwargs: Dict[str, Any] = dict(
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
        limit_categories=limit_categories,
    )

    # ---- Config específica de TABLAS ----
    if func is graficar_tabla:
        percentage_of     = p.get("percentage_of")
        percentage_colname = p.get("percentage_colname", "porcentaje")
        common_kwargs["percentage_of"] = percentage_of
        common_kwargs["percentage_colname"] = percentage_colname

        # legend_col también usable en tablas (pivot columnas)
        legend_col = p.get("legend_col")
        if legend_col is not None:
            common_kwargs["legend_col"] = legend_col
        # === NUEVO: múltiples medidas adicionales ===
        extra_measures = p.get("extra_measures")
        if extra_measures is not None:
            if not isinstance(extra_measures, list):
                raise ValueError("'extra_measures' debe ser una lista de objetos.")
            common_kwargs["extra_measures"] = extra_measures
        # Nueva opción: ocultar la métrica principal
            hide_main_measure = p.get("hide_main_measure", False)
            common_kwargs["hide_main_measure"] = hide_main_measure
        add_total_row = p.get("add_total_row", False)
        add_total_column = p.get("add_total_column", False)
        common_kwargs["add_total_row"] = add_total_row
        common_kwargs["add_total_column"] = add_total_column

    # ---- Config específica de BARRAS / HORIZONTAL ----
    if func.__name__ in ("graficar_barras", "graficar_barras_horizontal", "graficar_piramide"):
        legend_col = p.get("legend_col")
        if legend_col is not None:
            common_kwargs["legend_col"] = legend_col
        colors_by_category = p.get("colors_by_category")
        if colors_by_category is not None:
            common_kwargs["colors_by_category"] = colors_by_category
        if sort is not None:
            common_kwargs["sort"] = sort

    # ---- Config específica de TORTA ----
    if func is graficar_torta:
        # torta solo soporta una métrica → normaliza y si vino lista
        if isinstance(y, list):
            common_kwargs["y"] = y[0] if y else None
        if sort is not None:
            common_kwargs["sort"] = sort
        colors_by_category = p.get("colors_by_category")
        if colors_by_category is not None:
            common_kwargs["colors_by_category"] = colors_by_category

    # -------------------------------------------------
    # 10. Evitar doble filtrado dentro de las funciones
    # -------------------------------------------------
    safe_kwargs = common_kwargs.copy()
    safe_kwargs["conditions_all"] = None
    safe_kwargs["conditions_any"] = None
    safe_kwargs["unique_by"] = None

    # Llamado real a la función de gráfico
    fig, ax = func(df_filtered, **safe_kwargs)

    # -------------------------------------------------
    # 11. POST-FORMATEO: wrapping de labels, leyenda, etc.
    # -------------------------------------------------
    # Determinar tipo final si no vino explícito
    t = (tipo or "").lower()
    if not t:
        ct = (p.get("chart_type") or fn_key or "").lower()
        if ct in ("barras", "bar", "column", "col", "graficar_barras"):
            t = "barras"
        elif ct in ("horizontal", "barh", "barras_horizontal", "graficar_barras_horizontal"):
            t = "barh"
        elif ct in ("torta", "pie", "pastel", "graficar_torta"):
            t = "torta"
        elif ct in ("tabla", "table", "graficar_tabla"):
            t = "tabla"
        elif ct in ("piramide", "graficar_piramide", "piram"):
            t = "piramide"

    if t in ("barras", "bar", "column", "col"):
        _wrap_shorten_ticks(
            ax,
            axis="x",
            wrap_width=wrap_width_x,
            max_chars=max_chars_x,
            fontsize=tick_fontsize,
            rotation=rotation,
        )

    elif t in ("barh", "barras_h", "horizontal", "barra_horizontal"):
        _wrap_shorten_ticks(
            ax,
            axis="y",
            wrap_width=wrap_width_y,
            max_chars=max_chars_y,
            fontsize=tick_fontsize,
        )

    elif t in ("torta", "pie", "pastel"):
        # Tomar SOLO textos que no son porcentajes (autopct)
        labels_raw = [txt.get_text() for txt in ax.texts if "%" not in txt.get_text()]
        labels_fmt = _format_pie_labels(
            labels_raw,
            max_chars=pie_max_chars,
            wrap_width=pie_wrap_w,
        )
        _apply_pie_texts(ax, labels_fmt)
    
    elif t in ("piramide", "piram"):
        _wrap_shorten_ticks(
            ax,
            axis="x",
            wrap_width=wrap_width_x,
            max_chars=max_chars_x,
            fontsize=tick_fontsize,
            rotation=rotation,
        )



    _finalize_layout(fig, ax, legend_outside=legend_out)

    if show:
        plt.tight_layout()
        plt.show()

    return fig, ax


# def plot_from_params(df, params, *, show: bool=False):
#     import copy
#     import matplotlib.pyplot as plt

#     p = copy.deepcopy(params)

#     # --- Normalización del selector ---
#     fn_raw = p.get("function_name") or p.get("chart_type")
#     if fn_raw is None:
#         raise ValueError("Falta 'function_name' o 'chart_type' en params.")

#     fn_key = str(fn_raw).strip().lower()

#     DISPATCH = {
#         "graficar_barras": graficar_barras,
#         "graficar_barras_horizontal": graficar_barras_horizontal,
#         "graficar_torta": graficar_torta,
#         "graficar_tabla": graficar_tabla,

#         "barras": graficar_barras,
#         "bar": graficar_barras,
#         "barras_horizontal": graficar_barras_horizontal,
#         "horizontal": graficar_barras_horizontal,

#         "torta": graficar_torta,
#         "pie": graficar_torta,

#         "tabla": graficar_tabla,
#         "table": graficar_tabla,
#     }

#     func = DISPATCH.get(fn_key)
#     if func is None:
#         raise ValueError(f"Función de gráfico no reconocida: {fn_raw!r} (normalizado: {fn_key!r})")

#     # --- Parámetros comunes ---
#     xlabel  = p.get("xlabel")
#     y       = p.get("y")
#     agg     = p.get("agg", "sum")
#     titulo  = ""
#     #titulo  = p.get("title", "Gráfico")
#     color   = p.get("color")
#     tipo    = p.get("tipo")

#     # --- Formateo general ---
#     tick_fontsize = int(p.get("tick_fontsize", 9))
#     rotation      = p.get("rotation", 35)
#     wrap_width_x  = int(p.get("wrap_width_x", 20))
#     wrap_width_y  = int(p.get("wrap_width_y", 30))
#     max_chars_x   = int(p.get("max_chars_x", 85))
#     max_chars_y   = int(p.get("max_chars_y", 120))
#     legend_out    = bool(p.get("legend_outside", True))

#     # --- Formateo torta ---
#     pie_max_chars = int(p.get("pie_max_chars", 60))
#     pie_wrap_w    = int(p.get("pie_wrap_width", 25))

#     # --- Limpieza de condiciones ---
#     def _clean_conditions(cond):
#         if not cond:
#             return []
#         cleaned = []
#         for c in cond:
#             if not c:
#                 continue
#             if not isinstance(c, (list, tuple)):
#                 continue
#             if len(c) != 3:
#                 continue
#             col, op, val = c
#             if col is None or col == "" or str(col).strip() == "":
#                 continue
#             cleaned.append([col, op, val])
#         return cleaned

#     # --- Extra: parámetros faltantes ---
#     unique_by             = p.get("unique_by")
#     conditions_all        = _clean_conditions(p.get("conditions_all"))
#     conditions_any        = _clean_conditions(p.get("conditions_any"))
#     distinct_on           = p.get("distinct_on")
#     drop_dupes_before_sum = p.get("drop_dupes_before_sum", False)

#     # --- BINNING ---
#     binning = p.get("binning")
#     if binning:
#         df, bucket_col = _apply_binning(df, binning)
#         xlabel = bucket_col

#     # --- PRE-FILTRADO ANTES DEL STACK ---
#     df_filtered = _prefilter_df(
#         df,
#         unique_by=unique_by,
#         conditions_all=conditions_all,
#         conditions_any=conditions_any,
#     )

#     # --- STACK ---
#     stack = p.get("stack_columns")
#     if stack:
#         cols      = stack["columns"]
#         out_col   = stack.get("output_col", "tipo_riesgo")
#         val_col   = stack.get("value_col", "valor")
#         label_map = stack.get("label_map")
#         keep_val  = stack.get("keep_value")

#         df_filtered = _stack_columns(df_filtered, cols, output_col=out_col, value_col=val_col, label_map=label_map)

#         if keep_val not in (None, "any", "", []):
#             df_filtered = df_filtered[df_filtered[val_col] == keep_val]

#         xlabel = out_col


#     # --- Preparar kwargs comunes ---
#     common_kwargs = dict(
#         xlabel=xlabel,
#         y=y,
#         agg=agg,
#         titulo=titulo,
#         color=color,
#         unique_by=unique_by,
#         conditions_all=conditions_all,
#         conditions_any=conditions_any,
#         distinct_on=distinct_on,
#         drop_dupes_before_sum=drop_dupes_before_sum,
#         limit_categories=p.get("limit_categories"),
#     )
#         # Config específica de tablas: columna de porcentaje
#     if func is graficar_tabla:
#         percentage_of = p.get("percentage_of")
#         percentage_colname = p.get("percentage_colname", "porcentaje")
#         common_kwargs["percentage_of"] = percentage_of
#         common_kwargs["percentage_colname"] = percentage_colname

#     # legend_col también usable en tablas (para pivotear columnas)
#     if func is graficar_tabla:
#         legend_col = p.get("legend_col")
#         if legend_col is not None:
#             common_kwargs["legend_col"] = legend_col

            
#     # --- Normalización mínima de torta ---
#     if fn_key in ("graficar_torta", "torta", "pie") and isinstance(y, list):
#         y = y[0] if y else None
#         common_kwargs["y"] = y
    
#     # --- Config específica de barras: leyendas / colores por categoría ---
#     if func in (graficar_barras, graficar_barras_horizontal):
#         legend_col = p.get("legend_col")
#         colors_by_category = p.get("colors_by_category")

#         if legend_col is not None:
#             common_kwargs["legend_col"] = legend_col
#         if colors_by_category is not None:
#             common_kwargs["colors_by_category"] = colors_by_category

#     # === Llamado central a la función sin repetir filtros internos ===
#     safe_kwargs = common_kwargs.copy()
#     safe_kwargs["conditions_all"] = None
#     safe_kwargs["conditions_any"] = None
#     safe_kwargs["unique_by"] = None

#     fig, ax = func(df_filtered, **safe_kwargs)


#     # --- Post-formateo según el tipo ---
#     t = (tipo or "").lower()

#     if not t:
#         ct = (p.get("chart_type") or "").lower()
#         if ct in ("barras", "bar", "column", "col"):
#             t = "barras"
#         elif ct in ("horizontal", "barh", "barras_horizontal", "graficar_barras_horizontal"):
#             t = "barh"
#         elif ct in ("torta", "pie", "pastel", "graficar_torta"):
#             t = "torta"
#         elif ct in ("tabla", "table", "graficar_tabla"):
#             t = "tabla"

#     if t in ("barras", "bar", "column", "col"):
#         _wrap_shorten_ticks(ax, axis="x", wrap_width=wrap_width_x, max_chars=max_chars_x, fontsize=tick_fontsize, rotation=rotation)

#     elif t in ("barh", "barras_h", "horizontal", "barra_horizontal"):
#         _wrap_shorten_ticks(ax, axis="y", wrap_width=wrap_width_y, max_chars=max_chars_y, fontsize=tick_fontsize)

#     elif t in ("torta", "pie", "pastel"):
#         # Tomar SOLO los textos que no son porcentajes (autopct)
#         labels_raw = [txt.get_text() for txt in ax.texts if "%" not in txt.get_text()]

#     # Aplicar abreviación / wrap
#         labels_fmt = _format_pie_labels(
#             labels_raw,
#             max_chars=pie_max_chars,
#             wrap_width=pie_wrap_w
#         )
#         _apply_pie_texts(ax, labels_fmt)


#     _finalize_layout(fig, ax, legend_outside=legend_out)

#     if show:
#         plt.tight_layout()
#         plt.show()

#     return fig, ax

