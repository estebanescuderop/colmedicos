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
            return pd.DataFrame({"unique_count": [val]})
        else:
            s = _df.groupby(group_keys, dropna=False)[distinct_on].nunique(dropna=True)
            return s.to_frame(name="unique_count")

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

import plotly.express as px
import plotly.graph_objects as go

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
    show_values: bool = False,
    **kwargs,   # absorbe cualquier otro parámetro enviado
) -> Tuple[plt.Figure, plt.Axes]:

    # --------------------------------------------
    # Validación base
    # --------------------------------------------
    if not isinstance(df, pd.DataFrame):
        raise TypeError("El parámetro 'df' debe ser un DataFrame de pandas.")

    # Unificar X (acepta lista -> crea multiindex si aplica)
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
        if y not in df2.columns:
            raise ValueError(f"La columna '{y}' no existe en el DataFrame.")
        y_cols = [y]

    # --------------------------------------------
    # Aplicar filtro + deduplicación previa
    # --------------------------------------------
    dff = _prefilter_df(
        df2,
        unique_by=unique_by,
        conditions_all=conditions_all,
        conditions_any=conditions_any,
    )

    # --------------------------------------------
    # Agregar (usa tu agregador unificado)
    # --------------------------------------------
    df_plot = _aggregate_frame(
        dff,
        xlabel=xlabel_final,
        y_cols=y_cols,
        agg=agg,
        distinct_on=distinct_on,
        drop_dupes_before_sum=drop_dupes_before_sum,
    )

    # Asegurar DataFrame
    if isinstance(df_plot, pd.Series):
        df_plot = df_plot.to_frame(name=y_cols[0] if y_cols else "valor")

    # --------------------------------------------
    # Sort opcional
    # --------------------------------------------
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

    # --------------------------------------------
    # Construir figura con estilo profesional
    # --------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 7), facecolor='white')
    ax.set_facecolor('#f8f9fa')

    colname = df_plot.columns[0]

    # Paleta de colores profesional si no se especifica
    if color is None:
        color = '#5A6C7D'  # Azul grisáceo corporativo

    bars = ax.bar(df_plot.index.astype(str), df_plot[colname],
                  color=color, edgecolor='white', linewidth=1.2, alpha=0.9)

    # Agregar sombra sutil a las barras
    for bar in bars:
        bar.set_zorder(3)

    # Título con mejor formato
    ax.set_title(titulo, fontsize=16, fontweight='bold', pad=20, color="#000000")
    ax.set_ylabel(colname, fontsize=11, fontweight='600', color='#34495e')
    ax.set_xlabel(xlabel_final if xlabel_final else '', fontsize=11, fontweight='600', color='#34495e')

    # Grid mejorado
    ax.grid(axis="y", linestyle="--", alpha=0.3, color='#bdc3c7', zorder=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#bdc3c7')
    ax.spines['bottom'].set_color('#bdc3c7')

    # Rotar etiquetas con mejor ángulo
    ax.tick_params(axis="x", rotation=45, labelsize=9, colors='#34495e')
    ax.tick_params(axis="y", labelsize=9, colors='#34495e')

    # Valores sobre las barras con formato mejorado
    if show_values:
        for i, v in enumerate(df_plot[colname]):
            formatted_val = f"{v:,.0f}" if v >= 1 else f"{v:.2f}"
            ax.text(i, v, formatted_val, ha="center", va="bottom",
                   fontsize=9, fontweight='600', color='#2c3e50')

    # Leyenda opcional con mejor estilo
    if show_legend:
        ax.legend([colname], loc='upper right', framealpha=0.95,
                 edgecolor='#bdc3c7', fontsize=10)
    else:
        leg = ax.get_legend()
        if leg:
            leg.remove()

    fig.tight_layout()
    return fig, ax

# import plotly.express as px
# import plotly.graph_objects as go

# def graficar_barras(
#     df: pd.DataFrame,
#     xlabel: Optional[Union[str, List[str]]] = None,
#     y: Optional[Union[str, List[str]]] = None,
#     agg: Union[str, Dict[str, str]] = "sum",
#     titulo: str = "Gráfica de Barras",
#     color: Optional[Union[str, List[str]]] = None,
#     *,
#     # Compatibilidad completa con plot_from_params
#     unique_by: Optional[Union[str, List[str]]] = None,
#     conditions_all: Optional[List[List[Any]]] = None,
#     conditions_any: Optional[List[Union[List[Any], List[List[Any]]]]] = None,
#     distinct_on: Optional[str] = None,
#     drop_dupes_before_sum: bool = False,
#     sort: Optional[Dict[str, str]] = None,
#     limit_categories: Optional[int] = None,
#     show_legend: bool = True,
#     show_values: bool = False,
#     **kwargs,   # absorbe cualquier otro parámetro enviado
# ) -> Tuple[plt.Figure, plt.Axes]:

#     # --------------------------------------------
#     # Validación base
#     # --------------------------------------------
#     if not isinstance(df, pd.DataFrame):
#         raise TypeError("El parámetro 'df' debe ser un DataFrame de pandas.")

#     # Unificar X (acepta lista -> crea multiindex si aplica)
#     df2, xlabel_final, _ = _ensure_xlabel(df, xlabel)

#     # --------------------------------------------
#     # Determinar Y
#     # --------------------------------------------
#     if y is None:
#         num_cols = df2.select_dtypes(include=[np.number]).columns.tolist()
#         if not num_cols and not (distinct_on and agg in ("count", "distinct_count")):
#             raise ValueError("No se encontraron columnas numéricas ni 'distinct_on'.")
#         y_cols = [num_cols[0]] if num_cols else []
#     else:
#         if y not in df2.columns:
#             raise ValueError(f"La columna '{y}' no existe en el DataFrame.")
#         y_cols = [y]

#     # --------------------------------------------
#     # Aplicar filtro + deduplicación previa
#     # --------------------------------------------
#     dff = _prefilter_df(
#         df2,
#         unique_by=unique_by,
#         conditions_all=conditions_all,
#         conditions_any=conditions_any,
#     )

#     # --------------------------------------------
#     # Agregar (usa tu agregador unificado)
#     # --------------------------------------------
#     df_plot = _aggregate_frame(
#         dff,
#         xlabel=xlabel_final,
#         y_cols=y_cols,
#         agg=agg,
#         distinct_on=distinct_on,
#         drop_dupes_before_sum=drop_dupes_before_sum,
#     )

#     # Asegurar DataFrame
#     if isinstance(df_plot, pd.Series):
#         df_plot = df_plot.to_frame(name=y_cols[0] if y_cols else "valor")

#     # --------------------------------------------
#     # Sort opcional
#     # --------------------------------------------
#     if sort:
#         by = sort.get("by", "y")
#         order = sort.get("order", "desc")
#         ascending = (order == "asc")

#         if by == "label":
#             df_plot = df_plot.sort_index(ascending=ascending)
#         else:
#             df_plot = df_plot.sort_values(by=df_plot.columns[0], ascending=ascending)

#     # Top-N opcional
#     if limit_categories and limit_categories > 0:
#         df_plot = df_plot.head(limit_categories)

#     # --------------------------------------------
#     # Construir figura
#     # --------------------------------------------
#     fig, ax = plt.subplots(figsize=(10, 6))

#     colname = df_plot.columns[0]
#     ax.bar(df_plot.index.astype(str), df_plot[colname], color=color)

#     ax.set_title(titulo)
#     ax.set_ylabel(colname)
#     ax.set_xlabel(xlabel_final)
#     ax.grid(axis="y", linestyle="--", alpha=0.6)

#     # Rotar etiquetas largas
#     ax.tick_params(axis="x", rotation=35)

#     # Valores sobre las barras
#     if show_values:
#         for i, v in enumerate(df_plot[colname]):
#             ax.text(i, v, f"{v}", ha="center", va="bottom")

#     # Leyenda opcional
#     if show_legend:
#         ax.legend([colname])
#     else:
#         leg = ax.get_legend()
#         if leg:
#             leg.remove()

#     fig.tight_layout()
#     return fig, ax



# import plotly.express as px
# import plotly.graph_objects as go
# import numpy as np
# import pandas as pd

# def graficar_barras(
#     df: pd.DataFrame,
#     xlabel: str | list[str] | None = None,
#     y: str | list[str] | None = None,
#     agg: str | dict[str, str] = "sum",
#     titulo: str = "Gráfica de Barras",
#     color: str | list[str] | None = None,
#     *,
#     unique_by: str | list[str] | None = None,
#     conditions_all: list[list[object]] | None = None,
#     conditions_any: list[list[object] | list[list[object]]] | None = None,
#     max_cats: int = 40,
#     show_legend: bool = True,
#     show_values: bool = False,
# ):
#     if not isinstance(df, pd.DataFrame):
#         raise TypeError("El parámetro 'df' debe ser un DataFrame de pandas.")

#     # --- Selección/validación de Y
#     if y is None:
#         y_cols = df.select_dtypes(include=[np.number]).columns.tolist()
#         if not y_cols:
#             raise ValueError("No se encontraron columnas numéricas en el DataFrame.")
#     elif isinstance(y, str):
#         if y not in df.columns:
#             raise ValueError(f"La columna '{y}' no existe en el DataFrame.")
#         y_cols = [y]
#     else:
#         missing = [col for col in y if col not in df.columns]
#         if missing:
#             raise ValueError(f"Las siguientes columnas no existen en el DataFrame: {missing}")
#         y_cols = list(y)

#     # --- Combinar múltiples X si vienen como lista
#     df2, xlabel_final, _ = _ensure_xlabel(df, xlabel)

#     # --- Filtros y deduplicación previas
#     dff = _prefilter_df(
#         df2,
#         unique_by=unique_by,
#         conditions_all=conditions_all,
#         conditions_any=conditions_any
#     )

#     # --- Agregación
#     df_plot = _aggregate_frame(dff, xlabel_final, y_cols, agg)

#     # --- Recorte de categorías (top-N por primera serie)
#     if xlabel is not None and df_plot.shape[0] > max_cats:
#         first_col = df_plot.columns[0]
#         df_plot = df_plot.sort_values(by=first_col, ascending=False).head(max_cats)

#     # --- Preparación para Plotly
#     df_plot = df_plot.copy()
#     df_plot["__cat__"] = df_plot.index.astype(str)

#     if len(df_plot.columns) == 2:  # solo 1 serie (una col de valores + __cat__)
#         # Detectar automáticamente el nombre de la única serie
#         ycol = [c for c in df_plot.columns if c != "__cat__"][0]
#         fig = px.bar(
#             df_plot,
#             x="__cat__",
#             y=ycol,
#             title=titulo,
#             text=ycol if show_values else None
#         )
#         # Color único (si viene)
#         if isinstance(color, str):
#             fig.update_traces(marker_color=color)
#         elif isinstance(color, (list, tuple)) and color:
#             fig.update_traces(marker_color=color[0])

#     else:
#         # Varias series: pasar a formato largo
#         value_cols = [c for c in df_plot.columns if c != "__cat__"]
#         long_df = df_plot.melt(
#             id_vars="__cat__",
#             value_vars=value_cols,
#             var_name="serie",
#             value_name="valor"
#         )
#         fig = px.bar(
#             long_df,
#             x="__cat__",
#             y="valor",
#             color="serie",
#             barmode="group",
#             title=titulo,
#             text="valor" if show_values else None
#         )
#         # Aplicar paleta si viene como lista
#         if isinstance(color, (list, tuple)) and color:
#             # Asignar color por orden de trazas
#             for i, tr in enumerate(fig.data):
#                 if i < len(color):
#                     tr.marker.color = color[i]

#     # Layout y detalles
#     fig.update_layout(
#         showlegend=show_legend,
#         xaxis_title=(xlabel_final if xlabel_final else "Índice"),
#         yaxis_title=("Valor" if len(y_cols) > 1 else y_cols[0]),
#         uniformtext_minsize=8,
#         uniformtext_mode='hide'
#     )
#     # Orden de categorías por total (queda más parecido a tu Matplotlib)
#     fig.update_xaxes(categoryorder="total descending")

#     if show_values:
#         fig.update_traces(texttemplate="%{text}", textposition="outside", cliponaxis=False)

#     return fig



# def graficar_torta(
#     df: pd.DataFrame,
#     xlabel: Optional[Union[str, List[str]]] = None,
#     y: Optional[str] = None,
#     agg: Union[str, Dict[str, str]] = "sum",
#     titulo: str = "Gráfico de Torta",
#     color: Optional[Union[str, List[str]]] = None,
#     *,
#     # Controles de filtro/deduplicación
#     unique_by: Optional[Union[str, List[str]]] = None,
#     conditions_all: Optional[List[List[Any]]] = None,
#     conditions_any: Optional[List[Union[List[Any], List[List[Any]]]]] = None,
#     # Controles de agregación extendida
#     distinct_on: Optional[str] = None,
#     drop_dupes_before_sum: bool = False,
#     # Orden y top-N (opcionales)
#     sort: Optional[Dict[str, str]] = None,          # {"by": "y"|"label", "order": "asc"|"desc"}
#     limit_categories: Optional[int] = None,
# ) -> Tuple[plt.Figure, plt.Axes]:
#     """
#     Gráfico de torta con:
#       - xlabel simple o múltiple
#       - filtros AND/OR + unique_by
#       - agregaciones extendidas
#       - sort + top-N
#       - etiquetas externas sin superposición (algoritmo propio, sin adjustText)
#     """

#     # ---------------- Validación base ----------------
#     if not isinstance(df, pd.DataFrame):
#         raise TypeError("El parámetro 'df' debe ser un DataFrame de pandas.")

#     # Aceptar múltiples columnas en X → combinar
#     df2, xlabel_final, _ = _ensure_xlabel(df, xlabel)

#     # ---------------- Determinar Y ----------------
#     if y is None:
#         num_cols = df2.select_dtypes(include=[np.number]).columns.tolist()
#         if not num_cols and not (distinct_on and agg in ("count", "distinct_count")):
#             raise ValueError("No se encontraron columnas numéricas ni 'distinct_on' para contar.")
#         y_cols = [num_cols[0]] if num_cols else []
#     else:
#         if y not in df2.columns:
#             raise ValueError(f"La columna '{y}' no existe en el DataFrame.")
#         y_cols = [y]

#     # ---------------- Filtro + deduplicación previa ----------------
#     dff = _prefilter_df(
#         df2,
#         unique_by=unique_by,
#         conditions_all=conditions_all,
#         conditions_any=conditions_any,
#     )

#     # ---------------- Agregación ----------------
#     df_plot = _aggregate_frame(
#         dff,
#         xlabel=xlabel_final,
#         y_cols=y_cols,
#         agg=agg,
#         distinct_on=distinct_on,
#         drop_dupes_before_sum=drop_dupes_before_sum,
#     )

#     # Convertir a serie única
#     if isinstance(df_plot, pd.DataFrame):
#         if df_plot.shape[1] != 1:
#             raise ValueError("El gráfico de torta requiere una única serie numérica después de agregar.")
#         serie = df_plot.iloc[:, 0]
#     else:
#         serie = df_plot

#     # ---------------- Limpieza y agregación por categoría ----------------
#     serie = serie.replace([np.inf, -np.inf], np.nan).dropna()

#     if serie.empty:
#         raise ValueError("No hay datos válidos para graficar (todo es NaN/inf o vacío).")

#     # Importante: si hay índices repetidos, se suman → evita duplicar etiquetas
#     serie = serie.groupby(serie.index).sum()

#     # ---------------- Orden y top-N ----------------
#     if sort:
#         by = sort.get("by", "y")
#         order = sort.get("order", "desc")
#         asc = (order == "asc")
#         if by == "label":
#             serie = serie.sort_index(ascending=asc)
#         else:
#             serie = serie.sort_values(ascending=asc)

#     if limit_categories and limit_categories > 0:
#         serie = serie.head(limit_categories)

#     labels = [str(x) for x in serie.index.tolist()]
#     values = serie.values.tolist()

#     # ---------------- Colores ----------------
#     if not color:
#         color = ["#1976d2", "#ffe066"]  # azul y amarillo por defecto

#     if isinstance(color, str):
#         color = [color]

#     if len(color) < len(values):
#         color = (color * ((len(values) // len(color)) + 1))[:len(values)]

#     # ---------------- Gráfico base ----------------
#     fig, ax = plt.subplots(figsize=(8, 8), facecolor="white")

#     wedges, autotexts = ax.pie(
#         values,
#         labels=None,
#         colors=color,
#         shadow=True,
#         explode=[0.04] * len(values),
#         autopct=lambda pct: f"{pct:.1f}%" if pct > 0 else "",
#         pctdistance=0.68,
#         startangle=140,
#         textprops={"fontsize": 13, "color": "#263238", "fontweight": "bold"},
#     )

#     for w in wedges:
#         w.set_edgecolor("#212121")
#         w.set_linewidth(1.2)

#     # ---------------- Etiquetas externas sin solaparse ----------------
#     # Pre-calculamos ángulos y lado (izquierda / derecha)
#     label_info = []
#     for w, lbl in zip(wedges, labels):
#         ang = 0.5 * (w.theta2 + w.theta1)
#         x = np.cos(np.deg2rad(ang))
#         y = np.sin(np.deg2rad(ang))
#         side = "right" if x >= 0 else "left"
#         label_info.append({"wedge": w, "label": lbl, "x": x, "y": y, "side": side})

#     # Ajuste vertical independiente para cada lado
#     min_delta = 0.12  # distancia mínima entre labels en Y (ajusta si quieres)
#     offset = 1.25     # radio donde se ubican las etiquetas

#     for side in ("left", "right"):
#         items = [d for d in label_info if d["side"] == side]
#         if not items:
#             continue

#         # Ordenar de abajo hacia arriba
#         items.sort(key=lambda d: d["y"])

#         # Empujar en Y para separar
#         for i in range(1, len(items)):
#             if items[i]["y"] - items[i - 1]["y"] < min_delta:
#                 items[i]["y"] = items[i - 1]["y"] + min_delta

#         # Dibujar anotaciones y flechas
#         for d in items:
#             x_dir = 1 if d["side"] == "right" else -1
#             x_text = x_dir * offset
#             y_text = d["y"] * offset

#             ax.annotate(
#                 d["label"],
#                 xy=(d["x"] * 0.98, d["y"] * 0.98),  # borde de la torta
#                 xytext=(x_text, y_text),
#                 ha="left" if d["side"] == "right" else "right",
#                 va="center",
#                 fontsize=14,
#                 fontweight="bold",
#                 color="#263238",
#                 arrowprops=dict(
#                     arrowstyle="-",
#                     color="#212121",
#                     lw=1.1,
#                     shrinkA=0,
#                     shrinkB=0,
#                     connectionstyle="arc3,rad=0.15",
#                 ),
#             )

#     # ---------------- Título ----------------
#     ax.set_title(
#         titulo,
#         fontsize=20,
#         fontweight="bold",
#         pad=30,
#         color="#1565c0",
#     )

#     ax.axis("equal")
#     fig.tight_layout(pad=3.5)

#     return fig, ax

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

    # --- aquí cambiamos la lógica para obtener valores y etiquetas ---
    # Convertir siempre a DataFrame para manipular mejor
    if isinstance(df_plot, pd.Series):
        df_plot = df_plot.to_frame(name="valor")

    if df_plot.shape[1] != 1:
        raise ValueError("El gráfico de torta requiere una única serie numérica después de agregar.")

    # Pasar índice a columna para que las etiquetas vengan de allí
    df_plot = df_plot.reset_index()

    # Columna numérica
    col_val = df_plot.columns[-1]
    # Columna de etiquetas (la primera tras reset_index)
    col_lab = df_plot.columns[0]

    # Serie numérica limpia
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

    # Colores
    if color is None:
        color = plt.cm.Greens(np.linspace(0.4, 0.8, len(serie)))
    elif isinstance(color, str):
        color = [color] * len(serie)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.pie(
        serie.values,
        labels=etiquetas.values,   # 👈 ahora usa la columna categórica, no el índice
        autopct='%1.1f%%',
        startangle=90,
        colors=color,
        shadow=False,
    )
    ax.set_title(titulo)
    ax.axis('equal')
    fig.tight_layout()
    return fig, ax

# def graficar_torta(
#     df: pd.DataFrame,
#     xlabel: Optional[Union[str, List[str]]] = None,
#     y: Optional[str] = None,
#     agg: Union[str, Dict[str, str]] = "sum",
#     titulo: str = "Gráfico de Torta",
#     color: Optional[Union[str, List[str]]] = None,
#     *,
#     # Controles de filtro/deduplicación
#     unique_by: Optional[Union[str, List[str]]] = None,
#     conditions_all: Optional[List[List[Any]]] = None,
#     conditions_any: Optional[List[Union[List[Any], List[List[Any]]]]] = None,
#     # Controles de agregación extendida
#     distinct_on: Optional[str] = None,
#     drop_dupes_before_sum: bool = False,
#     # Orden y top-N (opcionales)
#     sort: Optional[Dict[str, str]] = None,          # {"by":"y"|"label","order":"asc"|"desc"}
#     limit_categories: Optional[int] = None,
# ) -> Tuple[plt.Figure, plt.Axes]:
#     """
#     Pie chart con:
#       - xlabel como str o list[str] (se combina con _ensure_xlabel)
#       - filtros AND/OR y unique_by
#       - agregaciones extendidas (distinct_count, sum_distinct, distinct_on)
#       - sort + top-N opcional
#     Requiere UNA sola serie numérica tras la agregación.
#     """
#     if not isinstance(df, pd.DataFrame):
#         raise TypeError("El parámetro 'df' debe ser un DataFrame de pandas.")

#     # Aceptar múltiples columnas en X → combinar
#     df2, xlabel_final, _ = _ensure_xlabel(df, xlabel)

#     # Determinar y
#     if y is None:
#         num_cols = df2.select_dtypes(include=[np.number]).columns.tolist()
#         if not num_cols and not (distinct_on and agg in ("count", "distinct_count")):
#             raise ValueError("No se encontraron columnas numéricas ni 'distinct_on' para contar.")
#         y_cols = [num_cols[0]] if num_cols else []
#     else:
#         if y not in df2.columns:
#             raise ValueError(f"La columna '{y}' no existe en el DataFrame.")
#         y_cols = [y]

#     # Filtro + deduplicación previa por unique_by
#     dff = _prefilter_df(df2, unique_by=unique_by, conditions_all=conditions_all, conditions_any=conditions_any)

#     # Agregar (usa versión multi-X de _aggregate_frame)
#     df_plot = _aggregate_frame(
#         dff, xlabel=xlabel_final, y_cols=y_cols, agg=agg,
#         distinct_on=distinct_on, drop_dupes_before_sum=drop_dupes_before_sum
#     )

#     # Asegurar serie única y consolidar etiquetas repetidas
#     if isinstance(df_plot, pd.DataFrame):
#         if df_plot.shape[1] != 1:
#             raise ValueError("El gráfico de torta requiere una única serie numérica después de agregar.")
#         serie = df_plot.iloc[:, 0]
#     else:
#         serie = df_plot

#     # 🔥 Paso esencial: evitar etiquetas duplicadas agrupando por índice
#     serie = serie.groupby(serie.index).sum()

#     # limpiar y ordenar opcionalmente
#     serie = serie.replace([np.inf, -np.inf], np.nan).dropna()
#     if serie.empty:
#         raise ValueError("No hay datos válidos para graficar (todo es NaN/inf o vacío).")

#     # sort / top-N
#     if sort:
#         by = sort.get("by", "y")
#         order = sort.get("order", "desc")
#         ascending = (order == "asc")
#         if by == "label":
#             serie = serie.sort_index(ascending=ascending)
#         else:
#             serie = serie.sort_values(ascending=ascending)
#     if limit_categories and limit_categories > 0:
#         serie = serie.head(limit_categories)

#     total = serie.clip(lower=0).sum()
#     if total <= 0:
#         raise ValueError("La suma de valores es 0; no es posible construir la torta.")

#     # plot
#     fig, ax = plt.subplots(figsize=(8, 8))
#     ax.pie(serie.values, labels=serie.index.astype(str), autopct='%1.1f%%', startangle=90, colors=color, shadow=False)
#     ax.set_title(titulo)
#     ax.axis('equal')
#     fig.tight_layout()
#     return fig, ax

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
    show_values: bool = False,
    **kwargs,   # <<< IMPORTANTE: absorbe parámetros extra
) -> Tuple[plt.Figure, plt.Axes]:

    # Validación base
    if not isinstance(df, pd.DataFrame):
        raise TypeError("El parámetro 'df' debe ser un DataFrame de pandas.")

    # Combinar X
    df2, xlabel_final, _ = _ensure_xlabel(df, xlabel)

    # Determinar columnas numéricas
    if y is None:
        num_cols = df2.select_dtypes(include=[np.number]).columns.tolist()
        if not num_cols and not (distinct_on and agg in ("count", "distinct_count")):
            raise ValueError("No se encontraron columnas numéricas ni 'distinct_on'.")
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

    # Agregación (usa tu agregador unificado)
    df_plot = _aggregate_frame(
        dff,
        xlabel=xlabel_final,
        y_cols=y_cols,
        agg=agg,
        distinct_on=distinct_on,
        drop_dupes_before_sum=drop_dupes_before_sum,
    )

    # Asegurar objeto gráfico
    if isinstance(df_plot, pd.Series):
        df_plot = df_plot.to_frame(name=y_cols[0] if y_cols else "valor")

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

    # Construir barra horizontal con mejor estilo
    colname = df_plot.columns[0]
    
    # Color profesional si no se especifica
    if color is None:
        color = '#607D8B'  # Azul grisáceo corporativo
    
    bars = ax.barh(df_plot.index.astype(str), df_plot[colname], 
                   color=color, edgecolor='white', linewidth=1.2, alpha=0.9)
    
    for bar in bars:
        bar.set_zorder(3)

    # Títulos y etiquetas mejoradas
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
        for i, v in enumerate(df_plot[colname]):
            formatted_val = f"{v:,.0f}" if v >= 1 else f"{v:.2f}"
            ax.text(v + max(df_plot[colname]) * 0.01, i, formatted_val, 
                   va='center', ha='left', fontsize=9, fontweight='600', color='#2c3e50')

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

# def graficar_barras_horizontal(
#     df: pd.DataFrame,
#     xlabel: Optional[Union[str, List[str]]] = None,
#     y: Optional[Union[str, List[str]]] = None,
#     agg: Union[str, Dict[str, str]] = "sum",
#     titulo: str = "Gráfica de Barras Horizontal",
#     color: Optional[Union[str, List[str]]] = None,
#     *,
#     # Controles adicionales (compatibilidad total)
#     unique_by: Optional[Union[str, List[str]]] = None,
#     conditions_all: Optional[List[List[Any]]] = None,
#     conditions_any: Optional[List[Union[List[Any], List[List[Any]]]]] = None,
#     distinct_on: Optional[str] = None,
#     drop_dupes_before_sum: bool = False,
#     sort: Optional[Dict[str, str]] = None,          # {"by": "y"|"label", "order": "asc"|"desc"}
#     limit_categories: Optional[int] = None,
#     show_legend: bool = True,
#     show_values: bool = False,
#     **kwargs,   # <<< IMPORTANTE: absorbe parámetros extra
# ) -> Tuple[plt.Figure, plt.Axes]:

#     # Validación base
#     if not isinstance(df, pd.DataFrame):
#         raise TypeError("El parámetro 'df' debe ser un DataFrame de pandas.")

#     # Combinar X
#     df2, xlabel_final, _ = _ensure_xlabel(df, xlabel)

#     # Determinar columnas numéricas
#     if y is None:
#         num_cols = df2.select_dtypes(include=[np.number]).columns.tolist()
#         if not num_cols and not (distinct_on and agg in ("count", "distinct_count")):
#             raise ValueError("No se encontraron columnas numéricas ni 'distinct_on'.")
#         y_cols = [num_cols[0]] if num_cols else []
#     else:
#         if y not in df2.columns:
#             raise ValueError(f"La columna '{y}' no existe en el DataFrame.")
#         y_cols = [y]

#     # Filtro + deduplicación previa
#     dff = _prefilter_df(
#         df2,
#         unique_by=unique_by,
#         conditions_all=conditions_all,
#         conditions_any=conditions_any,
#     )

#     # Agregación (usa tu agregador unificado)
#     df_plot = _aggregate_frame(
#         dff,
#         xlabel=xlabel_final,
#         y_cols=y_cols,
#         agg=agg,
#         distinct_on=distinct_on,
#         drop_dupes_before_sum=drop_dupes_before_sum,
#     )

#     # Asegurar objeto gráfico
#     if isinstance(df_plot, pd.Series):
#         df_plot = df_plot.to_frame(name=y_cols[0] if y_cols else "valor")

#     # Ordenamiento opcional
#     if sort:
#         by = sort.get("by", "y")
#         order = sort.get("order", "desc")
#         ascending = (order == "asc")

#         if by == "label":
#             df_plot = df_plot.sort_index(ascending=ascending)
#         else:
#             df_plot = df_plot.sort_values(by=df_plot.columns[0], ascending=ascending)

#     # Top-N opcional
#     if limit_categories and limit_categories > 0:
#         df_plot = df_plot.head(limit_categories)

#     # Crear figura
#     fig, ax = plt.subplots(figsize=(10, 6))

#     # Construir barra horizontal
#     colname = df_plot.columns[0]
#     ax.barh(df_plot.index.astype(str), df_plot[colname], color=color)

#     ax.set_title(titulo)
#     ax.set_xlabel(colname)
#     ax.set_ylabel(xlabel_final)
#     ax.grid(axis="x", linestyle="--", alpha=0.6)

#     # Mostrar valores
#     if show_values:
#         for i, v in enumerate(df_plot[colname]):
#             ax.text(v, i, f"{v}", va='center', ha='left')

#     # Leyenda
#     if show_legend:
#         ax.legend([colname])
#     else:
#         leg = ax.get_legend()
#         if leg:
#             leg.remove()

#     fig.tight_layout()
#     return fig, ax


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
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Tabla con:
      - xlabel como str o list[str] (se combina con _ensure_xlabel)
      - filtros AND/OR, unique_by, distinct_on y agregaciones extendidas
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
    dff = _prefilter_df(df2, unique_by=unique_by, conditions_all=conditions_all, conditions_any=conditions_any)

    # Agregar (respeta distinct_count / sum_distinct / distinct_on)
    df_plot = _aggregate_frame(
        dff, xlabel=xlabel_final, y_cols=y_cols, agg=agg,
        distinct_on=distinct_on, drop_dupes_before_sum=drop_dupes_before_sum, where=where
    )

    # Reset index para mostrar categoría(s) como columna(s)
    df_plot = df_plot.reset_index() if xlabel_final is not None else df_plot.reset_index(drop=True)

    # Formateo
    df_display = df_plot.copy()
    for col in df_display.columns:
        if is_numeric_dtype(df_display[col]):
            df_display[col] = df_display[col].round(2)
        elif is_datetime64_any_dtype(df_display[col]):
            df_display[col] = df_display[col].dt.strftime("%Y-%m-%d")
    df_display = df_display.fillna("").astype(str)

    # Formatear números con separador de miles
    df_formatted = df_display.copy()
    for col in df_display.columns:
        try:
            # Intentar formatear como número
            numeric_col = pd.to_numeric(df_display[col], errors='coerce')
            if numeric_col.notna().any():
                df_formatted[col] = numeric_col.apply(
                    lambda x: f"{x:,.0f}" if pd.notna(x) and x >= 1 else 
                             (f"{x:.2f}" if pd.notna(x) else "")
                )
        except:
            pass
    
    # Figura con diseño profesional y más grande
    n_cols = len(df_formatted.columns)
    n_rows = len(df_formatted)
    fig_w = max(18, n_cols * 4.5)
    fig_h = max(9, n_rows * 2.2 + 6)

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

    # Color solo para encabezado, filas blancas
    if color and "," in color:
        color_list = [c.strip() for c in color.split(",") if c.strip()]
    elif color:
        color_list = [color]
    else:
        color_list = ["#2E86AB"]

    header_bg = "#1565c0"  # azul más oscuro para encabezado

    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap
    grad_cmap = LinearSegmentedColormap.from_list("excel_blue_grad", ["#1565c0", "#42a5f5"])
    grad_img = np.linspace(0, 1, 256).reshape(1, -1)
    row_colors = ['#e3f2fd', '#ffffff']  # Azul claro y blanco alternados
    # Alternancia de colores en filas de datos según la lista de colores
    from matplotlib.colors import to_rgb, to_hex
    def aclarar_color(color, factor=0.7):
        rgb = to_rgb(color)
        aclarado = tuple(1 - (1 - c) * factor for c in rgb)
        return to_hex(aclarado)
    def oscurecer_color(color, factor=0.7):
        rgb = to_rgb(color)
        oscurecido = tuple(c * factor for c in rgb)
        return to_hex(oscurecido)

    # El primer color es para encabezado, los demás para datos
    if len(color_list) > 1:
        header_bg = oscurecer_color(color_list[0], 0.7)
        data_colors = [aclarar_color(c, 0.85) for c in color_list[1:]]
    else:
        header_bg = oscurecer_color(color_list[0] if color_list else "#1976d2", 0.7)
        data_colors = [aclarar_color(header_bg, 0.85), aclarar_color(header_bg, 0.93)]
    n_rows = len(df_formatted)
    for (row, col), cell in tabla.get_celld().items():
        if row == 0:
            # Si el fondo es blanco, texto negro; si no, texto blanco
            text_color = "#212121" if header_bg.lower() in ["#fff", "#ffffff", "white"] else "white"
            cell.set_text_props(weight="bold", color=text_color, fontsize=36)
            cell.set_facecolor(header_bg)
        else:
            valor = cell.get_text().get_text()
            try:
                float(valor.replace(',', ''))
                cell.set_text_props(color="#212121", fontsize=32, weight="bold")
            except:
                cell.set_text_props(color="#2c3e50", fontsize=28, weight="normal")
            # Alternar colores aclarados para filas de datos
            color_idx = (row - 1) % len(data_colors)
            cell.set_facecolor(data_colors[color_idx])
        cell.set_edgecolor('#424242')
        cell.set_linewidth(1.5)
        cell.PAD = 0.42

    # Título profesional
    ax.set_title(titulo, fontsize=38, fontweight="bold", pad=60, color="#000000")
    fig.tight_layout(pad=6.5)
    return fig, ax
# def graficar_tabla(
#     df: pd.DataFrame,
#     xlabel: Optional[Union[str, List[str]]] = None,
#     y: Optional[Union[str, List[str]]] = None,
#     agg: Union[str, Dict[str, str]] = "sum",
#     titulo: str = "Tabla de Datos",
#     color: Optional[str] = None,
#     *,
#     # Filtros / deduplicación
#     unique_by: Optional[Union[str, List[str]]] = None,
#     conditions_all: Optional[List[List[Any]]] = None,
#     conditions_any: Optional[List[Union[List[Any], List[List[Any]]]]] = None,
#     # Agregación extendida
#     distinct_on: Optional[str] = None,
#     drop_dupes_before_sum: bool = False,
#     where: Optional[pd.Series] = None,
# ) -> Tuple[plt.Figure, plt.Axes]:
#     """
#     Tabla con:
#       - xlabel como str o list[str] (se combina con _ensure_xlabel)
#       - filtros AND/OR, unique_by, distinct_on y agregaciones extendidas
#     """
#     if not isinstance(df, pd.DataFrame):
#         raise TypeError("El parámetro 'df' debe ser un DataFrame de pandas.")

#     # Aceptar multi-X → combinar etiquetas
#     df2, xlabel_final, _ = _ensure_xlabel(df, xlabel)

#     # y_cols
#     if y is None:
#         y_cols = df2.select_dtypes(include=[np.number]).columns.tolist()
#         # permitir tablas con conteo único vía distinct_on aun si no hay numéricas
#         if not y_cols and not (distinct_on and isinstance(agg, str) and agg in ("count", "distinct_count")):
#             raise ValueError("No se encontraron columnas numéricas para la tabla.")
#     elif isinstance(y, str):
#         if y not in df2.columns:
#             raise ValueError(f"La columna '{y}' no existe en el DataFrame.")
#         y_cols = [y]
#     else:
#         missing = [col for col in y if col not in df2.columns]
#         if missing:
#             raise ValueError(f"Las siguientes columnas no existen en el DataFrame: {missing}")
#         y_cols = y

#     # Filtro y deduplicación previa
#     dff = _prefilter_df(df2, unique_by=unique_by, conditions_all=conditions_all, conditions_any=conditions_any)

#     # Agregar (respeta distinct_count / sum_distinct / distinct_on)
#     df_plot = _aggregate_frame(
#         dff, xlabel=xlabel_final, y_cols=y_cols, agg=agg,
#         distinct_on=distinct_on, drop_dupes_before_sum=drop_dupes_before_sum, where=where
#     )

#     # Reset index para mostrar categoría(s) como columna(s)
#     df_plot = df_plot.reset_index() if xlabel_final is not None else df_plot.reset_index(drop=True)

#     # Formateo
#     df_display = df_plot.copy()
#     for col in df_display.columns:
#         if is_numeric_dtype(df_display[col]):
#             df_display[col] = df_display[col].round(2)
#         elif is_datetime64_any_dtype(df_display[col]):
#             df_display[col] = df_display[col].dt.strftime("%Y-%m-%d")
#     df_display = df_display.fillna("").astype(str)

#     # Figura
#     n_cols = len(df_display.columns)
#     n_rows = len(df_display)
#     fig_w = max(6, n_cols * 2.5)
#     fig_h = max(2.5, n_rows * 0.5 + 2)

#     fig, ax = plt.subplots(figsize=(fig_w, fig_h))
#     ax.axis("off")
#     tabla = ax.table(
#         cellText=df_display.values.tolist(),
#         colLabels=df_display.columns.tolist(),
#         cellLoc="center",
#         loc="center"
#     )
#     tabla.auto_set_font_size(False)
#     tabla.set_fontsize(10)
#     tabla.scale(1.1, 1.2)
#     for col_idx in range(n_cols):
#         tabla.auto_set_column_width(col=col_idx)

#     header_bg = color if color else "#25347a"
#     for (row, col), cell in tabla.get_celld().items():
#         if row == 0:
#             cell.set_text_props(weight="bold", color="white")
#             cell.set_facecolor(header_bg)
#         else:
#             cell.set_facecolor("#f9f9f9" if (row % 2 == 1) else "white")

#     ax.set_title(titulo, fontsize=13, fontweight="bold", pad=20)
#     fig.tight_layout()
#     return fig, ax
# import pandas as pd
# from pandas.api.types import is_numeric_dtype, is_datetime64_any_dtype
# from typing import Optional, List, Any, Union, Tuple
# import matplotlib.pyplot as plt
# from PIL import Image, ImageDraw, ImageFont
# from io import BytesIO
# from PIL import Image, ImageDraw, ImageFont
# from typing import Optional, Union, List, Any, Dict, Tuple
# import pandas as pd
# import matplotlib.pyplot as plt
# from pandas.api.types import is_numeric_dtype

# def graficar_tabla(
#     df: pd.DataFrame,
#     xlabel: Optional[Union[str, List[str]]] = None,
#     y: Optional[Union[str, List[str]]] = None,
#     agg: Union[str, Dict[str, str]] = "sum",
#     titulo: str = "Tabla de Datos",
#     color: str = "#25347a",
#     *,
#     unique_by: Optional[Union[str, List[str]]] = None,
#     conditions_all: Optional[List[List[Any]]] = None,
#     conditions_any: Optional[List[Union[List[Any], List[List[Any]]]]] = None,
#     distinct_on: Optional[str] = None,
#     drop_dupes_before_sum: bool = False,
#     where: Optional[pd.Series] = None,
# ) -> Tuple[plt.Figure, plt.Axes]:
#     """
#     Tabla renderizada como imagen, compatible con _fig_to_data_uri.
#     Mantiene compatibilidad con firma original para plot_from_params.
#     """

#     # Helper compatible con Pillow >=10
#     def get_text_size(font, text):
#         if hasattr(font, "getbbox"):
#             bbox = font.getbbox(text)
#             return bbox[2] - bbox[0], bbox[3] - bbox[1]
#         else:
#             return font.getsize(text)

#     # Validaciones y preprocesamiento
#     df2, xlabel_final, _ = _ensure_xlabel(df, xlabel)

#     # Determinar y
#     if y is None:
#         y_cols = df2.select_dtypes(include=[np.number]).columns.tolist()
#     elif isinstance(y, str):
#         y_cols = [y]
#     else:
#         y_cols = y

#     # Filtros (igual que en la función original)
#     dff = _prefilter_df(df2, unique_by=unique_by, conditions_all=conditions_all, conditions_any=conditions_any)

#     # Agregar
#     df_plot = _aggregate_frame(
#         dff,
#         xlabel=xlabel_final,
#         y_cols=y_cols,
#         agg=agg,
#         distinct_on=distinct_on,
#         drop_dupes_before_sum=drop_dupes_before_sum,
#         where=where
#     )
#     df_plot = df_plot.reset_index() if xlabel_final is not None else df_plot.reset_index(drop=True)

#     # Formateo
#     for col in df_plot.columns:
#         if is_numeric_dtype(df_plot[col]):
#             df_plot[col] = df_plot[col].round(2)
#     df_plot = df_plot.fillna("").astype(str)

#     # Renderizar tabla en imagen
#     cell_padding = 10
#     font = ImageFont.load_default()

#     # Calcular tamaños de columnas
#     col_widths = []
#     for col in df_plot.columns:
#         max_width = max(
#             get_text_size(font, str(col))[0],
#             *(get_text_size(font, str(v))[0] for v in df_plot[col])
#         )
#         col_widths.append(max_width + 2 * cell_padding)

#     # Altura de filas
#     row_height = get_text_size(font, "Ay")[1] + 2 * cell_padding

#     # Dimensiones finales de tabla
#     table_width = sum(col_widths)
#     table_height = row_height * (len(df_plot) + 1)  # +1 header row

#     # Crear imagen
#     img = Image.new("RGB", (table_width, table_height), "white")
#     draw = ImageDraw.Draw(img)

#     # Pintar encabezado
#     x_offset = 0
#     for i, col in enumerate(df_plot.columns):
#         draw.rectangle([x_offset, 0, x_offset + col_widths[i], row_height], fill=color)
#         draw.text((x_offset + cell_padding, cell_padding), str(col), font=font, fill="white")
#         x_offset += col_widths[i]

#     # Pintar filas
#     for row_num, (_, row) in enumerate(df_plot.iterrows(), start=1):
#         x_offset = 0
#         for i, val in enumerate(row):
#             fill_color = "#f9f9f9" if row_num % 2 else "white"
#             draw.rectangle([x_offset, row_num * row_height, x_offset + col_widths[i], (row_num + 1) * row_height], fill=fill_color)
#             draw.text((x_offset + cell_padding, row_num * row_height + cell_padding), str(val), font=font, fill="black")
#             x_offset += col_widths[i]

#     # Convertir imagen en figura matplotlib
#     fig, ax = plt.subplots(figsize=(table_width / 100, table_height / 100), dpi=100)
#     ax.axis("off")
#     ax.imshow(img)

#     return fig, ax




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

# def _stack_columns(df: pd.DataFrame,
#                    columns: list[str],
#                    *,
#                    output_col: str = "tipo_riesgo",
#                    value_col: str = "valor",
#                    label_map: dict[str,str] | None = None) -> pd.DataFrame:
#     m = df.melt(value_vars=columns, var_name=output_col, value_name=value_col)
#     if label_map:
#         m[output_col] = m[output_col].map(lambda x: label_map.get(x, x))
#     return m


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
    titulo  = p.get("title", "Gráfico")
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

    # --- Normalización mínima de torta ---
    if fn_key in ("graficar_torta", "torta", "pie") and isinstance(y, list):
        y = y[0] if y else None
        common_kwargs["y"] = y

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
        labels_raw = None
        if xlabel and xlabel in df_filtered.columns:
            labels_raw = df_filtered[xlabel].astype(str).tolist()
        else:
            labels_raw = [txt.get_text() for txt in ax.texts if "%" not in txt.get_text()]
        labels_fmt = _format_pie_labels(labels_raw, max_chars=pie_max_chars, wrap_width=pie_wrap_w)
        _apply_pie_texts(ax, labels_fmt)

    _finalize_layout(fig, ax, legend_outside=legend_out)

    if show:
        plt.tight_layout()
        plt.show()

    return fig, ax


# @register("plt_from_params")
# def plot_from_params(df, params, *, show: bool=False):
#     import copy
#     import matplotlib.pyplot as plt

#     p = copy.deepcopy(params)

#     # --- Normalización de selector de función ---
#     fn_raw = p.get("function_name") or p.get("chart_type")
#     if fn_raw is None:
#         raise ValueError("Falta 'function_name' o 'chart_type' en params.")

#     fn_key = str(fn_raw).strip().lower()

#     DISPATCH = {
#         # Gráficas por nombre “canónico”
#         "graficar_barras": graficar_barras,
#         "graficar_barras_horizontal": graficar_barras_horizontal,
#         "graficar_torta": graficar_torta,
#         "graficar_tabla": graficar_tabla,
#         # Aliases por chart_type o variantes comunes
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
#         # ayuda de debug: imprime qué llegó exactamente
#         raise ValueError(f"Función de gráfico no reconocida: {fn_raw!r} (normalizado: {fn_key!r})")

#     # --- Parámetros comunes ---
#     xlabel  = p.get("xlabel")
#     y       = p.get("y")
#     agg     = p.get("agg", "sum")
#     titulo  = p.get("title", "Gráfico")
#     color   = p.get("color")
#     tipo    = p.get("tipo")

#         # controles de formateo con defaults
#     tick_fontsize = int(p.get("tick_fontsize", 9))
#     rotation      = p.get("rotation", 35)  # solo para eje X en barras verticales
#     wrap_width_x  = int(p.get("wrap_width_x", 20))
#     wrap_width_y  = int(p.get("wrap_width_y", 30))
#     max_chars_x   = int(p.get("max_chars_x", 85))
#     max_chars_y   = int(p.get("max_chars_y", 120))
#     legend_out    = bool(p.get("legend_outside", True))

#     # formateo de torta
#     pie_max_chars = int(p.get("pie_max_chars", 60))
#     pie_wrap_w    = int(p.get("pie_wrap_width", 25))


#     # --- BINNING ---
#     binning = p.get("binning")
#     if binning:
#         df, bucket_col = _apply_binning(df, binning)
#         xlabel = bucket_col  # forzamos el xlabel al resultado del binning

#     # --- STACK ---
#     stack = p.get("stack_columns")
#     if stack:
#         cols      = stack["columns"]
#         out_col   = stack.get("output_col", "tipo_riesgo")
#         val_col   = stack.get("value_col", "valor")
#         label_map = stack.get("label_map")
#         keep_val  = stack.get("keep_value")
#         df = _stack_columns(df, cols, output_col=out_col, value_col=val_col, label_map=label_map)
#         # keep_value: si viene "any" o None → no filtrar
#         if keep_val not in (None, "any", "", []):
#             df = df[df[val_col] == keep_val]
#         xlabel = out_col

#     # --- Normalizaciones mínimas (compat) ---
#     if fn_key in ("graficar_torta", "torta", "pie") and isinstance(y, list):
#         y = y[0] if y else None

#     # --- DEBUG opcional ---
#     # print(f"[plot_from_params] Ejecutando: {fn_key} -> {func.__name__} | title={titulo}")

#     # --- Llamada dinámica ---
#     if func is graficar_barras:
#         fig, ax = func(df, xlabel=xlabel, y=y, agg=agg, titulo=titulo, color=color)
#     elif func is graficar_barras_horizontal:
#         fig, ax = func(df, xlabel=xlabel, y=y, agg=agg, titulo=titulo, color=color)
#     elif func is graficar_torta:
#         fig, ax = func(df, xlabel=xlabel, y=y, agg=agg, titulo=titulo, color=color)
#     elif func is graficar_tabla:
#         fig, ax = func(df, xlabel=xlabel, y=y, agg=agg, titulo=titulo, color=color)
#     else:
#         # por si agregas nuevas funciones en el DISPATCH, delega con kwargs genéricos
#         fig, ax = func(df, xlabel=xlabel, y=y, agg=agg, titulo=titulo, color=color)


# # --- post-formateo según tipo
#     # --- Derivar 't' si 'tipo' no viene en params
#     if not tipo:
#         if func is graficar_barras:
#             tipo = "bar"
#         elif func is graficar_barras_horizontal:
#             tipo = "barh"
#         elif func is graficar_torta:
#             tipo = "pie"
#         elif func is graficar_tabla:
#             tipo = "tabla"
#     t = (tipo or "").lower()


#     # Barras verticales
#     if t in ("barras", "bar", "column", "col"):
#         _wrap_shorten_ticks(
#             ax,
#             axis="x",
#             wrap_width=wrap_width_x,
#             max_chars=max_chars_x,
#             fontsize=tick_fontsize,
#             rotation=rotation
#         )

#     # Barras horizontales
#     elif t in ("barh", "barras_h", "horizontal", "barra_horizontal"):
#         _wrap_shorten_ticks(
#             ax,
#             axis="y",
#             wrap_width=wrap_width_y,
#             max_chars=max_chars_y,
#             fontsize=tick_fontsize,
#             rotation=None
#         )

#     # Tortas
#     elif t in ("torta", "pie", "pastel"):
#         labels_raw = None
#         if xlabel and xlabel in df.columns:
#             labels_raw = df[xlabel].astype(str).tolist()
#         if labels_raw is None:
#             labels_raw = [txt.get_text() for txt in ax.texts if "%" not in txt.get_text()]
#         if labels_raw:
#             labels_fmt = _format_pie_labels(labels_raw, max_chars=pie_max_chars, wrap_width=pie_wrap_w)
#             _apply_pie_texts(ax, labels_fmt)
#             if ax.get_legend() is None and p.get("pie_force_legend", False):
#                 ax.legend(labels_fmt, loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))

#     # Ajuste general de layout y leyenda (sin cambiar estilo)
#     _finalize_layout(fig, ax, legend_outside=legend_out)

#     if show:
#         plt.tight_layout()
#         plt.show()

#     return fig, ax
