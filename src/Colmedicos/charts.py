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

def _normalize_name(s: str) -> str:
    if s is None:
        return ""
    s2 = "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")
    return s2.strip().lower().replace(" ", "_")

def _to_numeric_like(series: pd.Series) -> pd.Series:
    """Convierte a numérico donde aplique, preservando no-numéricos como NaN."""
    if np.issubdtype(series.dtype, np.number):
        return series
    return pd.to_numeric(series, errors="coerce")

def _coerce_condition_value(series: pd.Series, op: str, val: Any):
    """Intenta convertir el valor de condición al tipo compatible con la serie."""
    if op in ("in", "not in"):
        if not isinstance(val, (list, tuple, set)):
            val = [val]
        if np.issubdtype(series.dtype, np.number):
            return [pd.to_numeric(x, errors="coerce") for x in val]
        return list(val)
    else:
        if np.issubdtype(series.dtype, np.number):
            try:
                return pd.to_numeric(val, errors="coerce")
            except Exception:
                return val
        return val

_OPS = {
    '>':  lambda a, b: a > b,
    '<':  lambda a, b: a < b,
    '==': lambda a, b: a == b,
    '!=': lambda a, b: a != b,
    '>=': lambda a, b: a >= b,
    '<=': lambda a, b: a <= b,
    'in': lambda a, b: a.isin(b),
    'not in': lambda a, b: ~a.isin(b),
}

def _mask_from_conditions(df: pd.DataFrame, conds: list[list]) -> pd.Series:
    """AND de condiciones simples: [[col, op, val], ...]."""
    if not conds:
        return pd.Series(True, index=df.index)
    m = pd.Series(True, index=df.index)
    for col, op, val in conds:
        if col not in df.columns:
            raise ValueError(f"Columna en condición no existe: {col}")
        if op not in _OPS:
            raise ValueError(f"Operador no soportado: {op}")
        # coerción suave numérica si aplica
        s = df[col]
        if pd.api.types.is_numeric_dtype(s):
            try:
                val = pd.to_numeric(val, errors="coerce")
            except Exception:
                pass
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

def _to_numeric_like(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def _ensure_xlabel(
    df: pd.DataFrame,
    xlabel: Optional[Union[str, List[str]]],
    *,
    joiner: str = " | ",
    out_col: str = "__xlabel_combo"
) -> Tuple[pd.DataFrame, Optional[str], Optional[List[str]]]:
    """
    Acepta xlabel como str o list[str].
    - Si list[str], crea una columna combinada (string) con joiner.
    - Devuelve (df_mod, xlabel_final, group_keys)
      * xlabel_final: nombre de columna categórica a usar en el gráfico
      * group_keys: lista original de columnas si se quiere un groupby multinivel en _aggregate_frame
    """
    if xlabel is None or isinstance(xlabel, str):
        return df, xlabel, None

    # validar todas existen
    missing = [c for c in xlabel if c not in df.columns]
    if missing:
        raise ValueError(f"Las siguientes columnas de 'xlabel' no existen: {missing}")

    df2 = df.copy()
    df2[out_col] = (
        df2[xlabel]
        .astype(str)
        .agg(joiner.join, axis=1)
        .replace("nan", "")  # limpieza suave si hubiera NaN
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

def graficar_barras(
    df: pd.DataFrame,
    xlabel: Optional[Union[str, List[str]]] = None,
    y: Optional[Union[str, List[str]]] = None,
    agg: Union[str, Dict[str, str]] = "sum",
    titulo: str = "Gráfica de Barras",
    color: Optional[Union[str, List[str]]] = None,
    *,
    unique_by: Optional[Union[str, List[str]]] = None,
    conditions_all: Optional[List[List[Any]]] = None,
    conditions_any: Optional[List[Union[List[Any], List[List[Any]]]]] = None,
    max_cats: int = 40,
    show_legend: bool = True,
    show_values: bool = False,
) -> Tuple[plt.Figure, plt.Axes]:
    """Barras con soporte de filtros AND/OR y deduplicación por unique_by."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("El parámetro 'df' debe ser un DataFrame de pandas.")

    # y_cols
    if y is None:
        y_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not y_cols:
            raise ValueError("No se encontraron columnas numéricas en el DataFrame.")
    elif isinstance(y, str):
        if y not in df.columns:
            raise ValueError(f"La columna '{y}' no existe en el DataFrame.")
        y_cols = [y]
    else:
        missing = [col for col in y if col not in df.columns]
        if missing:
            raise ValueError(f"Las siguientes columnas no existen en el DataFrame: {missing}")
        y_cols = y

     # 0) combinar múltiples X si vienen como lista
    df2, xlabel_final, _ = _ensure_xlabel(df, xlabel)
    dff = _prefilter_df(df2, unique_by=unique_by, conditions_all=conditions_all, conditions_any=conditions_any)

     # Agregar
    df_plot = _aggregate_frame(dff, xlabel_final, y_cols, agg)

    # recorte de categorías
    if xlabel is not None and df_plot.shape[0] > max_cats:
        # ordena por la primera columna para tomar top-N
        first_col = df_plot.columns[0]
        df_plot = df_plot.sort_values(by=first_col, ascending=False).head(max_cats)

    fig, ax = plt.subplots(figsize=(10, 5))
    if df_plot.shape[1] == 1:
        colname = df_plot.columns[0]
        ax.bar(df_plot.index.astype(str), df_plot[colname], color=color)
        ax.set_ylabel(colname)
        if show_legend:
            ax.legend([colname])
    else:
        df_plot.plot(kind="bar", ax=ax, color=color)
        ax.set_ylabel("Valor")
        if show_legend:
            ax.legend(title="Columnas")
        else:
            leg = ax.get_legend()
            if leg: leg.remove()

    ax.set_title(titulo)
    ax.set_xlabel(xlabel_final if xlabel_final else "Índice")
    for label in ax.get_xticklabels():
        label.set_rotation(45)

    if show_values:
        _annotate_bars(ax)

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
    # Controles de filtro/deduplicación
    unique_by: Optional[Union[str, List[str]]] = None,
    conditions_all: Optional[List[List[Any]]] = None,
    conditions_any: Optional[List[Union[List[Any], List[List[Any]]]]] = None,
    # Controles de agregación extendida
    distinct_on: Optional[str] = None,
    drop_dupes_before_sum: bool = False,
    # Orden y top-N (opcionales)
    sort: Optional[Dict[str, str]] = None,          # {"by":"y"|"label","order":"asc"|"desc"}
    limit_categories: Optional[int] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Pie chart con:
      - xlabel como str o list[str] (se combina con _ensure_xlabel)
      - filtros AND/OR y unique_by
      - agregaciones extendidas (distinct_count, sum_distinct, distinct_on)
      - sort + top-N opcional
    Requiere UNA sola serie numérica tras la agregación.
    """
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

    # Filtro + deduplicación previa por unique_by
    dff = _prefilter_df(df2, unique_by=unique_by, conditions_all=conditions_all, conditions_any=conditions_any)

    # Agregar (usa versión multi-X de _aggregate_frame)
    df_plot = _aggregate_frame(
        dff, xlabel=xlabel_final, y_cols=y_cols, agg=agg,
        distinct_on=distinct_on, drop_dupes_before_sum=drop_dupes_before_sum
    )

    # Asegurar serie única
    if isinstance(df_plot, pd.DataFrame):
        if df_plot.shape[1] != 1:
            raise ValueError("El gráfico de torta requiere una única serie numérica después de agregar.")
        serie = df_plot.iloc[:, 0]
    else:
        serie = df_plot

    # limpiar y ordenar opcionalmente
    serie = serie.replace([np.inf, -np.inf], np.nan).dropna()
    if serie.empty:
        raise ValueError("No hay datos válidos para graficar (todo es NaN/inf o vacío).")

    # sort / top-N
    if sort:
        by = sort.get("by", "y")
        order = sort.get("order", "desc")
        ascending = (order == "asc")
        if by == "label":
            serie = serie.sort_index(ascending=ascending)
        else:
            serie = serie.sort_values(ascending=ascending)
    if limit_categories and limit_categories > 0:
        serie = serie.head(limit_categories)

    total = serie.clip(lower=0).sum()
    if total <= 0:
        raise ValueError("La suma de valores es 0; no es posible construir la torta.")

    # plot
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.pie(serie.values, labels=serie.index.astype(str), autopct='%1.1f%%', startangle=90, colors=color, shadow=False)
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
    distinct_on: str | None = None,
    drop_dupes_before_sum: bool = False,
    where: pd.Series | None = None,
    show: bool = False,
    show_legend: bool = True,
    show_values: bool = False,
):
    # --- Validaciones previas (igual que antes) ---
    if not isinstance(df, pd.DataFrame):
        raise TypeError("El parámetro 'df' debe ser un DataFrame de pandas.")
    if xlabel is not None and xlabel not in df.columns:
        raise ValueError(f"La columna '{xlabel}' no existe en el DataFrame.")

    # Determinar columnas numéricas (igual que antes)
    if y is None:
        y_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not y_cols and distinct_on is None:
            raise ValueError("No se encontraron columnas numéricas y no se indicó 'distinct_on'.")
    elif isinstance(y, str):
        if y not in df.columns:
            raise ValueError(f"La columna '{y}' no existe en el DataFrame.")
        y_cols = [y]
    else:
        missing = [col for col in y if col not in df.columns]
        if missing:
            raise ValueError(f"Las siguientes columnas no existen en el DataFrame: {missing}")
        y_cols = y


    # --- Agregación consistente con tus nuevas reglas ---
    df2, xlabel_final, _ = _ensure_xlabel(df, xlabel)

    df_plot = _aggregate_frame(
        df=df2,
        xlabel=xlabel_final,
        y_cols=y_cols,
        agg=agg,
        distinct_on=distinct_on,
        drop_dupes_before_sum=drop_dupes_before_sum,
        where=where,
    )
    # --- Graficar (igual que antes) ---
    fig, ax = plt.subplots(figsize=(10, 6))
    if isinstance(df_plot, pd.Series):
        df_plot = df_plot.to_frame(name=y_cols[0] if y_cols else "valor")

    if df_plot.shape[1] == 1:
        colname = df_plot.columns[0]
        ax.barh(df_plot.index.astype(str), df_plot[colname], color=color)
        ax.set_xlabel(colname)
        ax.set_ylabel(xlabel_final if xlabel_final else "Índice")
        if show_legend:
            ax.legend([colname])
    else:
        df_plot.plot(kind="barh", ax=ax, color=color)
        ax.set_xlabel("Valor")
        ax.set_ylabel(xlabel_final if xlabel_final else "Índice")
        if show_legend:
            ax.legend(title="Columnas")
        else:
            leg = ax.get_legend()
            if leg: leg.remove()

    ax.set_title(titulo)
    ax.grid(axis="x", linestyle="--", alpha=0.6)
    if show_values:
        _annotate_bars(ax)
    fig.tight_layout()
    if show:
        plt.show()
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

    # Figura
    n_cols = len(df_display.columns)
    n_rows = len(df_display)
    fig_w = max(6, n_cols * 2.5)
    fig_h = max(2.5, n_rows * 0.5 + 2)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    tabla = ax.table(
        cellText=df_display.values.tolist(),
        colLabels=df_display.columns.tolist(),
        cellLoc="center",
        loc="center"
    )
    tabla.auto_set_font_size(False)
    tabla.set_fontsize(10)
    tabla.scale(1.1, 1.2)
    for col_idx in range(n_cols):
        tabla.auto_set_column_width(col=col_idx)

    header_bg = color if color else "#25347a"
    for (row, col), cell in tabla.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold", color="white")
            cell.set_facecolor(header_bg)
        else:
            cell.set_facecolor("#f9f9f9" if (row % 2 == 1) else "white")

    ax.set_title(titulo, fontsize=13, fontweight="bold", pad=20)
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

def _stack_columns(df: pd.DataFrame,
                   columns: list[str],
                   *,
                   output_col: str = "tipo_riesgo",
                   value_col: str = "valor",
                   label_map: dict[str,str] | None = None) -> pd.DataFrame:
    m = df.melt(value_vars=columns, var_name=output_col, value_name=value_col)
    if label_map:
        m[output_col] = m[output_col].map(lambda x: label_map.get(x, x))
    return m

@register("plt_from_params")
def plot_from_params(df, params):
    fn       = params["function_name"]
    xlabel   = params.get("xlabel")
    y        = params.get("y")
    agg      = params.get("agg", "sum")
    titulo   = params.get("title", "Gráfico")
    color    = params.get("color")

    # 1) BINNING (si viene del prompt)
    binning  = params.get("binning")
    if binning:
        df, bucket_col = _apply_binning(df, binning)
        params["xlabel"] = bucket_col
        xlabel = bucket_col

    # 1.5) STACK (nuevo): apilar columnas → una columna categórica
    stack = params.get("stack_columns")
    if stack:
        cols       = stack["columns"]               # ej.: ["riesgo_ergonomico","riesgo_quimico",...]
        out_col    = stack.get("output_col", "tipo_riesgo")
        val_col    = stack.get("value_col", "valor")
        label_map  = stack.get("label_map")         # opcional renombrar etiquetas
        keep_val   = stack.get("keep_value")        # ej.: "si" (filtrar)
        df = _stack_columns(df, cols, output_col=out_col, value_col=val_col, label_map=label_map)
        if keep_val is not None:
            df = df[df[val_col] == keep_val]
        params["xlabel"] = out_col
        xlabel = out_col
        # tras “stack”, normalmente `y` será un ID a contar (distinct_on)

    # 2) Llamar a la función elegida
    if fn == "graficar_barras":
        fig, ax = graficar_barras(df, xlabel=xlabel, y=y, agg=agg, titulo=titulo, color=color)
    elif fn == "graficar_barras_horizontal":
        fig, ax = graficar_barras_horizontal(df, xlabel=xlabel, y=y, agg=agg, titulo=titulo, color=color)
    elif fn == "graficar_torta":
        fig, ax = graficar_torta(df, xlabel=xlabel, y=y, agg=agg, titulo=titulo, color=color)
    elif fn == "graficar_tabla":
        fig, ax = graficar_tabla(df, xlabel=xlabel, y=y, agg=agg, titulo=titulo, color=color)
    else:
        raise ValueError(f"Función de gráfico no reconocida: {fn}")
    return fig, ax
