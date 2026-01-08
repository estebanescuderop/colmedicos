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
