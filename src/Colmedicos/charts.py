
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype, is_datetime64_any_dtype
from typing import List, Dict, Any, Tuple, Optional
import base64
from io import BytesIO
from matplotlib._pylab_helpers import Gcf
import operator
import os
from openai import OpenAI
import openai
import time
import json
from pathlib import Path
import html
import re
from typing import Callable, Literal
import webbrowser
import tempfile
import plotly.express as px
from .registry import register


@register("graficar_barras")
def graficar_barras(df, xlabel=None, y=None, agg="sum", titulo="Gráfica de Barras", color=None):
    """
    Crea una gráfica de barras a partir de un DataFrame de pandas.

    Parámetros:
    - df: DataFrame de pandas.
    - xlabel: nombre de la columna categórica (eje X). Si None, se usa el índice del DataFrame.
    - y: nombre o lista de columnas numéricas (eje Y). Si None, se grafican todas las numéricas.
    - agg: método de agregación si hay categorías repetidas (por defecto 'sum').
    - titulo: título de la gráfica.
    - color: color o lista de colores opcional.
    """

    # Validar tipo de entrada
    if not isinstance(df, pd.DataFrame):
        raise TypeError("El parámetro 'df' debe ser un DataFrame de pandas.")

    # Validar columna X
    if xlabel is not None and xlabel not in df.columns:
        raise ValueError(f"La columna '{xlabel}' no existe en el DataFrame.")

    # Validar columna(s) Y
    if y is None:
        # Si no se especifica 'y', usar todas las columnas numéricas
        y_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not y_cols:
            raise ValueError("No se encontraron columnas numéricas en el DataFrame.")
    elif isinstance(y, str):
        if y not in df.columns:
            raise ValueError(f"La columna '{y}' no existe en el DataFrame.")
        y_cols = [y]
    else:
        # Si 'y' es una lista
        missing = [col for col in y if col not in df.columns]
        if missing:
            raise ValueError(f"Las siguientes columnas no existen en el DataFrame: {missing}")
        y_cols = y

    # Si se especifica xlabel, agrupar y agregar
    if xlabel is not None:
        df_plot = df.groupby(xlabel, dropna=False)[y_cols].agg(agg)
    else:
        # Si no hay columna X, usar índice directamente
        df_plot = df[y_cols]

    # SIEMPRE trabajar con fig/ax explícitos
    fig, ax = plt.subplots(figsize=(10, 5))

    if df_plot.shape[1] == 1:
        colname = df_plot.columns[0]
        ax.bar(df_plot.index.astype(str), df_plot[colname], color=color)
        ax.set_ylabel(colname)
    else:
        # Usar el ax existente (¡no crear otro!)
        df_plot.plot(kind="bar", ax=ax, color=color)
        ax.set_ylabel("Valor")
        ax.legend(title="Columnas")

    ax.set_title(titulo)
    ax.set_xlabel(xlabel if xlabel else "Índice")
    for label in ax.get_xticklabels():
        label.set_rotation(45)

    fig.tight_layout()

    show = False
    if show:
        plt.show()

    return fig, ax

def graficar_torta(
    df,
    xlabel=None,
    y=None,
    agg="sum",
    titulo="Gráfico de Torta",
    color=None,
    show=False,                 # <- NO mostramos por defecto (útil para exportar/base64)
):
    """
    Crea un gráfico de torta (pie chart) a partir de un DataFrame de pandas.

    Parámetros:
    - df: DataFrame de pandas.
    - xlabel: nombre de la columna categórica (etiquetas de la torta). Si None, se usa el índice.
    - y: columna numérica cuyos valores se representarán en la torta. Si None, toma la primera numérica.
    - agg: método de agregación si hay categorías repetidas (por defecto 'sum').
    - titulo: título del gráfico.
    - color: color o lista de colores opcional.
    - show: si True, hace plt.show().
    Retorna:
    - (fig, ax)
    """

    # --- Validaciones base ---
    if not isinstance(df, pd.DataFrame):
        raise TypeError("El parámetro 'df' debe ser un DataFrame de pandas.")

    if xlabel is not None and xlabel not in df.columns:
        raise ValueError(f"La columna '{xlabel}' no existe en el DataFrame.")

    # Determinar columna numérica (y)
    if y is None:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not num_cols:
            raise ValueError("No se encontraron columnas numéricas en el DataFrame.")
        y = num_cols[0]
    elif y not in df.columns:
        raise ValueError(f"La columna '{y}' no existe en el DataFrame.")

    # --- Agrupar/seleccionar datos ---
    if xlabel is not None:
        # Agrupa por categoría y agrega la métrica
        df_plot = df.groupby(xlabel, dropna=False)[y].agg(agg)
    else:
        # Si no hay categoría, usa los valores de la columna numérica
        # (el índice del DF será la “etiqueta”)
        df_plot = df[y]

    # Asegurar Series (no DataFrame) y limpiar NaN/infs
    if isinstance(df_plot, pd.DataFrame):
        # Si alguien pasó un agg que devolvió DataFrame, forzamos a Series de una sola columna
        if df_plot.shape[1] != 1:
            raise ValueError("El gráfico de torta requiere una única serie numérica.")
        df_plot = df_plot.iloc[:, 0]

    df_plot = df_plot.replace([np.inf, -np.inf], np.nan).dropna()

    if df_plot.empty:
        raise ValueError("No hay datos válidos para graficar (todo es NaN/inf o está vacío).")

    # Valores no negativos (por lo general una torta no tiene sentido con negativos)
    if (df_plot < 0).any():
        # Puedes cambiar a 'raise' si quieres bloquear negativos
        df_plot = df_plot.clip(lower=0)

    total = df_plot.sum()
    if total <= 0:
        raise ValueError("La suma de los valores es 0; no es posible construir la torta.")

    # --- Preparar etiquetas/valores ---
    etiquetas = df_plot.index.astype(str)
    valores = df_plot.values

    # --- Crear gráfico ---
    fig, ax = plt.subplots(figsize=(8, 8))  # <- consistente con barras
    wedges, texts, autotexts = ax.pie(
        valores,
        labels=etiquetas,
        autopct='%1.1f%%',
        startangle=90,
        colors=color,
        shadow=False
    )

    ax.set_title(titulo)
    ax.axis('equal')  # círculo perfecto
    fig.tight_layout()

    if show:
        plt.show()

    return fig, ax


@register("graficas_barras_horizontales")
def graficar_barras_horizontal(
    df,
    xlabel=None,
    y=None,
    agg="sum",
    titulo="Gráfica de Barras Horizontal",
    color=None,
    show=False,                     # <- NO mostramos por defecto (útil para exportar/base64)
):
    """
    Crea una gráfica de barras horizontal a partir de un DataFrame de pandas.
    Retorna: (fig, ax)
    """

    # --- Validaciones ---
    if not isinstance(df, pd.DataFrame):
        raise TypeError("El parámetro 'df' debe ser un DataFrame de pandas.")

    if xlabel is not None and xlabel not in df.columns:
        raise ValueError(f"La columna '{xlabel}' no existe en el DataFrame.")

    # Determinar columna(s) numéricas (y)
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

    # --- Agrupar datos si se especifica xlabel ---
    if xlabel is not None:
        df_plot = df.groupby(xlabel, dropna=False)[y_cols].agg(agg)
    else:
        df_plot = df[y_cols]

    # --- Crear la gráfica de forma explícita en fig/ax ---
    fig, ax = plt.subplots(figsize=(10, 6))

    if df_plot.shape[1] == 1:
        # Una sola serie -> barh directa
        colname = df_plot.columns[0]
        ax.barh(df_plot.index.astype(str), df_plot[colname], color=color)
        ax.set_xlabel(colname)
        ax.set_ylabel(xlabel if xlabel else "Índice")
        ax.legend().remove() if ax.get_legend() else None
    else:
        # Varias series -> usar pandas.plot pero sobre el mismo ax
        df_plot.plot(kind="barh", ax=ax, color=color)
        ax.set_xlabel("Valor")
        ax.set_ylabel(xlabel if xlabel else "Índice")
        ax.legend(title="Columnas")

    ax.set_title(titulo)
    ax.grid(axis="x", linestyle="--", alpha=0.6)

    fig.tight_layout()
    if show:
        plt.show()

    return fig, ax

@register("graficar_tabla")
def graficar_tabla(
    df,
    xlabel=None,
    y=None,
    agg="sum",
    titulo="Tabla de Datos",
    color=None,
    show=False,                     # <- NO mostramos por defecto (útil para exportar)
):
    """
    Crea una representación tipo tabla a partir de un DataFrame de pandas.
    Acepta: xlabel (categoría), y (col/cols numéricas), agg (sum|mean|...), titulo y color del header.
    Retorna: (fig, ax)
    """

    # --- Validaciones ---
    if not isinstance(df, pd.DataFrame):
        raise TypeError("El parámetro 'df' debe ser un DataFrame de pandas.")

    if xlabel is not None and xlabel not in df.columns:
        raise ValueError(f"La columna '{xlabel}' no existe en el DataFrame.")

    # Determinar columnas numéricas (y)
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

    # --- Agrupar datos / preparar base ---
    if xlabel is not None:
        df_plot = df.groupby(xlabel, dropna=False)[y_cols].agg(agg).reset_index()
    else:
        df_plot = df[y_cols].reset_index()
        # renombrar la columna creada por reset_index() para que sea más clara
        if "index" in df_plot.columns:
            df_plot = df_plot.rename(columns={"index": "Índice"})

    # --- Formateo seguro (sin romper strings) ---
    df_display = df_plot.copy()

    for col in df_display.columns:
        if is_numeric_dtype(df_display[col]):
            df_display[col] = df_display[col].round(2)
        elif is_datetime64_any_dtype(df_display[col]):
            df_display[col] = df_display[col].dt.strftime("%Y-%m-%d")

    df_display = df_display.fillna("").astype(str)

    # --- Crear figura/axes explícitos ---
    n_cols = len(df_display.columns)
    n_rows = len(df_display)

    # ancho ~2.5 por columna, alto ~0.5 por fila + margen
    fig_w = max(6, n_cols * 2.5)
    fig_h = max(2.5, n_rows * 0.5 + 2)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    # --- Construir la tabla ---
    tabla = ax.table(
        cellText=df_display.values.tolist(),
        colLabels=df_display.columns.tolist(),
        cellLoc="center",
        loc="center"
    )

    # Estilos base
    tabla.auto_set_font_size(False)
    tabla.set_fontsize(10)
    tabla.scale(1.1, 1.2)

    # Ancho de columnas más uniforme
    # (pequeño ajuste: todas iguales; puedes tunear si quieres medir texto)
    for col_idx in range(n_cols):
        tabla.auto_set_column_width(col=col_idx)

    # Colorear encabezado (fila 0 es header cuando usas colLabels)
    header_bg = color if color else "#25347a"
    for (row, col), cell in tabla.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold", color="white")
            cell.set_facecolor(header_bg)
        else:
            # alternar filas de datos: como el header es 0, datos empiezan en 1
            cell.set_facecolor("#f9f9f9" if (row % 2 == 1) else "white")

    # Título
    ax.set_title(titulo, fontsize=13, fontweight="bold", pad=20)

    fig.tight_layout()
    if show:
        plt.show()

    return fig, ax

@register("plt_from_params")
def plot_from_params(df, params):
    fn     = params["function_name"]
    xlabel = params.get("xlabel")
    y      = params.get("y")
    agg    = params.get("agg", "sum")   # default útil
    titulo = params.get("title", "Gráfico")
    color  = params.get("color")  # NO mostrar por defecto (para exportar)

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

