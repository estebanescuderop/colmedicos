

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
from .config import OPENAI_API_KEY

API_KEY = "***REMOVED***"  # Reemplaza por tu API key de OpenAI  # Reemplaza por tu API key de OpenAI
instruccion = "Todo lo que no esté entre los signos ++, redactalo exactamente igual, lo que si esté, sigue las instrucciones y lo reemplazas por lo que haya originalmente entre ++: "
client = openai.OpenAI(api_key=API_KEY)

@register("ask_gpt5")
def ask_gpt5(pregunta):
    """Envía un prompt y devuelve la respuesta de GPT-5."""
    time.sleep(3)
    respuesta = client.chat.completions.create(
        model="gpt-5",  # 👈 Aquí usas GPT-5 directamente
        messages=[
            {"role": "system", "content": "Eres un asistente preciso y coherente con instrucciones de edición de texto."},
            {"role": "user", "content": instruccion + pregunta}
        ]
    )

    texto_respuesta = respuesta.choices[0].message.content
    return texto_respuesta


msj_grafo = """## Objetivo
A partir de:
- Una instrucción del usuario (español, puede tener acentos o variantes)
- Las columnas disponibles del DataFrame
- Un texto plano con mucha información, sólo enfocate UNICAMENTE en la instrucción que se encuentra entre #

Devuelve EXCLUSIVAMENTE un JSON válido que contenga los parámetros necesarios para llamar a una de estas funciones de plotting:

- graficar_barras(df, xlabel, y, agg, titulo, color)
- graficar_barras_horizontal(df, xlabel, y, agg, titulo, color)
- graficar_torta(df, xlabel, y, agg, titulo, color)
- graficar_tabla(df, xlabel, y, agg, titulo, color)

No incluyas nada de texto explicativo fuera del JSON.

## Entradas
- instruccion: "{{INSTRUCCION}}"
- columnas: {{COLUMNAS_JSON}}   # ejemplo: ["genero","nro_hijos","ventas"]

## Reglas de mapeo y normalización
1) Tipo de gráfica → "chart_type" y "function_name"
   - Barras / barra(s) → chart_type: "barras", function_name: "graficar_barras"
   - Barras horizontal(es) → "barras_horizontal", "graficar_barras_horizontal"
   - Torta / pie → "torta", "graficar_torta"
   - Tabla → "tabla", "graficar_tabla"
   Si no se especifica, asume "barras".

2) Título → "title"
   - Si la instrucción trae comillas ('...' o "…") úsalo.
   - Sino, sintetiza un título breve a partir del pedido.

3) Columnas → "xlabel" (categórica) y "y" (numérica o lista de numéricas)
   - Case-insensitive y acento-insensitive. Empareja con la lista exacta de 'columnas'.
   - Acepta sinónimos como “eje X/columna X/usar X/por X”.
   - Si el usuario pide “por <col>”, usa <col> como xlabel.
   - Si faltan datos:
       - Si NO se indica 'y', elige la primera columna numérica de 'columnas_numéricas' (si te la dan) o devuelve null y marca "needs_disambiguation": true.
       - Si NO se indica 'xlabel', elige una columna no numérica (si te la dan) o devuelve null y marca "needs_disambiguation": true.

4) Agregación → "agg"
   - "sumatoria", "suma", "acumulado" → "sum"
   - "promedio", "media" → "mean"
   - "conteo", "número de", "cantidad", "count" → "count"
   - "máximo" → "max", "mínimo" → "min", "mediana" → "median"
   Si no se indica, usa "sum" por defecto.

5) Color → "color"
   - Inclúyelo solo si el usuario lo pide explícitamente (ej. "en azul").

6) Validación frente a 'columnas'
   - "xlabel" y "y" deben pertenecer a 'columnas'. Si no hay match razonable, devuelve null y marca:
     "needs_disambiguation": true, y lista "candidates" con las columnas más probables.

7) Salida
   Responde **únicamente** con un JSON válido UTF-8, sin comentarios, sin texto extra, con esta forma:

{
  "chart_type": "barras" | "barras_horizontal" | "torta" | "tabla",
  "function_name": "graficar_barras" | "graficar_barras_horizontal" | "graficar_torta" | "graficar_tabla",
  "title": "string",
  "xlabel": "string|null",
  "y": "string | string[] | null",
  "agg": "sum" | "mean" | "count" | "max" | "min" | "median",
  "color": "string | null",
  "needs_disambiguation": true | false,
  "candidates": { "xlabel": string[], "y": string[] }
}

- Si no hay ambigüedad, "needs_disambiguation": false y "candidates" debe ser {}.

## Ejemplos

### Ejemplo 1
instruccion: "Gráfica de barras llamada 'conteo de hijos por genero' que utilice la columna de genero en la columna X y la columna de número de hijos para hacer una sumatoria"
columnas: ["genero","nro_hijos"]

→
{
  "chart_type": "barras",
  "function_name": "graficar_barras",
  "title": "conteo de hijos por genero",
  "xlabel": "genero",
  "y": "nro_hijos",
  "agg": "sum",
  "color": null,
  "needs_disambiguation": false,
  "candidates": {}
}

### Ejemplo 2
instruccion: "Haz un pie chart del total de ventas por categoría"
columnas: ["categoria", "ventas", "fecha"]

→
{
  "chart_type": "torta",
  "function_name": "graficar_torta",
  "title": "Total de ventas por categoría",
  "xlabel": "categoria",
  "y": "ventas",
  "agg": "sum",
  "color": null,
  "needs_disambiguation": false,
  "candidates": {}
}

### Ejemplo 3 (ambigüedad)
instruccion: "Quiero una barra horizontal por área con promedio"
columnas: ["area", "score", "nota"]

→
{
  "chart_type": "barras_horizontal",
  "function_name": "graficar_barras_horizontal",
  "title": "Promedio por área",
  "xlabel": "area",
  "y": null,
  "agg": "mean",
  "color": null,
  "needs_disambiguation": true,
  "candidates": { "xlabel": [], "y": ["score","nota"] }
}"""

@register("graficos_gpt5")
def graficos_gpt5(df, pregunta):
    """Envía un prompt y devuelve la respuesta de GPT-5."""
    time.sleep(3)
    
    # Extraer nombres de columnas
    columnas = df.columns.tolist()
    
    subprompt = (
    msj_grafo
    .replace("{INSTRUCCION}", pregunta)
    .replace("{COLUMNAS_JSON}", json.dumps(columnas, ensure_ascii=False))
)
    
    respuesta = client.chat.completions.create(
        model="gpt-5",  # 👈 Aquí usas GPT-5 directamente
        messages=[
            {"role": "system", "content": "Eres un analista que extrae parámetros para construir gráficas a partir de una instrucción en lenguaje natural y una lista de columnas disponibles de un DataFrame de pandas."},
            {"role": "user", "content": subprompt}
        ]
    )

    texto_respuesta = respuesta.choices[0].message.content
    params = json.loads(texto_respuesta)
    return params

MSJ_OPS = """
Eres un analista de datos. A partir de una instrucción en español y una lista de columnas de un DataFrame de pandas,
debes devolver EXCLUSIVAMENTE un JSON válido con una especificación de MÚLTIPLES operaciones a ejecutar sobre el DataFrame.
Trabaja EXCLUSIVAMENTE con la instrucción que se encuentre entre ||

El JSON debe incluir:
- "operations": lista de operaciones independientes (cada una con su propia columna/condiciones/alias).
- "group_by": null, string o lista de strings para agrupar resultados (opcional).

Operaciones disponibles (operations[i].op):
- Básicas por columna:
  "sum" | "count" | "avg" | "min" | "max" | "distinct_count"
- Avanzadas:
  "ratio" (numerator/denominator con condiciones propias)
  "weighted_avg" (promedio ponderado con columna de pesos)
- (Opcionalmente puedes mapear frases comunes a estas ops: 
  suma/total -> sum; contar/número de -> count; promedio/media -> avg; 
  mínimo -> min; máximo -> max; únicos/distintos -> distinct_count; 
  porcentaje/tasa = ratio o avg con multiplicador externo si aplica)

Reglas:
1) Interpreta la instrucción y desglósala en una o más operaciones. 
   Si el usuario pide “por región” o “por categoría”, usa "group_by" con esos nombres de columnas.
2) Cada operación DEBE tener "alias" para nombrar la métrica resultante.
3) Valida que todas las columnas referidas existan en 'columnas'. Si hay ambigüedad, usa "needs_disambiguation": true y propone alternativas en "candidates".
4) Condiciones: usa una lista de tuplas/objetos con (columna, operador, valor). Operadores soportados: 
   ">", "<", "==", "!=", ">=", "<=", "in", "not in". 
   Para "in"/"not in" el valor debe ser lista.
5) Para "count", por defecto cuenta NO nulos en la columna indicada. Si se requiere contar nulos, agrega "count_nulls": true.
6) Para "avg" o "sum", convierte a numérico implícitamente (coerción), ignorando NaN (equivalente a skipna=True).
7) Para "ratio":
   - Debes especificar "numerator" y "denominator", cada uno con "column" y "conditions" propias.
   - Usa "safe_div0" para definir el valor a retornar si el denominador es 0 (por defecto null/NaN).
8) Para "weighted_avg":
   - "column": valores, "weights": columna de pesos, "conditions": condiciones del subconjunto (opcional).
   - Usa "safe_div0" si la suma de pesos es 0 (por defecto null/NaN).
9) Si el usuario no pide agrupación, "group_by": null.

Salida: SOLO JSON válido UTF-8 (sin texto extra), con esta forma:

{
  "operations": [
    {
      "op": "sum" | "count" | "avg" | "min" | "max" | "distinct_count" | "ratio" | "weighted_avg",
      "alias": "string",
      // Para ops simples:
      "column": "string|null",
      "conditions": [ ["col","op","valor"], ... ],
      "count_nulls": true | false,                // solo para "count"
      // Para ratio:
      "numerator":   { "column":"string", "conditions":[ ["col","op","valor"], ... ] },
      "denominator": { "column":"string", "conditions":[ ["col","op","valor"], ... ] },
      "safe_div0": number | null,
      // Para weighted_avg:
      "weights": "string",
      "safe_div0": number | null
    }
  ],
  "group_by": null | "string" | ["string", ...],
  "needs_disambiguation": true | false,
  "candidates": {
    "columns": ["sugerencia_col1","sugerencia_col2", ...],
    "group_by": ["sugerencia_gb1","sugerencia_gb2", ...],
    "by_operation": [
      {
        "op_index": 0,
        "column_alternatives": ["colA","colB"],
        "invalid_reasons": ["columna 'X' no existe", "..."]
      }
    ]
  }
}

Ejemplos:

# 1) “Suma de Ventas y conteo de Pedidos para la Categoría B”
{
  "operations": [
    {"op":"sum",   "column":"Ventas",   "conditions":[["Categoria","==","B"]], "alias":"ventas_B"},
    {"op":"count", "column":"PedidoID", "conditions":[["Categoria","==","B"]], "alias":"n_pedidos_B"}
  ],
  "group_by": null,
  "needs_disambiguation": false,
  "candidates": { "columns": [], "group_by": [], "by_operation": [] }
}

# 2) “Promedio de Ticket y máximo de Ventas por Región”
{
  "operations": [
    {"op":"avg", "column":"Ticket", "conditions":[], "alias":"ticket_prom"},
    {"op":"max", "column":"Ventas", "conditions":[], "alias":"ventas_max"}
  ],
  "group_by": "Region",
  "needs_disambiguation": false,
  "candidates": { "columns": [], "group_by": [], "by_operation": [] }
}

# 3) “Tasa de aprobación (Aprobados/Solicitudes) por Sucursal, solo 2025”
{
  "operations": [
    {
      "op":"ratio",
      "alias":"tasa_aprob",
      "numerator":   {"column":"Aprobados",  "conditions":[["Anio","==",2025]]},
      "denominator": {"column":"Solicitudes","conditions":[["Anio","==",2025]]},
      "safe_div0": 0.0
    }
  ],
  "group_by": "Sucursal",
  "needs_disambiguation": false,
  "candidates": { "columns": [], "group_by": [], "by_operation": [] }
}

Entradas:
- instruccion: "{INSTRUCCION}"
- columnas: {COLUMNAS_JSON}
"""
@register("operaciones_gpt5")
def operaciones_gpt5(df, pregunta):
    """Envía un prompt y devuelve la respuesta de GPT-5."""
    time.sleep(3)
    
    # Extraer nombres de columnas
    columnas = df.columns.tolist()
    
    subprompt = (
    MSJ_OPS
    .replace("{INSTRUCCION}", pregunta)
    .replace("{COLUMNAS_JSON}", json.dumps(columnas, ensure_ascii=False))
)
    
    respuesta = client.chat.completions.create(
        model="gpt-5",  # 👈 Aquí usas GPT-5 directamente
        messages=[
            {"role": "system", "content": "Eres un analista que extrae parámetros para realizar calculos a partir de una instrucción en lenguaje natural y una lista de columnas disponibles de un DataFrame de pandas."},
            {"role": "user", "content": subprompt}
        ]
    )

    texto_respuesta = respuesta.choices[0].message.content
    params = json.loads(texto_respuesta)
    return params

