

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
from Colmedicos.registry import register
from Colmedicos.config import OPENAI_API_KEY

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
- Una instrucción del usuario (español, puede tener acentos o variantes).
- Las columnas disponibles del DataFrame.
- Un texto de contexto adicional: IGNÓRALO. ENFÓCATE EXCLUSIVAMENTE en la instrucción delimitada entre # ... #.

Devuelve EXCLUSIVAMENTE un JSON **válido** con los parámetros necesarios para llamar una de estas funciones:

- graficar_barras(df, xlabel, y, agg, titulo, color, ...)
- graficar_barras_horizontal(df, xlabel, y, agg, titulo, color, ...)
- graficar_torta(df, xlabel, y, agg, titulo, color, ...)
- graficar_tabla(df, xlabel, y, agg, titulo, color, ...)

> Puedes añadir campos opcionales (filtros, únicos, binning, stack/apilar columnas, orden, top-N, multi-X, leyenda) descritos en “Esquema de salida” y “Reglas”.

## Entradas
- instruccion: "{{INSTRUCCION}}"
- columnas: {{COLUMNAS_JSON}}   // ej.: ["edad","identificacion","genero","ventas","IMC","proveedor","categoria","fecha","riesgo_ergonomico","riesgo_quimico","riesgo_psicosocial","riesgo_biomecanico"]

## Reglas de interpretación
1) Tipo de gráfica → "chart_type" y "function_name"
   - Barra(s) → chart_type="barras", function_name="graficar_barras"
   - Barras horizontal(es) → "barras_horizontal", "graficar_barras_horizontal"
   - Torta / Pie → "torta", "graficar_torta"
   - Tabla → "tabla", "graficar_tabla"
   Si no se especifica, asume barras.

2) Título → "title"
   - Si la instrucción trae comillas ('...' o "…"), úsalo tal cual.
   - En otro caso, sintetiza un título breve y claro.

3) Columnas → "xlabel" (categórica) y "y" (numérica o lista de numéricas)
   - Emparejamiento case-insensitive y acentos-insensitive contra ‘columnas’.
   - “por <col>” implica `<col>` en el eje X → `xlabel`.
   - **Multi-X**: si el usuario pide agrupar por varias columnas (p. ej. área + sede), permite `"xlabel": ["area","sede"]` (las funciones combinarán internamente).
   - Si NO se indica `y`, selecciona una numérica razonable; si no es posible, pon `"y": null` y `"needs_disambiguation": true`.
   - Si NO se indica `xlabel`, elige una no numérica razonable; si no es posible, `"xlabel": null` y `"needs_disambiguation": true`.

4) Agregación → "agg"
   - "sumatoria/suma/acumulado" → "sum"
   - "promedio/media" → "mean"
   - "conteo/número/cantidad" → "count"
   - "máximo/mínimo/mediana" → "max" / "min" / "median"
   - **Cómputos sobre únicos**:
     - “conteo de únicos/distintos/sin duplicados de <id>” → `"agg": "distinct_count"` y `"distinct_on": "<id>"`.
     - “sumar <métrica> considerando <id> únicos” → `"agg":"sum"` + `"distinct_on":"<id>"` (se deduplica por <id> antes de agregar).
   - **Suma sobre valores únicos de la propia métrica**: `"agg":"sum_distinct"` (si el agregador lo soporta).

5) Filtros condicionales (bloques AND/OR)
   - Usa **dos** campos:
     - `"conditions_all"`: lista de condiciones combinadas con AND.
     - `"conditions_any"`: lista de **bloques** combinados con OR. Cada ítem puede ser:
       - una condición única `["col","op","valor"]`, o
       - un bloque AND `[[...],[...]]`.
   - Operadores soportados: `">","<","==","!=","?>=","<=","in","not in"`.
     - Para `"in"/"not in"` el valor debe ser **lista**.
   - Rangos del tipo “18.5 ≤ IMC ≤ 24.9” se expresan como **dos** condiciones en el mismo bloque.

6) **Binning / Agrupaciones por rangos**
   - Si la instrucción pide agrupar por intervalos (p. ej., grupos etarios), incluye el bloque `"binning"`:
     {
       "column": "edad",                  // columna fuente a trocear
       "bins":   [lim1, lim2, ..., limN], // límites (pueden ser -inf y +inf)
       "labels": ["r1","r2",...],         // etiquetas en el mismo orden
       "output_col": "grupo_edad"         // opcional; si falta, se crea "<column>_bucket"
     }
   - Cuando se define `binning`, **el `xlabel` debe ser el nombre del bucket** (`output_col` o el auto-generado).

7) **Stack / Apilar columnas → un eje X común**
   - Si el usuario pide armar el eje X a partir de **múltiples columnas** (p. ej., varios tipos de riesgo como “riesgo_ergonomico”, “riesgo_quimico”…), usa `"stack_columns"`:
     {
       "columns": ["colA","colB","colC"], // columnas a apilar
       "output_col": "nombre_eje_x",      // nombre de la nueva columna categórica (obligatorio si quieres controlarlo)
       "value_col": "valor",              // nombre de la columna con valores apilados (por defecto "valor")
       "keep_value": "si",                // opcional: filtra las filas apiladas por este valor exacto
       "label_map": { "colA":"Etiqueta A", "colB":"Etiqueta B" } // opcional: renombra etiquetas del eje
     }
   - Cuando se define `stack_columns`, **el `xlabel` debe ser `output_col`** (o el nombre por defecto si no se especifica).

8) Series múltiples y torta
   - En barras y barras horizontales, `y` puede ser lista (múltiples series).
   - En torta, se espera una sola serie (una métrica agregada por `xlabel`).

9) Colores y leyenda (opcionales)
   - `"color"` puede ser:
     - string (un color): "steelblue", "#25347a", etc.
     - lista de strings (uno por serie/rebanada).
   - `"colors_by_category"` (opcional) mapea color por etiqueta de `xlabel`.
   - `"show_legend": true|false` para mostrar/ocultar leyenda (por defecto, true si hay múltiples series).
   - `"legend_title": "string|null"` para titular la leyenda (si aplica).

10) Orden y top-N (opcionales)
   - `"sort": {"by": "y" | "label", "order": "asc" | "desc"}`
     - Con múltiples series, `"by":"y"` aplica al total/primera serie (elige razonablemente).
   - `"limit_categories": number` para top-N (aplícalo tras ordenar).

11) Control de unicidad y deduplicación previa
   - `"distinct_on": "col|[colA,colB]"` define la clave de unicidad de “entidades” (p. ej., personas).
   - `"drop_dupes_before_sum": true|false` permite deduplicar por `(xlabel, distinct_on)` antes de sumar/promediar.
   - `"unique_by": "col|[colA,colB]"` permite deduplicar filas antes de cualquier cálculo.

12) Validación frente a 'columnas'
   - Si un nombre no coincide, deja el campo en null y marca:
     `"needs_disambiguation": true`, proponiendo alternativas en `"candidates"`.

13) Salida: SOLO JSON válido UTF-8, sin comentarios ni texto adicional.

## Esquema de salida (JSON)
{
  "chart_type": "barras" | "barras_horizontal" | "torta" | "tabla",
  "function_name": "graficar_barras" | "graficar_barras_horizontal" | "graficar_torta" | "graficar_tabla",
  "title": "string",
  "xlabel": "string | string[] | null",
  "y": "string | string[] | null",
  "agg": "sum" | "mean" | "count" | "max" | "min" | "median" | "distinct_count" | "sum_distinct",
  "distinct_on": "string | string[] | null",
  "drop_dupes_before_sum": true | false | null,
  "unique_by": "string | string[] | null",
  "conditions_all": [ ["col","op",valor], ... ],
  "conditions_any": [
    ["col","op",valor],
    [["col","op",v],["col","op",v]]
  ],
  "binning": {
    "column": "string",
    "bins": [number|"-inf"|"+inf", ...],
    "labels": ["string", ...],
    "output_col": "string|null"
  } | null,
  "stack_columns": {
    "columns": ["string", ...],
    "output_col": "string",
    "value_col": "string" | null,
    "keep_value": "any" | null,
    "label_map": { "string":"string", ... } | null
  } | null,
  "color": "string | string[] | null",
  "colors_by_category": { "Etiqueta":"#RRGGBB", ... } | null,
  "show_legend": true | false | null,
  "legend_title": "string | null",
  "sort": { "by":"y"|"label", "order":"asc"|"desc" } | null,
  "limit_categories": number | null,
  "needs_disambiguation": true | false,
  "candidates": { "xlabel": string[], "y": string[] }
}

## Ejemplos

### Ejemplo 1 — Barras horizontales con grupos etarios y conteo único
instruccion: # gráfico de barras horizontales llamado 'Personas por grupos etarios' que utilice la columna de edad para la siguiente agrupación por rangos: (total_0_y_5, total_6_y_11, total_12_y_18, total_19_y_26, total_27_y_59, total_mayores_60) y la columna identificacion para hacer un conteo de personas únicas #
columnas: ["edad","identificacion","genero"]
→
{
  "chart_type": "barras_horizontal",
  "function_name": "graficar_barras_horizontal",
  "title": "Personas por grupos etarios",
  "xlabel": "grupo_edad",
  "y": "identificacion",
  "agg": "distinct_count",
  "distinct_on": "identificacion",
  "drop_dupes_before_sum": false,
  "unique_by": null,
  "conditions_all": [],
  "conditions_any": [],
  "binning": {
    "column": "edad",
    "bins":   ["-inf", 5, 11, 18, 26, 59, "+inf"],
    "labels": [
      "total_0_y_5",
      "total_6_y_11",
      "total_12_y_18",
      "total_19_y_26",
      "total_27_y_59",
      "total_mayores_60"
    ],
    "output_col": "grupo_edad"
  },
  "stack_columns": null,
  "color": null,
  "colors_by_category": null,
  "show_legend": false,
  "legend_title": null,
  "sort": { "by": "label", "order": "asc" },
  "limit_categories": null,
  "needs_disambiguation": false,
  "candidates": { "xlabel": [], "y": [] }
}

### Ejemplo 2 — Torta con filtro y top-N
instruccion: # pie chart del total de ventas por categoría solo 2025, mostrar top 5 #
columnas: ["categoria","ventas","anio"]
→
{
  "chart_type": "torta",
  "function_name": "graficar_torta",
  "title": "Ventas por categoría (2025)",
  "xlabel": "categoria",
  "y": "ventas",
  "agg": "sum",
  "distinct_on": null,
  "drop_dupes_before_sum": false,
  "unique_by": null,
  "conditions_all": [["anio","==",2025]],
  "conditions_any": [],
  "binning": null,
  "stack_columns": null,
  "color": null,
  "colors_by_category": null,
  "show_legend": true,
  "legend_title": null,
  "sort": { "by": "y", "order": "desc" },
  "limit_categories": 5,
  "needs_disambiguation": false,
  "candidates": { "xlabel": [], "y": [] }
}

### Ejemplo 3 — Barras con múltiples series y colores
instruccion: # barras por región comparando ventas y costos, ordenar ascendente por etiqueta, colores azul y verde #
columnas: ["region","ventas","costos"]
→
{
  "chart_type": "barras",
  "function_name": "graficar_barras",
  "title": "Ventas y costos por región",
  "xlabel": "region",
  "y": ["ventas","costos"],
  "agg": "sum",
  "distinct_on": null,
  "drop_dupes_before_sum": false,
  "unique_by": null,
  "conditions_all": [],
  "conditions_any": [],
  "binning": null,
  "stack_columns": null,
  "color": ["steelblue","seagreen"],
  "colors_by_category": null,
  "show_legend": true,
  "legend_title": "Métricas",
  "sort": { "by": "label", "order": "asc" },
  "limit_categories": null,
  "needs_disambiguation": false,
  "candidates": { "xlabel": [], "y": [] }
}

### Ejemplo 4 — Multi-X (dos columnas en el eje)
instruccion: # barras por sede y área sumando ventas #
columnas: ["sede","area","ventas"]
→
{
  "chart_type": "barras",
  "function_name": "graficar_barras",
  "title": "Ventas por sede y área",
  "xlabel": ["sede","area"],
  "y": "ventas",
  "agg": "sum",
  "distinct_on": null,
  "drop_dupes_before_sum": false,
  "unique_by": null,
  "conditions_all": [],
  "conditions_any": [],
  "binning": null,
  "stack_columns": null,
  "color": null,
  "colors_by_category": null,
  "show_legend": false,
  "legend_title": null,
  "sort": { "by": "y", "order": "desc" },
  "limit_categories": null,
  "needs_disambiguation": false,
  "candidates": { "xlabel": [], "y": [] }
}

### Ejemplo 5 — **Apilar columnas de riesgo** + conteo de personas únicas
instruccion: # Grafica de barras llamada 'Tipo de riesgo' con un conteo de registros únicos de identificación, y los siguientes ejes x: riesgo ergonómico="si", riesgo quimico="si", riesgo psicosocial="si", riesgo biomecanico="si" #
columnas: ["identificacion","riesgo_ergonomico","riesgo_quimico","riesgo_psicosocial","riesgo_biomecanico"]
→
{
  "chart_type": "barras",
  "function_name": "graficar_barras",
  "title": "Tipo de riesgo",
  "xlabel": "tipo_riesgo",
  "y": "identificacion",
  "agg": "distinct_count",
  "distinct_on": "identificacion",
  "drop_dupes_before_sum": false,
  "unique_by": null,
  "conditions_all": [],
  "conditions_any": [],
  "binning": null,
  "stack_columns": {
    "columns": ["riesgo_ergonomico","riesgo_quimico","riesgo_psicosocial","riesgo_biomecanico"],
    "output_col": "tipo_riesgo",
    "value_col": "valor",
    "keep_value": "si",
    "label_map": {
      "riesgo_ergonomico": "Ergonómico",
      "riesgo_quimico": "Químico",
      "riesgo_psicosocial": "Psicosocial",
      "riesgo_biomecanico": "Biomecánico"
    }
  },
  "color": null,
  "colors_by_category": null,
  "show_legend": false,
  "legend_title": null,
  "sort": { "by": "y", "order": "desc" },
  "limit_categories": null,
  "needs_disambiguation": false,
  "candidates": { "xlabel": [], "y": [] }
}
"""



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
- Nuevas (para únicos):
"distinct_sum" → suma de una columna tras eliminar duplicados por una o varias llaves.
También puedes usar cualquier op simple (sum, count, avg, etc.) con el modificador opcional dedupe_by para pedir “quitar duplicados antes de calcular”.
- Avanzadas:
  "ratio" (numerator/denominator con condiciones propias)
  "weighted_avg" (promedio ponderado con columna de pesos)
- (Opcionalmente puedes mapear frases comunes a estas ops: 
  suma/total -> sum; contar/número de -> count; promedio/media -> avg; 
  mínimo -> min; máximo -> max; únicos/distintos -> distinct_count; 
  porcentaje/tasa = ratio o avg con multiplicador externo si aplica)

Campos por operación (además de los ya existentes):
- dedupe_by: ["colA", "colB", ...] (opcional, nuevo):
    - Si existe, antes de calcular la métrica, eliminar duplicados usando esas columnas como clave (equivalente a drop_duplicates(subset=dedupe_by)).
    - Útil para “sumatoria de valores únicos” o “conteo único de personas dentro de condiciones”.

- conditions_logic: "AND" | "OR" (opcional, nuevo; por defecto "AND"):
    - Define cómo combinar las condiciones de conditions (si se usa ese campo).

- condition_groups: [ { "conditions":[...], "logic":"AND|OR" }, ... ] (opcional, nuevo):
    - Permite expresar lógicas más complejas del tipo (A AND B) OR (C AND D).
    - Regla: si se especifica condition_groups, el agente no debe usar conditions plano en esa misma operación.

Nota: Para conteo de personas únicas usa op: "distinct_count", column: "IdPersona" y si además necesitas filtrar, agrega conditions o condition_groups.
Para sumatoria única (p. ej., sumar el “Monto” por “FacturaID” único), usa op: "distinct_sum", column: "Monto", dedupe_by: ["FacturaID"].  


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
10) Al mapear frases del usuario:
    - “personas únicas”, “sin duplicados”, “únicos por …” → usar distinct_count o añadir dedupe_by a la operación.
    - “sumatoria única”, “sumar una vez por …” → usar distinct_sum o sum con dedupe_by.
11) Si el usuario pide agrupación (“por región/proveedor/mes…”), usar "group_by" (string o lista).
12) Si el usuario expresa rangos (p. ej., IMC), descompón en múltiples operaciones con condiciones

Salida: SOLO JSON válido UTF-8 (sin texto extra), con esta forma:

{
  "operations": [
    {
      "op": "sum | count | avg | min | max | distinct_count | distinct_sum | ratio | weighted_avg",
      "alias": "string",

      "column": "string|null",
      "conditions": [["col","op","valor"], ...],
      "conditions_logic": "AND|OR",
      "condition_groups": [
        { "conditions": [["col","op","valor"], ...], "logic": "AND|OR" }
      ],

      "dedupe_by": ["colA","colB"],         // NUEVO (opcional)
      "count_nulls": true|false,            // solo "count"

      "numerator":   { "column":"string", "conditions":[["col","op","valor"]], "conditions_logic":"AND|OR" },
      "denominator": { "column":"string", "conditions":[["col","op","valor"]], "conditions_logic":"AND|OR" },
      "weights": "string",                  // weighted_avg
      "safe_div0": number|null
    }
  ],
  "group_by": null | "string" | ["string", ...],
  "needs_disambiguation": false,
  "candidates": { "columns": [], "group_by": [], "by_operation": [] }
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

# 2) “Sumatoria única del Monto por Factura (evitar duplicados por FacturaID) solo 2025”
{
  "operations": [
    {
      "op": "distinct_sum",
      "alias": "monto_unico_2025",
      "column": "Monto",
      "dedupe_by": ["FacturaID"],
      "conditions": [["Anio", "==", 2025]]
    }
  ],
  "group_by": null,
  "needs_disambiguation": false,
  "candidates": { "columns": [], "group_by": [], "by_operation": [] }
}

# 3) “IMC por rangos (buckets): contar personas por categoría”
{
  "operations": [
    {"op":"count","alias":"bajo_peso","column":"IdPersona","conditions":[["IMC","<",18.5]]},
    {"op":"count","alias":"peso_normal","column":"IdPersona","conditions":[["IMC",">=",18.5],["IMC","<=",24.9]]},
    {"op":"count","alias":"sobrepeso","column":"IdPersona","conditions":[["IMC",">=",25.0],["IMC","<=",29.9]]},
    {"op":"count","alias":"obesidad_I","column":"IdPersona","conditions":[["IMC",">=",30.0],["IMC","<=",34.9]]},
    {"op":"count","alias":"obesidad_II","column":"IdPersona","conditions":[["IMC",">=",35.0],["IMC","<=",39.9]]},
    {"op":"count","alias":"obesidad_III","column":"IdPersona","conditions":[["IMC",">",40.0]]}
  ],
  "group_by": null,
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

