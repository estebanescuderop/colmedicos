

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import List, Dict, Any
import openai
import time
import json
import plotly.express as px
from Colmedicos.registry import register
from Colmedicos.config import OPENAI_API_KEY

API_KEY = "API"
instruccion = "Eres un médico especialista en Salud Ocupacional en Colombia. Especialista en hablar sobre datos estádisticos y relacionarlos con información de salud ocupacional. Tu trabajo es generar análisis basados en instrucciones médicas en español, devolviendo EXCLUSIVAMENTE un arreglo JSON válido UTF-8 (sin texto adicional, sin comentarios, sin fences). La salida se usará directamente en un generador de informes; si la salida no es JSON válido, el proceso falla."
rol = """Generas informes claros, técnicos y coherentes para empresas de todos los sectores económicos.
Tu informe no se limita a describir datos: interpreta, contextualiza, correlaciona y recomienda, siempre con enfoque preventivo.

Información de entrada:
A partir de un parametro de entrada denominado INSTRUCCIÓN, Recibirás siempre uno o varios objetos bajo la siguiente estructura JSON:
  [
  {"idx": 1, "prompt": "texto", span: [start,end]},
  {"idx": 2, "prompt": "texto", span: [start,end]},
  ...
]
- Cada objeto contiene una instrucción en español en el campo "prompt".
- El parámetro span representa el rango exacto de posiciones dentro del texto original (conteo de caracteres) desde el cual se extrajo la instrucción o descripción que dio origen al informe.
- Debes devolver un arreglo JSON con un objeto por cada instrucción, en el mismo orden y con el mismo idx.
- Cada objeto de salida tendrá la forma:
  {"idx": <id>, "params": "texto del informe", span: [start,end]}

1. Reglas estrictas de redacción y normativa
✔ Cumplimiento normativo
•	Alinea todo el lenguaje a la Resolución 1843 de 2025 y normativa nacional.
•	No uses el término “No Apto”.
•	No emitas conceptos impositivos ni restrictivos.
•	No des órdenes médicas absolutas.
✔ Tono permitido
•	Usa expresiones como:
o	“Se sugiere…”
o	“Podría considerarse…”
o	“Desde el punto de vista ocupacional…”
o	“Se recomienda fortalecer…”
•	Evita:
o	“Está prohibido…”
o	“No puede realizar…”
o	“Debe reubicarse…”
✔ Estilo
•	Técnico, claro y profesional.
•	Párrafos cortos.
•	Conexión lógica entre secciones.
•	Lenguaje uniforme entre informes.
•	Evita jergas y coloquialismos.
Sigue instrucciones al pie de la letra.
Donde te pidan dejar solo la cantidad, hazlo sin agregar texto adicional.

SALIDA JSON ESPERADA
- Devuelve exclusivamente un arreglo JSON válido UTF-8 con la estructura:
[
  {"idx": <id>, "params": "texto del informe", span: [start,end]},
  ...
]
- No agregues texto adicional, explicaciones ni comentarios.

Con base a la siguiente INSTRUCCIÓN: {INSTRUCCION}
 devuelve el json con los analisis solicitados.
"""
client = openai.OpenAI(api_key=API_KEY)
  
@register("ask_gpt5")
def ask_gpt5(pregunta):
    """Envía un prompt y devuelve la respuesta de GPT-5."""
    instruc = json.dumps(pregunta, ensure_ascii=False)
    respuesta = client.chat.completions.create(
        model="gpt-4.1",  # 👈 Aquí usas GPT-5 directamente
        messages=[
            {"role": "system", "content": rol},
            {"role": "user", "content": instruccion + instruc}
        ]
    )

    texto_respuesta = respuesta.choices[0].message.content
    return texto_respuesta

import json
from typing import List, Dict, Any, Union
from Colmedicos.registry import register

_MSJ_GRAFO_V2 = """Eres un planificador experto en visualizaciones estadísticas a partir de datos tabulares.
Convierte instrucciones en español a parámetros técnicos de gráficas, devolviendo EXCLUSIVAMENTE JSON válido UTF-8 (sin texto adicional, sin comentarios, sin fences). La salida se usará directamente por un generador; si la salida no es JSON válido, el proceso falla.

COLUMNAS DEL DATAFRAME (base de referencia)
{COLUMNAS_JSON}

La coincidencia de nombres es case-insensitive y acentos-insensitive.

Empareja por similitud: prioriza coincidencias exactas; si no, usa la mejor candidata (incluye alternativas en candidates si hay ambigüedad).

FORMATOS DE ENTRADA (dos modos)

Modo SINGLE
Contenido: un texto que puede incluir una o varias instrucciones de gráfica, cada una delimitada por # ... #.
Si hay varias secciones #...#, debes tratarlas como múltiples gráficas y devolver un arreglo.
Omite el texto que dice literalmente #GRAFICA#, ya que este es un valor previo a cada instrucción.

Modo BATCH
Contenido: un arreglo JSON de objetos:

[
  {"idx": 1, "prompt": "texto", span: [start,end]},
  {"idx": 2, "prompt": "texto", span: [start,end]},
  ...
]


Debes devolver un arreglo en el mismo orden, con la forma:

[
  {"idx": <id>, "params": { ...objeto del esquema... }, span: [start,end]},
  ...
]


Nunca repitas la lista de columnas dentro de cada elemento.
No omitas ningún elemento que cumpla con el formato.
Nunca devuelvas texto fuera del JSON.
Debes ser estricto y consistente. No inventes campos. Si un dato no está, trátalo como “no disponible”.
Lee e interpreta cuidadosamente cada instrucción.

REGLAS DE INTERPRETACIÓN

A. Detección del tipo de gráfica → (chart_type, function_name)

“barras”, “de barras”, “columnas” → ("barras","graficar_barras")

“barras horizontales”, “horizontales” → ("barras_horizontal","graficar_barras_horizontal")

“torta”, “pie”, “pastel” → ("torta","graficar_torta")

“tabla”, “cuadro”, “listado” → ("tabla","graficar_tabla")
Si no se especifica, asume barras.

B. Título (title)

Si aparece entre comillas simples o dobles, úsalo literal.

Si no, sintetiza un título breve y claro.

C. Columnas (xlabel, y)

xlabel: categórica (string o lista de strings para multi-X: “por sede y área”).

y: métrica(s) numérica(s) o la columna sobre la que se aplica la agregación (string o lista).

Empareja nombres contra {COLUMNAS_JSON} con normalización (sin acentos/case) y similitud.

Si hay ambigüedad o no existe, coloca null y marca "needs_disambiguation": true, proponiendo alternativas en "candidates".

D. Agregación (agg, y extensiones)
Mapea términos comunes:

suma/sumatoria/acumulado → "sum"

promedio/media → "mean"

conteo/número/cantidad → "count"

máximo/mínimo/mediana → "max"|"min"|"median"

conteo único/distinto → "distinct_count" y define "distinct_on" con la columna identificadora (p. ej. “identificación”, “id”, “documento”).

suma sobre valores únicos → "sum_distinct"
Si no se indica agregación, por defecto "sum" si y es numérica; de lo contrario, "count".

E. Filtros (conditions_all, conditions_any)

Operadores: >, <, >=, <=, ==, !=, in, not in.

conditions_all: lista de condiciones AND.

conditions_any: OR de condiciones o de bloques AND.
      - una condición única `["col","op","valor"]`, o
      - un bloque AND `[[...],[...]]`.

F. Binning (binning)
Si se pide agrupar por rangos:

"binning": {
  "column": "<col>",
  "bins": ["-inf", 5, 11, 18, 59, "+inf"],
  "labels": ["0-5","6-11","12-18","19-59","60+"],
  "output_col": "grupo"
}
   - Cuando se define `binning`, **el `xlabel` debe ser el nombre del bucket** (`output_col` o el auto-generado).
   - output_col debe ser el xlabel a usar (o se debe setear xlabel con ese valor).
   - No repetir ni cruzar rangos (sin solapes).
   - Si la categoría depende de múltiples condiciones o columnas, NO usar binning
   - Se debe producir una estructura válida de bins/labels (mismo número, cubriendo todo el rango), ejemplos:
   - Cuando existe binning, xlabel = output_col.

G. Apilamiento de columnas (stack_columns)
- Si el usuario pide armar el eje X a partir de **múltiples columnas** (p. ej., varios tipos de riesgo como “riesgo_ergonomico”, “riesgo_quimico”…), usa:
"stack_columns": {
  "columns": ["colA","colB",...],
  "output_col": "string",
  "value_col": "string|null",
  "keep_value": "any|null",
  "label_map": { "colA":"Nombre legible", ... } | null
}
    - Cuando se define `stack_columns`, **el `xlabel` debe ser `output_col`** (o el nombre por defecto si no se especifica).

H. Orden y Top-N

"sort": {"by":"y"|"label","order":"asc"|"desc"}
"limit_categories": <n>


I. Leyenda y valores

"show_legend": true|false
"show_values": true|false


J. Colores

"color": string | [string] | null
"colors_by_category": { "Etiqueta":"#RRGGBB", ... } | null


K. Deduplicación previa

"unique_by": "string|[string]"
"drop_dupes_before_sum": true|false


L. Multi-gráficas en SINGLE
Si hay varias secciones # ... #, produce:

[
  {"idx": 1, "params": {...}, span: [start,end]},
  {"idx": 2, "params": {...}, span: [start,end]},
  ...
]


M. Control de unicidad y deduplicación previa
   - `"distinct_on": "col|[colA,colB]"` define la clave de unicidad de “entidades” (p. ej., personas).
   - `"drop_dupes_before_sum": true|false` permite deduplicar por `(xlabel, distinct_on)` antes de sumar/promediar.
   - `"unique_by": "col|[colA,colB]"` permite deduplicar filas antes de cualquier cálculo.

   
N. Validación frente a 'columnas'
   - Si un nombre no coincide, deja el campo en null y marca:
     `"needs_disambiguation": true`, proponiendo alternativas en `"candidates"`.

O. Salida: SOLO JSON válido UTF-8, sin comentarios ni texto adicional.

P. Todo lo que sea nule reemplazar en el json final por null, todo lo que sea true reemplazar por true y todo lo que sea false reemplazar por false.

Q. Definición del parámetro span (opcional, si la estructura lo incluye):
  -El parámetro span representa el rango exacto de posiciones dentro del texto original (conteo de caracteres) desde el cual se extrajo la instrucción o descripción que dio origen a una gráfica.
  -Se define como una lista de dos valores enteros [inicio, fin], donde:
      - inicio: indica la posición (índice) del primer carácter de la instrucción dentro de la cadena completa analizada.
      - fin: indica la posición inmediatamente posterior al último carácter de esa misma instrucción.
  -Este rango permite referenciar con precisión el fragmento textual original que dio contexto a la instrucción del gráfico.

R. Omite el texto que dice literalmente #GRAFICA#, ya que este es un valor previo a cada instrucción.

S. Existen dos parametros que se usarán para gráficas de tabla cuando se pida explicitamente un porcentaje o proporción sobre el conteo o suma de una columna:
  - percentage_of: string | null
    - Si se especifica, indica la columna base sobre la cual se calcularán los porcentajes.
  - percentage_colname: string | null
    - Si se especifica, define el nombre de la columna que contendrá los valores porcentuales calculados.
  Nota: Cuando se use distinct_count con percentage_of, por defecto se colocará en porcentaje_of 'Número trabajadores' por defecto ya que este es el nombre que siempre se pone cuando se usa distinct_count en tablas.

T. Para el uso de leyendas se agrega el parámetro legend_col: string | null, se usará sólo si de forma explícita se pide una leyenda en la gráfica, o se pide agregar una columna descriptiva o categórica adicional a la gráfica.
en este parametro se debe especificar la columna que se usará para las leyendas en las gráficas. Adicionalmente, si se especifica este parámetro, se debe asegurar que el parámetro show_legend esté configurado en true para que la leyenda sea visible en la gráfica. Por último si se quiere escoger colores por categoría, se debe usar colors_by_category.
   Nota: Si legend_col es especificado, show_legend debe ser true. adicionalmente, sólo se usa en gráficas de barras, gráficas de barras horizontales y en tablas.

I. Te explico como funciona el parámetro extra_measures:
    - extra_measures: [ { ... }, { ... }, ... ] | null
    - Si se especifica, permite definir medidas adicionales a calcular y mostrar en la gráfica o tabla.
    - Cada objeto dentro del arreglo representa una medida adicional con su propia configuración.
    - Cada medida adicional puede tener los siguientes campos:
        {"name": "nombre_columna_1",
        "conditions_all": [],
        "conditions_any": [],
        "agg": "sum",
        "distinct_on": null,
        "drop_dupes_before_sum": false}
    - name: Nombre de la columna que representará la medida adicional.
    - conditions_all: Condiciones que deben cumplirse (AND) para incluir datos en esta medida.
    - conditions_any: Condiciones alternativas (OR) para incluir datos en esta medida.
    - agg: Tipo de agregación a aplicar (sum, count, mean, etc.) para esta medida.
    - distinct_on: Columna(s) para conteo distinto, si aplica.
    - drop_dupes_before_sum: Indica si se deben eliminar duplicados antes de sumar, si aplica.
    - La forma de llamarlo será si de forma explicita se pide en la instrucción una o varias medidas adicionales a calcular y mostrar en la gráfica o tabla, con filtros especificos por medida.
 
J. Si de forma explícita se pide ocultar las medidas originales en la gráfica o tabla, se debe usar el parámetro hide_main_measure: true | false | null
    - Si se especifica true, las medidas originales no se mostrarán en la gráfica o tabla.
    - Si se especifica false, las medidas originales se mostrarán junto con las medidas adicionales.
    - Por defecto, si no se especifica, se asume false (mostrar medidas originales).

K. Devuelve el span [start,end] exacto de cada instrucción en el texto original.
    - start: posición del primer carácter.
    - end: posición inmediatamente posterior al último carácter.
    - no inventes span, sólo devuelve el valor correspondiente al idx de cada instrucción.
  

 ESQUEMA DE SALIDA (params)
 - Devolver exclusivamente los parametros indicados en este esquema, no devolver nada por fuera de esta estructura, no inventes columnas a menos que estén explicitamente indicadas en {COLUMNAS_JSON}.
{
  "chart_type": "...",
  "function_name": "...",
  "title": "...",
  "xlabel": string | [string] | null,
  "y": string | [string] | null,
  "agg": "...",
  "distinct_on": string | [string] | null,
  "drop_dupes_before_sum": true | false | null,
  "unique_by": string | [string] | null,
  "conditions_all": [...],
  "conditions_any": [...],
  "binning": { ... } | null,
  "stack_columns": { ... } | null,
  "color": string | [string] | null,
  "colors_by_category": { ... } | null,
  "legend_col": string | null,
  "colors_by_category": { ... } | null,
  "show_legend": true | false | null,
  "show_values": true | false | null,
  "sort": { ... } | null,
  "limit_categories": number | null,
  "needs_disambiguation": true | false,
  "candidates": { "xlabel": [...], "y": [...] }
  "percentage_of": string | null,
  "percentage_colname": string | null
  "extra_measures": [ { ... }, { ... }, ... ] | null
  "hide_main_measure": true | false | null
}

Para SINGLE con varias gráficas o BATCH, siempre devolver:

[
  {"idx": <id_o_orden>, "params": {...}, span: [start,end]},
  ...
]

start: corresponde a la posición inicial.
end: corresponde al número de caracteres final de la instrucción original.
NO DEVOLVER NADA FUERA DEL/LOS JSON.

SINÓNIMOS Y PATRONES ÚTILES

“conteo único de (personas|registros|identificación|id|documento)”
→ "agg": "distinct_count", "distinct_on": "<col identificadora>"

“clasificación por / según / dividido por / por categoría”
→ asigna xlabel

“por X y Y”
→ multi-X: xlabel = ["X","Y"]

“solo 2025”, “estado activo”, “categoría A”
→ condiciones → conditions_all

“top N”, “primeros N”, “mayores N”
→ sort + limit_categories

“apilar / stack / unir varias columnas en un eje x”
→ stack_columns

“mostrar valores / etiquetas / sin leyenda”
→ show_values, show_legend

EJEMPLO CLAVE

Entrada:

# Gráfica de tabla con el nombre 'Espirometria' con el conteo único de personas por identificación con la clasificación de xlabel de la columna de Espirometria #


Salida:

{
  "chart_type": "tabla",
  "function_name": "graficar_tabla",
  "title": "Espirometria",
  "xlabel": "Espirometria",
  "y": "identificacion",
  "agg": "distinct_count",
  "distinct_on": "identificacion",
  "drop_dupes_before_sum": false,
  "unique_by": null,
  "conditions_all": [],
  "conditions_any": [],
  "binning": null,
  "stack_columns": null,
  "color": null,
  "colors_by_category": null,
  "legend_col": null,
  "colors_by_category": null,
  "show_legend": false,
  "show_values": false,
  "sort": null,
  "limit_categories": null,
  "needs_disambiguation": false,
  "candidates": { "xlabel": [], "y": [] }
  "porcentage_of": null,
  "percentage_colname": null,
  "extra_measures": null
  "hide_main_measure": null
}

Ejemplo 2:

Entrada:
# Gráfica de Tablas llamada 'Tipo de riesgo' con un conteo de registros únicos de identificación donde incluya en x las columnas de riesgo_ergonomico = si o riesgo_quimico = si o riesgo_psicosocial = si o riesgo_biomecanico = si#

Salida:
{
      "chart_type": "tabla",
      "function_name": "graficar_tabla",
      "title": "Tipo de riesgo",
      "xlabel": "tipo_riesgo",
      "y": "documento",
      "agg": "distinct_count",
      "distinct_on": "documento",
      "drop_dupes_before_sum": False,
      "unique_by": None,
      "conditions_all": [],
      "conditions_any": [],
      "binning": None,
      "stack_columns": {
        "columns": ["riesgo_ergonomico", "riesgo_quimico", "riesgo_psicosocial", "riesgo_biomecanico"],
        "output_col": "tipo_riesgo",
        "value_col": None,
        "keep_value": "si",
        "label_map": None
      },
      "color": None,
      "legend_col": null,
      "colors_by_category": None,
      "show_legend": None,
      "show_values": None,
      "sort": None,
      "limit_categories": None,
      "needs_disambiguation": False,
      "candidates": {
        "xlabel": [],
        "y": []
      }
      "porcentage_of": null,
      "percentage_colname": null
    }
    
Si hubiera varias instrucciones:

[
  {"idx": 1, "params": {...}, span: [start,end]},
  {"idx": 2, "params": {...}, span: [start,end]},
]

FIN. SOLO JSON.

Ejecución: Con base en la siguiente {INSTRUCCION} y las columnas {COLUMNAS_JSON}, interpreta y devuelve los parámetros técnicos en JSON.
"""

def _strip_code_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        # remove first fence line
        s = s.split("```", 2)
        if len(s) >= 3:
            # s = ["", "json or lang", "body..."]
            return s[2].strip()
        return s[-1].strip()
    return s

def _json_loads_loose(s: str) -> Any:
    s = _strip_code_fences(s)
    try:
        return json.loads(s)
    except Exception:
        # intento simple: localizar el primer '[' o '{' y recortar
        first = min((i for i in [s.find("["), s.find("{")] if i != -1), default=-1)
        last = max(s.rfind("]"), s.rfind("}"))
        if first != -1 and last != -1 and last > first:
            return json.loads(s[first:last+1])
        raise

@register("graficos_gpt5")
def graficos_gpt5(df, pregunta: Union[str, List[Dict[str, Any]]]) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Retro-compatible:
    - Si `pregunta` es str => devuelve UN dict de params.
    - Si `pregunta` es list[{'id':int,'instruccion':str}] => devuelve
      UNA lista [{'id':int,'params':{...}}, ...] en el MISMO orden.
    """
    columnas = df.columns.tolist()
    payload_cols = json.dumps(columnas, ensure_ascii=False)
    instruccion_tipo = json.dumps(pregunta, ensure_ascii=False)
    subprompt = _MSJ_GRAFO_V2.replace("{COLUMNAS_JSON}", payload_cols).replace("{INSTRUCCION}", str(instruccion_tipo))
    time.sleep(1)
    respuesta = client.chat.completions.create(
    model="gpt-5",  # 👈 Aquí usas GPT-5 directamente
    messages=[
            {"role": "system", "content": "Eres un experto en análisis de datos y tu trabajo es interpretar textos y extraer las instrucciones precisas de acuerdo a las columnas de un dataframe"},
            {"role": "user", "content": subprompt}
        ]
    )

    texto_respuesta = respuesta.choices[0].message.content

    return texto_respuesta



MSJ_OPS = """
Eres un analista de datos. A partir de una o varias instrucciones en español y una lista de columnas de un DataFrame de pandas,
debes devolver EXCLUSIVAMENTE un arreglo JSON válido con especificaciones de MÚLTIPLES operaciones a ejecutar sobre el DataFrame.
Cada instrucción se entregará em un arreglo tipo json como se muestra a continuación y cada objeto debe interpretarse de forma independiente.

ENTRADA JSON
{INSTRUCCION}
El JSON debe tendrá esta forma:

[
  {"idx": 1, "prompt": { ... }, span: [start,end]},
  {"idx": 2, "prompt": { ... }, span: [start,end]},
  ...
]

Interpreta cada instrucción y desglósala en una o más operaciones.
Usa las columnas provistas globalmente: {COLUMNAS_JSON}.

---
## INSTRUCCIONES DE INTERPRETACIÓN

1. Cada bloque "prompt" representa una instrucción independiente y debe generar un objeto:
   {"idx": <número>, "prompt": { ...estructura anterior... }}
Nota: Unicamente procesa el texto dentro del parametro "prompt".
      - No te saltes ningún objeto que esté bien formado.
      - No agregues objetos adicionales que no estén en la entrada.
      - No modifiques el orden de los bloques.

2. El índice `idx` debes almacenarlos secuencialmente (1, 2, 3, ...).

3. En modo SINGLE (un texto con varias instrucciones o parametros):
   devuelve un arreglo con todos los objetos en orden de aparición y creales un `idx` secuencial (1, 2, 3, ...).

4. En modo BATCH (si la entrada ya es un arreglo con prompts e ids):
   conserva el mismo `idx` y orden de los elementos.

5. Usa las columnas del DataFrame provistas en {COLUMNAS_JSON} para mapear nombres.
  - Interpreta nombres con coincidencia case-insensitive y acentos-insensitive.
    - Si hay ambigüedad o no existe, deja el campo en null y marca `"needs_disambiguation": true`, proponiendo alternativas en `"candidates"`.
  -No repitas la lista de columnas dentro de cada objeto.
  - Usa unicamente las columnas que estén en {COLUMNAS_JSON}. No inventes columnas nuevas. 

6. Todos los valores nulos, verdaderos y falsos deben expresarse como JSON válido:
   - null → null  
   - true → true  
   - false → false  

7. Si se detecta ambigüedad, deja `"needs_disambiguation": true` e incluye `"candidates"` con alternativas.

8. NO devuelvas texto adicional, explicaciones, comentarios ni fences Markdown.
   SOLO JSON válido UTF-8 con el arreglo final.

9. Interpreta la instrucción y desglósala en una o más operaciones. 
   Si el usuario pide “por región” o “por categoría”, usa "group_by" con esos nombres de columnas.

10. Condiciones: usa una lista de tuplas/objetos con (columna, operador, valor). Operadores soportados: 
   ">", "<", "==", "!=", ">=", "<=", "in", "not in". 
   Para "in"/"not in" el valor debe ser lista.

11. Para "count", por defecto cuenta NO nulos en la columna indicada. Si se requiere contar nulos, agrega "count_nulls": true.

12. Para "avg" o "sum", convierte a numérico implícitamente (coerción), ignorando NaN (equivalente a skipna=true).

13. Al mapear frases del usuario:
    - “personas únicas”, “sin duplicados”, “únicos por …” → usar distinct_count o añadir dedupe_by a la operación.
    - “sumatoria única”, “sumar una vez por …” → usar distinct_sum o sum con dedupe_by.

14. Columnas → "xlabel" (categórica) y "y" (numérica o lista de numéricas)
   - Emparejamiento case-insensitive y acentos-insensitive contra ‘columnas’.
   - “por <col>” implica `<col>` en el eje X → `xlabel`.
   - **Multi-X**: si el usuario pide agrupar por varias columnas (p. ej. área + sede), permite `"xlabel": ["area","sede"]` (las funciones combinarán internamente).
   - Si NO se indica `y`, selecciona una numérica razonable; si no es posible, pon `"y": null` y `"needs_disambiguation": true`.
   - Si NO se indica `xlabel`, elige una no numérica razonable; si no es posible, `"xlabel": null` y `"needs_disambiguation": true`.

15. Filtros condicionales (bloques AND/OR)
   - Usa **dos** campos:
     - `"conditions_all"`: lista de condiciones combinadas con AND.
     - `"conditions_any"`: lista de **bloques** combinados con OR. Cada ítem puede ser:
       - una condición única `["col","op","valor"]`, o
       - un bloque AND `[[...],[...]]`.
   - Operadores soportados: `">","<","==","!=","?>=","<=","in","not in"`.
     - Para `"in"/"not in"` el valor debe ser **lista**.
   - Rangos del tipo “18.5 ≤ IMC ≤ 24.9” se expresan como **dos** condiciones en el mismo bloque.

16. Devuelve el span [start,end] exacto de cada instrucción en el texto original.
  - start: posición del primer carácter.
  - end: posición inmediatamente posterior al último carácter.

---

Dentro de cada `params`, usa exactamente la siguiente estructura (idéntica a la del esquema de salida original).

---

SALIDA JSON
Devuelve siempre para un conjunto de instrucciones:
[
  {"idx": <número>, "params": { ...estructura... }, span: [start,end]},
  ...
]

## FORMATO DE CADA ELEMENTO (params)

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

      "dedupe_by": ["colA","colB"],         // opcional
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

## EJEMPLO MULTIPLE

Entrada:

|| Suma de Ventas y conteo de Pedidos para la Categoría B ||
|| Promedio ponderado del Precio por Producto según cantidad vendida ||

Salida esperada:

[
  {"idx": 1, "params": {
    "operations": [
      {"op":"sum",   "column":"Ventas",   "conditions":[["Categoria","==","B"]], "alias":"ventas_B"},
      {"op":"count", "column":"PedidoID", "conditions":[["Categoria","==","B"]], "alias":"n_pedidos_B"}
    ],
    "group_by": null,
    "needs_disambiguation": false,
    "candidates": { "columns": [], "group_by": [], "by_operation": [] }
  }},
  {"idx": 2, "params": {
    "operations": [
      {"op":"weighted_avg", "column":"Precio", "weights":"Cantidad", "alias":"promedio_ponderado_precio"}
    ],
    "group_by": "Producto",
    "needs_disambiguation": false,
    "candidates": { "columns": [], "group_by": [], "by_operation": [] }
  }}
]

FIN. SOLO JSON.

Ejecución: Con base en la siguiente {INSTRUCCION} y las columnas {COLUMNAS_JSON}, interpreta y devuelve los parámetros técnicos en JSON.
"""

@register("operaciones_gpt5")
def operaciones_gpt5(df, pregunta):
    """Envía un prompt y devuelve la respuesta de GPT-5."""
    # Extraer nombres de columnas
    columnas = df.columns.tolist()
    payload_cols = json.dumps(columnas, ensure_ascii=False)
    instruccion = json.dumps(pregunta, ensure_ascii=False)
    subprompt = (MSJ_OPS.replace("{COLUMNAS_JSON}", payload_cols).replace("{INSTRUCCION}", str(instruccion)))
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


clasificador = """Eres un clasificador determinista por reglas.
Tu tarea es asignar exactamente UNA etiqueta entre las permitidas, usando los criterios recibidos.

Instrucciones:
1) Lee los CRITERIOS (diccionario cuyas claves son las etiquetas permitidas).
2) Lee el REGISTRO (objeto con los campos de una fila).
3) Aplica los criterios con lógica literal (Y/AND, O/OR, NO/NOT, comparaciones, contiene, empieza/termina, igualdad, números, fechas si vienen normalizadas).
4) Si varios criterios coinciden, gana el que aparezca PRIMERO en el orden de las claves recibidas.
5) Si ninguno coincide, asigna el último criterio que sea explícitamente “resto/caso por defecto”; si no existe, asigna la ÚLTIMA clave de la lista.
6) NO EXPLIQUES. NO DES FORMATO. NO AGREGUES TEXTO EXTRA.
7) SALIDA: devuelve ÚNICAMENTE una de las etiquetas permitidas, exactamente como aparece en las claves de CRITERIOS (sin comillas, sin espacios extra, sin saltos).

Debes ser estricto y consistente. No inventes campos. Si un dato no está, trátalo como “no disponible”.

Ejemplo criterios:
{
  "concepto1": "Se clasifica si 'diagnostico' contiene 'hipertensión' y 'edad' >= 40",
  "concepto2": "Se clasifica si 'imc' >= 30 o 'diagnostico' contiene 'obesidad'",
  "concepto3": "Resto de casos"
}

Respuesta etiqueta:
concepto1

Con base en {Criterios} y el siguiente registro {Registro}, devuelve la etiqueta asignada.
"""

@register("columns_gpt5")
def columns_gpt5(criterios, registro):
    """Envía un prompt y devuelve la respuesta de GPT-5."""
    time.sleep(1)
    respuesta = client.chat.completions.create(
        model="gpt-4.1-mini",  # 👈 Aquí usas GPT-5 directamente
        messages=[
            {"role": "system", "content": "Eres un asistente preciso y coherente con instrucciones de análisis de texto, especificamente hablando de temas relacionados con salud ocupacional."},
            {"role": "user", "content": clasificador.replace("{Criterios}", str(criterios['Criterios'])).replace("{Registro}", str(registro['Registro']))}
        ]
    )

    texto_respuesta = respuesta.choices[0].message.content
    return texto_respuesta

clasificador_batch = """
Eres un clasificador determinista por reglas.
Tu tarea es procesar VARIAS tareas de clasificación o cálculo en un solo lote.
Cada tarea define:
  - una columna de salida,
  - un conjunto de criterios,
  - y las columnas del registro que debe utilizar.

Tu trabajo consiste en:
  → Aplicar los criterios de cada tarea para TODOS los registros.
  → Generar una salida por cada registro para cada tarea.
  → Respetar estrictamente el orden de los registros.

NUEVO FORMATO DE ENTRADA (PAYLOAD):
Recibirás un JSON con dos claves principales: "Tareas" y "Registros" {payload}.
  - "Tareas": lista de objetos. Cada objeto define:
      - "columna": nombre de la columna de salida.  
      - "criterios": diccionario cuyas claves son las etiquetas permitidas y los valores son las reglas o cálculos.
      - "registro_cols": lista de nombres de columnas que el registro debe incluir.
  - "Registros": lista de objetos. Cada objeto define:
      - "idx": índice del registro (entero).
      - "registro": objeto con los campos de una fila (incluye las columnas indicadas en "registro_cols" de las tareas).
Recibirás un JSON con esta estructura:

{
  "Tareas": [
    {
      "columna": "NombreColumnaSalida",
      "criterios": {
          "Etiqueta1": "regla o cálculo",
          "Etiqueta2": "regla o cálculo",
          ...
      },
      "registro_cols": ["col1", "col2", ...]
    },
    ...
  ],
  "Registros": [
    { "idx": 0, "registro": {...} },
    { "idx": 1, "registro": {...} },
    ...
  ]
}

REGLAS GENERALES (se mantienen TODAS tus reglas originales):

1) Para cada tarea:
   - Aplica su conjunto de CRITERIOS literalmente y en orden.
   - Cada criterio puede ser:
        a) una condición de clasificación, o
        b) una instrucción de cálculo.

2) Para cada registro:
   a) Si el criterio describe una condición (“se clasifica si…”):
        - Usa AND/OR/NOT exactamente como estén escritos.
        - Comparación de texto: contiene, empieza, termina, igual.
        - Comparación numérica y de fechas si los datos lo permiten.
        - Si varios criterios aplican, gana el PRIMERO.

   b) Si el criterio describe un cálculo (“Calculo: …”, “Calcular …”):
        - Ejecuta la fórmula EXACTA con los datos del registro.
        - Si hay errores (división por cero, nulos, formato inválido),
          devuelve "0".
        - Si varios criterios aplican, gana el PRIMERO.

   c) Si ningún criterio aplica:
        - Usa el criterio de “resto/caso por defecto” si existe,
        - Si no, usa la ÚLTIMA clave del diccionario de criterios.

   d) La salida por cada registro ES UN SOLO VALOR.

3) NO OMITAS REGISTROS.
4) NO CAMBIES EL ORDEN DE LOS REGISTROS.
5) NO AGREGUES texto adicional fuera del JSON.
6) NO OMÍTAS tareas. Cada tarea debe generar su propia columna.
7) Para cada tarea, genera un resultado por cada registro.

FORMATO DE RESPUESTA ESPERADO (OBLIGATORIO):

{
  "Resultados": {
    "NombreColumna1": [
      {"id": <id_registro>, "etiqueta": <valor>},
      ...
    ],
    "NombreColumna2": [
      {"id": <id_registro>, "etiqueta": <valor>},
      ...
    ]
  }
}

Eres estricto, literal y completamente determinista.

con base en el siguiente payload {payload}, devuelve los resultados en el formato indicado.
No expliques nada. Devuelve únicamente el JSON.
"""


@register("columns_batch_gpt5")
def columns_batch_gpt5(payload):
    """Envía un prompt y devuelve la respuesta de GPT-5."""
    respuesta = client.chat.completions.create(
        model="gpt-4.1-mini",  # 👈 Aquí usas GPT-5 directamente
        messages=[
            {"role": "system", "content": "Eres un asistente preciso y coherente con instrucciones de análisis de texto, especificamente hablando de temas relacionados con salud ocupacional."},
            {"role": "user", "content": clasificador_batch.replace("{payload}", str(payload))}
        ]
    )

    texto_respuesta = respuesta.choices[0].message.content
    params = json.loads(texto_respuesta)
    return params






AG_P = """Eres un agente experto en documentación de salud ocupacional.
Tu tarea es, a partir de una sola cadena de texto que recibirás como entrada, construir una portada y una tabla de contenido en texto plano y analiza si en el texto no hay información o datos, elimina el apendice completo, siguiendo estrictamente estas reglas:

DEBES ANALIZAR CONTENIDO DEL TEXTO INTERNO para determinar si existe o no datos númericos, estadísticos o secciones específicas.
NO DEBES GENERAR UNA SEGUNDA PORTADA.
NO DEBES REPETIR NINGUNA SECCIÓN.
Instrucciones:
1. Genera SOLO UNA portada
2. Luego genera SOLO UNA tabla de contenido
3. Nunca repitas la portada
4. Nunca repitas la tabla de contenido
5. NO GENERES TEXTO DEL DOCUMENTO
6. NO RESUMAS EL DOCUMENTO
7. NO AGREGUES ANÁLISIS MÉDICO
8. NO DUPLIQUES NADA
9. Elimina el apéndice completo si no hay datos numéricos, estadísticos o secciones específicas en el texto de entrada.


1. Formato de SALIDA (siempre texto plano, sin JSON)

Debes devolver SIEMPRE, en este orden:

Portada, usando saltos de línea y estilo similar al siguiente ejemplo:

DIAGNOSTICO DE CONDICIONES DE SALUD POBLACIÓN TRABAJADORA*

EVALUACIONES MEDICAS OCUPACIONALES PERIODICAS PROGRAMADAS

EMPRESA:

[Nombre de la empresa]

RESULTADOS DE EVALUACIONES:

[Desde el dd/mm/aaaa hasta dd/mm/aaaa]

[Nombre de la institución responsable del informe]
[Ciudades donde opera / cobertura]
[URL o nota informativa]

(Respeta los asteriscos para resaltar en cursiva como en el ejemplo.)

Luego, varios saltos de línea y el título:

*TABLA DE CONTENIDO *

y a continuación la tabla de contenido en el siguiente estilo:

Introducción
Marco legal
Objetivos
Objetivo general
Objetivos específicos
Características de la empresa
Metodología
Materiales y métodos
Resultados
1 *PERFIL SOCIODEMOGRAFICO*
1.1 *PIRAMIDE POBLACIONAL*
1.2 *COMPOSICIÓN FAMILIAR*
1.3 *ESTRATO SOCIOECONOMICO*
1.4 *ESCOLARIDAD*
2 *PERFIL HABITOS Y ESTILOS DE VIDA SALUDABLE Y DE RIESGO PARA LA SALUD*
3 *PERFIL LABORAL*
3.1 *CARGO*
3.2 *ANTIGUEDAD EN LA EMPRESA*
3.3 *ANTECEDENTE DE EXPOSICION LABORAL A FACTORES DE RIESGOS OCUPACIONALES*
3.4 *EXPOSICION LABORAL ACTUAL*
3.5 *ANTECEDENTES PATOLÓGICOS OCUPACIONALES*
....

A. Sigue este patrón de numeración:

Títulos de nivel 1 → 1, 2, 3, 4, etc.

Subtítulos de nivel 2 → 3.1, 3.2, 10.1, 10.2, etc.

Subtítulos de nivel 3 → 11.2.1, 11.2.2, etc.

Usa una tabulación o varios espacios entre el número y el título.
Los títulos deben ir entre *…* tal como en el ejemplo.

B. Nunca enumerar las siguientes secciones pero incluyelas en la tabla de contenido:
- Introducción
- Marco legal
- Objetivos
- Objetivo general
- Objetivos específicos
- Características de la empresa
- Metodología
- Materiales y métodos

C. identifica los títulos y subtítulos del texto de entrada y constrúyelos en la tabla de contenido siguiendo las reglas del punto A.
D. Si los titulos ya tienen números, no los agregues, de lo contrario, numéralos siguiendo el patrón del punto A.
2. Cómo detectar la información para la portada

A partir del texto de entrada:

Título principal del informe:

Si hay un encabezado en mayúsculas globales o similar a “DIAGNÓSTICO…”, úsalo.

Si no, construye un título genérico:
DIAGNOSTICO DE CONDICIONES DE SALUD POBLACIÓN TRABAJADORA*

Nombre de la empresa:

Busca patrones como empresa, EMPRESA:, {{nombre_cliente}} o similares.

Si encuentras un nombre claro, úsalo dentro de [ ... ] en la sección EMPRESA.

Si no lo encuentras, escribe [Nombre de la empresa].

Rango de fechas de resultados:

Busca expresiones tipo {{fecha_inicio}}, {{fecha_fin}} o fechas explícitas.

Si las encuentras, construye el texto:
[Desde el dd/mm/aaaa hasta dd/mm/aaaa]
usando el formato más parecido posible a las fechas detectadas.

Si no hay fechas claras, escribe [Rango de fechas de las evaluaciones].

Institución responsable:

Siempre usa el nombre como “Laboratorio Clínico Colmedicos I.P.S S.A.S” o similar, úsalo.


Ciudades / cobertura y URL:

Utilida de forma literal lo siguiente: "Medellín – Bogotá D.C. - Cundinamarca – Rionegro – Cali – Palmira – Red nacional."
Si no, puedes dejar una línea genérica como:
*[Cobertura geográfica]*
Finalmente, usa una URL literal como la siguiente:
www.colmedicos.com

3. Cómo construir la tabla de contenido

A partir del texto de entrada:

Detecta títulos y subtítulos:

Líneas numeradas tipo 1., 2., 3.1, 11.2.3, etc.

Líneas con formato de encabezado claramente identificable (por ejemplo, rodeadas de *…* y con numeración previa).

Respeta el orden en que aparecen en el texto.

Escribelos de forma secuencial, sin saltarte ninguno.

Redactalos de forma exacta, sin modificar palabras, ni agregar ni quitar nada.

Asigna nivel jerárquico:

Si la línea inicia con un solo número (ej. 8.) → Nivel 1 → 8.

Si tiene formato X.Y (ej. 8.1) → Nivel 2 → 8.1.

Si tiene formato X.Y.Z (ej. 11.2.3) → Nivel 3 → 11.2.3.

Texto del título:

Usa el texto del encabezado sin los números ni puntos finales.

Ponlo entre *…*.

Ejemplo: 8.1 PIRAMIDE POBLACIONAL → 8.1 *PIRAMIDE POBLACIONA*.

Gráficos y tablas:

Si detectas secciones específicas para pruebas o gráficos (ej.: 11.3.1 Visiometría, 11.3.2 Optometría), inclúyelas tal cual en la tabla de contenido, respetando su numeración.

Si hay referencias entre corchetes [Visiometría], [Optometría], etc., puedes conservarlas.

No inventes secciones:

Solo construye la tabla de contenido con base en títulos, subtítulos y numeraciones que realmente estén en el texto.

Si algo no existe en el texto, no lo añadas.

4. Estilo general

Usa redacción neutra, formal y clara.

No expliques lo que estás haciendo.

Escribe de forma textual, sin formato especial (negritas, cursivas, etc.), excepto los asteriscos en la portada y tabla de contenido.

La salida debe ser solo la portada y la tabla de contenido, sin comentarios adicionales.

No devuelvas JSON, ni listas, ni marcas de código.

Instrucción final: Con base al {texto} devuelve la portada y la tabla de contenido siguiendo las reglas anteriores, asi mismo enumera los titulos acorde con la tabla de contenido y elimina apendices que no tengan información.
"""

rol1 = """Eres un agente experto en documentación de salud ocupacional.
Tu tarea es, a partir de una sola cadena de texto que recibirás como entrada, construir una portada y una tabla de contenido en texto plano."""

  
@register("portada_gpt5")
def portada_gpt5(texto):
    """Envía un prompt y devuelve la respuesta de GPT-5."""
    subprompt = AG_P.replace("{texto}", texto)
    time.sleep(3)
    respuesta = client.chat.completions.create(
        model="gpt-4.1",  # 👈 Aquí usas GPT-5 directamente
        messages=[
            {"role": "system", "content": rol1},
            {"role": "user", "content": subprompt}
        ]
    )

    texto_respuesta = respuesta.choices[0].message.content
    return texto_respuesta