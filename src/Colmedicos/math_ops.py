# --- Standard library ---
import os
import re
import json
import time
import base64
import html
import operator
import tempfile
import webbrowser
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union, Optional, Callable, Literal

# --- Third-party ---
import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype, is_datetime64_any_dtype
import matplotlib.pyplot as plt
from matplotlib._pylab_helpers import Gcf
import plotly.express as px
import openai  # (o reemplaza por: from openai import OpenAI)

# --- Project-local ---
from Colmedicos.registry import register

_ALLOWED_OPS = {
    '>', '<', '==', '!=', '>=', '<=',
    'in', 'not in',
    'contains', 'icontains', 'startswith', 'endswith', 'like', 'regex',
    'between', 'between_open',
}

_OPS = {
    '>': operator.gt,
    '<': operator.lt,
    '==': operator.eq,
    '!=': operator.ne,
    '>=': operator.ge,
    '<=': operator.le,
    'in':  lambda a, b: a.isin(b),
    'not in': lambda a, b: ~a.isin(b),

    # --- Nuevos operadores matemáticos con soporte textual ---
    'contains':   lambda a, s: a.astype(str).str.contains(str(s), case=False, na=False),
    'startswith': lambda a, s: a.astype(str).str.startswith(str(s), na=False),
    'endswith':   lambda a, s: a.astype(str).str.endswith(str(s), na=False),

    # LIKE estilo SQL
    'like': lambda a, pat: a.astype(str).str.contains(
        str(pat).replace("%", ""), case=False, na=False
    ),

    # regex
    'regex': lambda a, pat: a.astype(str).str.contains(
        pat, regex=True, na=False
    ),
}




@register("suma_condicional_multiple")
def suma_condicional_multiple(df, columna_suma, condiciones):
    """
    Suma valores de 'columna_suma' cumpliendo múltiples condiciones.
    
    condiciones: lista de tuplas (columna, operador, valor)
    
    Ejemplo:
    suma_condicional_multiple(df, 'Ventas', [
        ('Costos', '>', 100),
        ('Categoria', '==', 'B')
    ])
    """
    operadores = _OPS

    mascara = pd.Series(True, index=df.index)
    for col, op, val in condiciones:
        mascara &= operadores[op](df[col], val)
    
    return df.loc[mascara, columna_suma].sum()

def conteo_condicional_multiple(
    df: pd.DataFrame,
    columna_conteo: str,
    condiciones,
    contar_nulos: bool = False,
) -> int:

    if columna_conteo not in df.columns:
        raise ValueError(f"La columna '{columna_conteo}' no existe en el DataFrame.")

    ops = _OPS

    # Construir máscara condicional
    mascara = pd.Series(True, index=df.index)
    for col, op, val in condiciones:
        if col not in df.columns:
            raise ValueError(f"La columna de condición '{col}' no existe.")
        if op not in ops:
            raise ValueError(f"Operador no soportado: {op}")
        mascara &= ops[op](df[col], val)

    # Filtro por condiciones
    df_filtrado = df.loc[mascara]

    # Conteo sobre 'columna_conteo'
    if contar_nulos:
        return int(df_filtrado[columna_conteo].shape[0])  # cuenta filas (incluye NaN)
    else:
        return int(df_filtrado[columna_conteo].notna().sum())  # solo no nulos


from typing import List, Tuple, Any, Union
def promedio_condicional_multiple(
    df: pd.DataFrame,
    columna_promedio: str,
    condiciones: List[Tuple[str, str, Any]],
    skipna: bool = True,                 # True: ignora NaN al promediar
    coaccionar_a_numerico: bool = True,  # True: intenta convertir a numérico
    default_si_vacio: Union[float, None] = None,  # valor a retornar si no hay datos (None -> NaN)
) -> float:
    """
    Calcula el promedio de 'columna_promedio' cumpliendo múltiples condiciones.

    condiciones: lista de tuplas (columna, operador, valor)
    Operadores soportados: '>', '<', '==', '!=', '>=', '<=', 'in', 'not in'

    - skipna: si True, ignora NaN en el promedio.
    - coaccionar_a_numerico: si True, aplica pd.to_numeric(errors='coerce') a la columna.
    - default_si_vacio: si, tras filtrar/limpiar, no hay valores válidos, retorna este valor
      (por defecto None -> NaN).
    """
    if columna_promedio not in df.columns:
        raise ValueError(f"La columna '{columna_promedio}' no existe en el DataFrame.")

    ops = _OPS

    # Construir máscara condicional
    mascara = pd.Series(True, index=df.index)
    for col, op, val in condiciones:
        if col not in df.columns:
            raise ValueError(f"La columna de condición '{col}' no existe.")
        if op not in ops:
            raise ValueError(f"Operador no soportado: {op}")
        mascara &= ops[op](df[col], val)

    # Filtrar
    serie = df.loc[mascara, columna_promedio]

    # Convertir a numérico si se pide (útil si hay strings con números)
    if coaccionar_a_numerico:
        serie = pd.to_numeric(serie, errors='coerce')

    # Si no hay valores válidos después de ignorar NaN
    if skipna:
        serie_validos = serie.dropna()
        if serie_validos.empty:
            return float('nan') if default_si_vacio is None else default_si_vacio
        return float(serie_validos.mean(skipna=True))
    else:
        # Si no saltamos NaN y hay NaN, el resultado puede ser NaN
        if serie.empty:
            return float('nan') if default_si_vacio is None else default_si_vacio
        val = serie.mean(skipna=False)
        if pd.isna(val) and default_si_vacio is not None:
            return default_si_vacio
        return float(val)

def suma_columna(df, columna):
    return df[columna].sum()

def conteo_columna(df, columna):
    return df[columna].count()

def promedio_columna(df, columna):
    return df[columna].mean()

def mediana_columna(df, columna):
    return df[columna].median()

def minimo_columna(df, columna):
    return df[columna].min()

def maximo_columna(df, columna):
    return df[columna].max()

def desviacion_columna(df, columna):
    return df[columna].std()

def varianza_columna(df, columna):
    return df[columna].var()

def coeficiente_variacion(df, columna):
    return (df[columna].std() / df[columna].mean()) * 100

def rango_columna(df, columna):
    return df[columna].max() - df[columna].min()

def porcentaje_participacion(df, columna):
    total = df[columna].sum()
    return (df[columna] / total) * 100

def distinct_sum(
    df: pd.DataFrame,
    value_column: str,
    distinct_by: str,
    mask: pd.Series = None
) -> float:
    """
    Suma value_column luego de quitar duplicados por distinct_by (keep='first').
    Aplica opcionalmente una máscara condicional previa.
    """
    if mask is None:
        mask = pd.Series(True, index=df.index)
    if distinct_by not in df.columns:
        raise ValueError(f"distinct_by no existe: {distinct_by}")
    if value_column not in df.columns:
        raise ValueError(f"value_column no existe: {value_column}")

    sub = df.loc[mask, [distinct_by, value_column]].drop_duplicates(subset=[distinct_by], keep='first')
    return float(pd.to_numeric(sub[value_column], errors='coerce').sum())

def distinct_count_by(
    df: pd.DataFrame,
    distinct_by: str,
    mask: pd.Series = None,
    dropna: bool = True
) -> int:
    """
    Conteo de entidades únicas según distinct_by, con máscara opcional.
    """
    if mask is None:
        mask = pd.Series(True, index=df.index)
    if distinct_by not in df.columns:
        raise ValueError(f"distinct_by no existe: {distinct_by}")
    vals = df.loc[mask, distinct_by]
    return int(vals.dropna().nunique() if dropna else vals.nunique(dropna=False))


# --- Operadores soportados ---
# _OPS = {
#     '>': operator.gt,
#     '<': operator.lt,
#     '==': operator.eq,
#     '!=': operator.ne,
#     '>=': operator.ge,
#     '<=': operator.le,
#     'in':  lambda a, b: a.isin(b),
#     'not in': lambda a, b: ~a.isin(b),

#     # Nuevos
#     'between': lambda a, rng: (pd.to_numeric(a, errors='coerce') >= rng[0]) & (pd.to_numeric(a, errors='coerce') <= rng[1]),
#     'between_open': lambda a, rng: (pd.to_numeric(a, errors='coerce') > rng[0]) & (pd.to_numeric(a, errors='coerce') < rng[1]),
#     'contains': lambda a, s: a.astype('string').str.contains(s, case=False, na=False),
#     'regex': lambda a, pat: a.astype('string').str.contains(pat, regex=True, na=False),
# }

def _coerce_condition_value(series: pd.Series, op: str, val: Any):
    """
    Convierte el valor de condición al tipo compatible con la serie (numérico si aplica).
    Maneja listas para in/not in y tuplas para between.
    """
    if op in ('in', 'not in'):
        if not isinstance(val, (list, tuple, set)):
            val = [val]
        if np.issubdtype(series.dtype, np.number):
            return [pd.to_numeric(v, errors='coerce') for v in val]
        return list(val)

    if op in ('between', 'between_open'):
        if not isinstance(val, (list, tuple)) or len(val) != 2:
            raise ValueError(f"El operador {op} requiere un rango [min, max].")
        # Coerce numérico
        return (pd.to_numeric(val[0], errors='coerce'), pd.to_numeric(val[1], errors='coerce'))

    # contains / regex no convierten el patrón
    if op in ('contains', 'regex'):
        return val

    if np.issubdtype(series.dtype, np.number):
        return pd.to_numeric(val, errors='coerce')
    return val



def run_operation_from_params(df: pd.DataFrame, params: Dict[str, Any]) -> Union[float, int, pd.Series, Any]:
    """
    Ejecuta la operación indicada en params sobre df y devuelve el resultado.
    """
    fn = params.get("function_name")
    target = params.get("target_column")
    conditions = params.get("conditions") or []
    contar_nulos = params.get("contar_nulos", False)
    skipna = params.get("skipna", True)
    coaccionar = params.get("coaccionar_a_numerico", True)
    default_vacio = params.get("default_si_vacio", None)

    if fn is None:
        raise ValueError("Falta 'function_name' en params.")

    # Validaciones de columnas
    if target is not None and target not in df.columns:
        raise ValueError(f"target_column inexistente: {target}")

    # Normalizar condiciones (coercer valores a tipo de la columna)
    conds_norm = []
    for cond in conditions:
        col = cond["column"]
        op = cond["op"]
        val = cond["value"]
        if col not in df.columns:
            raise ValueError(f"Columna de condición inexistente: {col}")
        val2 = _coerce_condition_value(df[col], op, val)
        conds_norm.append((col, op, val2))

    # Mapear a funciones
    if fn == "suma_columna":
        if target is None: raise ValueError("target_column es requerido.")
        return float(suma_columna(df, target))

    if fn == "conteo_columna":
        if target is None: raise ValueError("target_column es requerido.")
        return int(conteo_columna(df, target))

    if fn == "promedio_columna":
        if target is None: raise ValueError("target_column es requerido.")
        return float(promedio_columna(df, target))

    if fn == "mediana_columna":
        if target is None: raise ValueError("target_column es requerido.")
        return float(mediana_columna(df, target))

    if fn == "minimo_columna":
        if target is None: raise ValueError("target_column es requerido.")
        return minimo_columna(df, target)

    if fn == "maximo_columna":
        if target is None: raise ValueError("target_column es requerido.")
        return maximo_columna(df, target)

    if fn == "desviacion_columna":
        if target is None: raise ValueError("target_column es requerido.")
        return float(desviacion_columna(df, target))

    if fn == "varianza_columna":
        if target is None: raise ValueError("target_column es requerido.")
        return float(varianza_columna(df, target))

    if fn == "coeficiente_variacion":
        if target is None: raise ValueError("target_column es requerido.")
        return float(coeficiente_variacion(df, target))

    if fn == "rango_columna":
        if target is None: raise ValueError("target_column es requerido.")
        return float(rango_columna(df, target))

    if fn == "porcentaje_participacion":
        if target is None: raise ValueError("target_column es requerido.")
        # Devuelve una Serie (% por fila). Puedes sumar, formatear o insertar en texto
        return porcentaje_participacion(df, target)

    # Condicionales múltiples
    if fn == "suma_condicional_multiple":
        if target is None: raise ValueError("target_column es requerido.")
        return float(suma_condicional_multiple(df, target, conds_norm))

    if fn == "conteo_condicional_multiple":
        if target is None: raise ValueError("target_column es requerido.")
        return int(conteo_condicional_multiple(df, target, conds_norm, contar_nulos=bool(contar_nulos)))

    if fn == "promedio_condicional_multiple":
        if target is None: raise ValueError("target_column es requerido.")
        return float(promedio_condicional_multiple(
            df, target, conds_norm,
            skipna=bool(skipna),
            coaccionar_a_numerico=bool(coaccionar),
            default_si_vacio=default_vacio
        ))
    # Nuevas rutas
    if fn == "suma_condicional_multiple_unique":
        unique_on = params.get("unique_on")
        if not unique_on:
            raise ValueError("unique_on es requerido para suma_condicional_multiple_unique.")
        return float(suma_condicional_multiple_unique(df, target, conditions, unique_on))

    if fn == "conteo_condicional_multiple_unique":
        unique_on = params.get("unique_on")
        if not unique_on:
            raise ValueError("unique_on es requerido para conteo_condicional_multiple_unique.")
        return int(conteo_condicional_multiple_unique(df, unique_on=unique_on, condiciones=conditions))
    raise ValueError(f"function_name no soportado: {fn}")

# --- máscaras condicionales ---
_NUMERIC_OPS = {'>', '<', '>=', '<=', '==', '!='}
_SET_OPS = {'in', 'not in'}

def _to_numeric_series(s: pd.Series) -> pd.Series:
    # limpia comas decimales comunes en CSV
    s2 = s.astype(str).str.replace(',', '.', regex=False)
    return pd.to_numeric(s2, errors='coerce')

def _to_datetime_series(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors='coerce', utc=False)

def _coerce_for_op(series: pd.Series, op: str, val: Any):
    """
    Devuelve (serie_convertida, val_convertido, tipo)
    tipo ∈ {'numeric','datetime','raw'}
    """
    # Si ya es numérica
    if np.issubdtype(series.dtype, np.number):
        v = pd.to_numeric(val, errors='coerce') if op not in _SET_OPS else [
            pd.to_numeric(x, errors='coerce') for x in (val if isinstance(val, (list, tuple, set)) else [val])
        ]
        return (series, v, 'numeric')

    # Si parece fecha o el valor es fecha
    try_dt_series = _to_datetime_series(series)
    # Heurística: si al menos 50% convierte a datetime, lo tratamos como fecha
    if try_dt_series.notna().mean() >= 0.5:
        if op in _SET_OPS:
            vals = val if isinstance(val, (list, tuple, set)) else [val]
            v = [pd.to_datetime(x, errors='coerce', utc=False) for x in vals]
        else:
            v = pd.to_datetime(val, errors='coerce', utc=False)
        return (try_dt_series, v, 'datetime')

    # Intento numérico si la serie es object pero con números como texto
    try_num_series = _to_numeric_series(series)
    if try_num_series.notna().mean() >= 0.5:
        if op in _SET_OPS:
            vals = val if isinstance(val, (list, tuple, set)) else [val]
            v = [pd.to_numeric(str(x).replace(',', '.'), errors='coerce') for x in vals]
        else:
            v = pd.to_numeric(str(val).replace(',', '.'), errors='coerce')
        return (try_num_series, v, 'numeric')

    # Sin conversión: trabajar en bruto (strings)
    if op in _SET_OPS:
        v = list(val) if isinstance(val, (list, tuple, set)) else [val]
    else:
        v = val
    return (series.astype(str), v, 'raw')

def _apply_op(series: pd.Series, op: str, val: Any) -> pd.Series:
    # --- operadores set ---
    if op == 'in':
        return series.isin(val)
    if op == 'not in':
        return ~series.isin(val)

    # --- operadores textuales nuevos ---
    if op in ("contains", "icontains", "startswith", "endswith", "like", "regex"):
        return _OPS[op](series, val)

    # --- operadores numéricos tradicionales ---
    OP = {
        '>': operator.gt, '<': operator.lt, '>=': operator.ge,
        '<=': operator.le, '==': operator.eq, '!=': operator.ne
    }[op]

    out = OP(series, val)
    return out.fillna(False)


def _mask_from_conditions(df: pd.DataFrame, condiciones: List[List[Any]], logic: str = "AND") -> pd.Series:
    """
    condiciones: [["col","op","valor"], ...]
    logic: "AND" o "OR" para combinar las condiciones planas.
    """
    if not condiciones:
        return pd.Series(True, index=df.index)

    masks = []
    for col, op, val in condiciones:
        if col not in df.columns:
            raise ValueError(f"Columna en condición no existe: {col}")
        if op not in _ALLOWED_OPS:
            raise ValueError(f"Operador no soportado: {op}")

        s0 = df[col]
        s, v, _kind = _coerce_for_op(s0, op, val)
        m = _apply_op(s, op, v)
        masks.append(m)

    if logic.upper() == "OR":
        m_all = masks[0].copy()
        for m in masks[1:]:
            m_all = m_all | m
        return m_all
    else:
        # AND por defecto
        m_all = masks[0].copy()
        for m in masks[1:]:
            m_all = m_all & m
        return m_all

def build_mask_with_groups(
    df: pd.DataFrame,
    *,
    conditions=None,                 # condiciones planas
    conditions_logic: str = "AND",
    condition_groups=None            # [{conditions:[...], logic:"AND|OR"}, ...]
) -> pd.Series:
    """
    Soporta:
      - conditions planas con AND/OR
      - condition_groups para expresiones tipo (A AND B) OR (C AND D)
    Si se especifica condition_groups, combina sus máscaras con OR por defecto,
    pudiendo cambiarlo si lo deseas.
    """
    if condition_groups:
        group_masks = []
        for grp in condition_groups:
            grp_conds = grp.get("conditions", [])
            grp_logic = grp.get("logic", "AND")
            group_masks.append(_mask_from_conditions(df, grp_conds, grp_logic))
        # Combina grupos con OR (puedes ajustar a tu necesidad)
        m_all = pd.Series(False, index=df.index)
        for gm in group_masks:
            m_all = m_all | gm
        return m_all

    # Solo condiciones planas
    return _mask_from_conditions(df, conditions or [], conditions_logic)


def _mask_from_condition_blocks(
    df: pd.DataFrame,
    conditions_all: List[List[Any]] = None,
    conditions_any: List[List[Any]] = None
) -> pd.Series:
    """
    Construye una máscara booleana combinando:
      - AND de 'conditions_all'
      - AND por bloque dentro de 'conditions_any', y luego OR entre bloques

    Formatos admitidos (compatibles con tu versión anterior):
      - Condición simple: ["col","op","valor"]
      - Bloque AND: [ ["col","op","valor"], ["col","op","valor"], ... ]

    Requiere que _mask_from_conditions(df, conditions, logic="AND") ya implemente:
      - Coerción numérica/fecha
      - Operadores: >, <, ==, !=, >=, <=, in, not in
      - Manejo de NaN
    """

    # Identidad de AND (todo verdadero)
    base_and = pd.Series(True, index=df.index) if len(df.index) else pd.Series(dtype=bool)

    # AND de todas las condiciones en conditions_all (si vienen)
    if conditions_all:
        m_all = _mask_from_conditions(df, conditions_all, logic="AND")
    else:
        m_all = base_and

    # Si no hay bloque OR, devolvemos el AND puro
    if not conditions_any:
        return m_all

    # OR de los bloques en conditions_any
    #   - cada elemento puede ser una condición simple o un bloque AND
    or_mask = pd.Series(False, index=df.index) if len(df.index) else pd.Series(dtype=bool)

    for cond in conditions_any:
        # ¿Es un bloque AND (lista de condiciones)?
        if isinstance(cond, (list, tuple)) and cond and isinstance(cond[0], (list, tuple)):
            # Bloque AND: cada item debe ser ["col","op","valor"]
            m_blk = _mask_from_conditions(df, cond, logic="AND")
            or_mask |= m_blk
        else:
            # Condición simple: normalízala a lista de 1 condición
            m_single = _mask_from_conditions(df, [cond], logic="AND")
            or_mask |= m_single

    # Resultado final: (ALL) AND (ANY)
    return m_all & or_mask


def suma_condicional_multiple_unique(
    df: pd.DataFrame,
    columna_suma: str,
    condiciones: List[List[Any]],
    unique_on: str
) -> float:
    mask = _mask_from_conditions(df, condiciones or [])
    return distinct_sum(df, value_column=columna_suma, distinct_by=unique_on, mask=mask)

def conteo_condicional_multiple_unique(
    df: pd.DataFrame,
    unique_on: str,
    condiciones: List[List[Any]]
) -> int:
    mask = _mask_from_conditions(df, condiciones or [])
    return distinct_count_by(df, distinct_by=unique_on, mask=mask, dropna=True)

def _metric_on_series(
    s: pd.Series,
    op: str,
    *,
    count_nulls: bool = False,
    weights: pd.Series = None,
    safe_div0=None
):
    if op == "sum":
        return float(pd.to_numeric(s, errors="coerce").sum())
    if op == "count":
        return int(s.shape[0]) if count_nulls else int(s.notna().sum())
    if op == "avg":
        s = pd.to_numeric(s, errors="coerce")
        return float(s.mean(skipna=True))
    if op == "min":
        s = pd.to_numeric(s, errors="coerce")
        return float(s.min(skipna=True))
    if op == "max":
        s = pd.to_numeric(s, errors="coerce")
        return float(s.max(skipna=True))
    if op == "distinct_count":
        return int(s.nunique(dropna=True))
    if op == "weighted_avg":
        s = pd.to_numeric(s, errors="coerce")
        w = pd.to_numeric(weights, errors="coerce")
        num = (s * w).sum(skipna=True)
        den = w.sum(skipna=True)
        if den == 0:
            return np.nan if safe_div0 is None else float(safe_div0)
        return float(num / den)
    raise ValueError(f"Operación no soportada: {op}")


def normalizar_columnas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpia espacios, unicode invisibles y normaliza nombres
    sin alterar la lógica del dataset.
    """
    def clean(col):
        if isinstance(col, str):
            col = col.replace("\xa0", " ")  # NBSP → espacio normal
            col = col.strip()
        return col

    df = df.rename(columns={c: clean(c) for c in df.columns})
    return df


def ejecutar_operaciones_condicionales(
    df: pd.DataFrame,
    spec: Union[Dict[str, Any], List[Dict[str, Any]]],
) -> Union[pd.DataFrame, Dict[str, Any]]:
    """
    Ejecuta operaciones condicionales sobre df.

    `spec` puede ser:
      - dict con claves:    
            {
              "operations": [ ... ],
              "group_by": ["col1", "col2"] | "col"
            }
      - o directamente una lista de operaciones:
            [ {...}, {...} ]
        (en este caso se asume que NO hay group_by)
    """
    df = normalizar_columnas(df)
    # --- Normalización de spec ---
    if isinstance(spec, list):
        # spec ya es la lista de operaciones
        operations = spec
        group_by = None
    elif isinstance(spec, dict):
        operations = spec.get("operations", [])
        group_by = spec.get("group_by")
    else:
        raise TypeError(
            f"'spec' debe ser dict o list, se recibió {type(spec).__name__}"
        )

    if not operations:
        return {} if not group_by else pd.DataFrame()

    def _resolve_mask(_df, op_spec):
        # Compat: 'conditions' (AND) + nuevos 'conditions_all'/'conditions_any'
        conds_all = op_spec.get("conditions_all")
        conds_any = op_spec.get("conditions_any")
        legacy = op_spec.get("conditions")
        if conds_all is None and conds_any is None and legacy is None:
            return pd.Series(True, index=_df.index)
        if conds_all is None and conds_any is None:
            return _mask_from_conditions(_df, legacy or [])
        return _mask_from_condition_blocks(_df, conds_all, conds_any)

    def _apply_ops(_df, op_spec):
        op = op_spec["op"]
        alias = op_spec.get("alias", f"{op}_result")

        if op == "ratio":
            num = op_spec["numerator"]     # {column, conditions_xxx}
            den = op_spec["denominator"]
            num_mask = _resolve_mask(_df, num)
            den_mask = _resolve_mask(_df, den)
            num_val = _metric_on_series(pd.to_numeric(_df.loc[num_mask, num["column"]], errors="coerce"), "sum")
            den_val = _metric_on_series(pd.to_numeric(_df.loc[den_mask, den["column"]], errors="coerce"), "sum")
            if den_val == 0:
                return alias, (np.nan if op_spec.get("safe_div0") is None else float(op_spec["safe_div0"]))
            return alias, float(num_val / den_val)

        if op == "weighted_avg":
            m = _resolve_mask(_df, op_spec)
            vals = _df.loc[m, op_spec["column"]]
            wts  = _df.loc[m, op_spec["weights"]]
            return alias, _metric_on_series(vals, "weighted_avg", weights=wts, safe_div0=op_spec.get("safe_div0"))

        if op == "distinct_sum":
            m = _resolve_mask(_df, op_spec)
            return alias, distinct_sum(_df, op_spec["column"], op_spec["distinct_by"], mask=m)

        if op in ("distinct_count", "distinct_count_by"):
            m = _resolve_mask(_df, op_spec)
            key = op_spec.get("distinct_by", op_spec.get("column"))
            if not key:
                raise ValueError("distinct_count requiere 'column' o 'distinct_by'.")
            return alias, distinct_count_by(_df, key, mask=m, dropna=True)

        # operaciones simples: sum, count, avg, min, max...
        m = _resolve_mask(_df, op_spec)
        serie = _df.loc[m, op_spec["column"]]
        val = _metric_on_series(
            serie, op,
            count_nulls=op_spec.get("count_nulls", False)
        )
        return alias, val

    # --- sin group_by ---
    if not group_by:
        out = {}
        for op_spec in operations:
            alias, val = _apply_ops(df, op_spec)
            out[alias] = val
        return out

    # --- con group_by ---
    if isinstance(group_by, str):
        group_by = [group_by]
    for g in group_by:
        if g not in df.columns:
            raise ValueError(f"Columna de group_by no existe: {g}")

    # ✅ NUEVO: PREFILTRO GLOBAL ANTES DEL GROUPBY
    def _spec_has_conditions(s: dict) -> bool:
        return bool(s.get("conditions") or s.get("conditions_all") or s.get("conditions_any") or s.get("condition_groups"))

    prefilter_specs = []
    for op_spec in operations:
        if op_spec.get("op") == "ratio":
            # ratio tiene numerator/denominator con conditions
            if _spec_has_conditions(op_spec.get("numerator", {})) or _spec_has_conditions(op_spec.get("denominator", {})):
                prefilter_specs.append(op_spec)
        else:
            if _spec_has_conditions(op_spec):
                prefilter_specs.append(op_spec)

    if prefilter_specs:
        m_pref = pd.Series(False, index=df.index)
        for op_spec in prefilter_specs:
            if op_spec.get("op") == "ratio":
                m_pref |= _resolve_mask(df, op_spec["numerator"])
                m_pref |= _resolve_mask(df, op_spec["denominator"])
            else:
                m_pref |= _resolve_mask(df, op_spec)

        df_group = df.loc[m_pref].copy()
    else:
        df_group = df

    # Si quedó vacío, devuelve DF vacío con columnas esperadas
    if df_group.empty:
        cols = list(group_by) + [op.get("alias", op.get("op", "result")) for op in operations]
        return pd.DataFrame(columns=cols)

    filas = []
    for gkey, gdf in df_group.groupby(group_by, dropna=False):
        fila = {}
        if isinstance(gkey, tuple):
            for col, val in zip(group_by, gkey):
                fila[col] = val
        else:
            fila[group_by[0]] = gkey

        for op_spec in operations:
            alias, val = _apply_ops(gdf, op_spec)
            fila[alias] = val

        filas.append(fila)

    return pd.DataFrame(filas)


# ============================================================================
# METADATOS DE PRESENCIA DE DATOS (No invasivo - funciones auxiliares)
# ============================================================================

def _evaluar_presencia_datos(resultado: Union[pd.DataFrame, Dict[str, Any]]) -> Tuple[bool, str]:
    """
    Determina si el resultado tiene datos significativos.

    Criterios:
    - 'Sin datos' = vacío O todos los valores numéricos son 0 o NaN
    - 'Hay datos' = al menos 1 valor numérico > 0

    Returns:
        (hay_datos: bool, resumen: str)
        resumen ∈ {"HAY_DATOS", "SIN_DATOS", "SOLO_CEROS", "ERROR"}
    """
    try:
        if isinstance(resultado, pd.DataFrame):
            if resultado.empty:
                return False, "SIN_DATOS"

            # Revisar columnas numéricas (excluyendo metadatos)
            cols_num = resultado.select_dtypes(include=[np.number]).columns.tolist()
            cols_num = [c for c in cols_num if not str(c).startswith("_")]

            if not cols_num:
                # No hay columnas numéricas - verificar si hay filas
                return (True, "HAY_DATOS") if len(resultado) > 0 else (False, "SIN_DATOS")

            valores = resultado[cols_num].values.flatten()
            valores_validos = valores[~np.isnan(valores)]

            if len(valores_validos) == 0:
                return False, "SIN_DATOS"

            if np.all(valores_validos == 0):
                return False, "SOLO_CEROS"

            return True, "HAY_DATOS"

        elif isinstance(resultado, dict):
            if not resultado:
                return False, "SIN_DATOS"

            # Revisar valores numéricos (excluyendo metadatos)
            valores = []
            for k, v in resultado.items():
                if str(k).startswith("_"):
                    continue
                if isinstance(v, (int, float)):
                    valores.append(v)

            if not valores:
                return False, "SIN_DATOS"

            # Filtrar NaN
            valores_validos = [v for v in valores if not (isinstance(v, float) and np.isnan(v))]

            if not valores_validos:
                return False, "SIN_DATOS"

            if all(v == 0 for v in valores_validos):
                return False, "SOLO_CEROS"

            return True, "HAY_DATOS"

        return False, "ERROR"

    except Exception:
        return False, "ERROR"


def operaciones_con_metadatos(
    df: pd.DataFrame,
    spec: Union[Dict[str, Any], List[Dict[str, Any]]]
) -> Tuple[Union[pd.DataFrame, Dict[str, Any]], Dict[str, Any]]:
    """
    Wrapper de ejecutar_operaciones_condicionales que agrega metadatos.

    NO modifica la lógica original, solo añade información sobre presencia de datos.

    Args:
        df: DataFrame de entrada
        spec: Especificación de operaciones (igual que ejecutar_operaciones_condicionales)

    Returns:
        (resultado_original, metadatos)
        metadatos = {
            "hay_datos": bool,
            "resumen": str,  # "HAY_DATOS", "SIN_DATOS", "SOLO_CEROS", "ERROR"
        }

    Ejemplo:
        resultado, meta = operaciones_con_metadatos(df, spec)
        if not meta["hay_datos"]:
            # Omitir apéndice sin consultar IA
            pass
    """
    resultado = ejecutar_operaciones_condicionales(df, spec)
    hay_datos, resumen = _evaluar_presencia_datos(resultado)

    return resultado, {
        "hay_datos": hay_datos,
        "resumen": resumen
    }


def enriquecer_texto_con_metadatos(
    texto: str,
    resultado: Union[pd.DataFrame, Dict[str, Any]]
) -> str:
    """
    Enriquece un texto con prefijo de metadatos para facilitar interpretación de IA.

    Ejemplo:
        texto_original = "actividad  personas\\n0 Transporte  15"
        texto_enriquecido = "[HAY_DATOS]\\nactividad  personas\\n0 Transporte  15"

    Args:
        texto: Texto original (representación string del resultado)
        resultado: DataFrame o Dict para evaluar presencia de datos

    Returns:
        Texto con prefijo [HAY_DATOS], [SIN_DATOS] o [SOLO_CEROS]
    """
    _, resumen = _evaluar_presencia_datos(resultado)
    return f"[{resumen}]\n{texto}"
