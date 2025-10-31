
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

from typing import Any, List, Dict, Union
import numpy as np

@register("")
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
    operadores = {
        '>': operator.gt,
        '<': operator.lt,
        '==': operator.eq,
        '!=': operator.ne,
        '>=': operator.ge,
        '<=': operator.le,
        'in': lambda a, b: a.isin(b)
    }

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

    ops = {
        '>': operator.gt,
        '<': operator.lt,
        '==': operator.eq,
        '!=': operator.ne,
        '>=': operator.ge,
        '<=': operator.le,
        'in':  lambda a, b: a.isin(b),
        'not in': lambda a, b: ~a.isin(b),
    }

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

    ops = {
        '>': operator.gt,
        '<': operator.lt,
        '==': operator.eq,
        '!=': operator.ne,
        '>=': operator.ge,
        '<=': operator.le,
        'in': lambda a, b: a.isin(b),
        'not in': lambda a, b: ~a.isin(b),
    }

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

def _coerce_condition_value(series: pd.Series, op: str, val: Any):
    """Intenta convertir el valor de condición al tipo compatible con la serie."""
    if op in ("in", "not in"):
        if not isinstance(val, (list, tuple, set)):
            val = [val]
        # casteo suave por tipo de la serie
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
    raise ValueError(f"function_name no soportado: {fn}")



import operator
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Union

# --- util: construir máscara desde condiciones ---
_OPS = {
    '>': operator.gt,
    '<': operator.lt,
    '==': operator.eq,
    '!=': operator.ne,
    '>=': operator.ge,
    '<=': operator.le,
    'in':  lambda a, b: a.isin(b),
    'not in': lambda a, b: ~a.isin(b),
}

def _mask_from_conditions(df: pd.DataFrame, condiciones: List[List[Any]]) -> pd.Series:
    if not condiciones:
        return pd.Series(True, index=df.index)
    m = pd.Series(True, index=df.index)
    for col, op, val in condiciones:
        if col not in df.columns:
            raise ValueError(f"Columna en condición no existe: {col}")
        if op not in _OPS:
            raise ValueError(f"Operador no soportado: {op}")
        m &= _OPS[op](df[col], val)
    return m

# --- métricas atómicas ---
def _metric_on_series(s: pd.Series, op: str, *, count_nulls=False, weights=None, safe_div0=None):
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
        # weights es otra serie alineada
        s = pd.to_numeric(s, errors="coerce")
        w = pd.to_numeric(weights, errors="coerce")
        num = (s * w).sum(skipna=True)
        den = w.sum(skipna=True)
        if den == 0:
            return np.nan if safe_div0 is None else float(safe_div0)
        return float(num / den)
    raise ValueError(f"Operación no soportada: {op}")

# --- ejecutor principal ---
def ejecutar_operaciones_condicionales(
    df: pd.DataFrame,
    spec: Dict[str, Any],          # JSON del subprompt
) -> Union[pd.DataFrame, Dict[str, Any]]:
    """
    Ejecuta múltiples operaciones condicionales.
    Si 'group_by' es None -> retorna dict {alias: valor}
    Si 'group_by' es lista -> retorna DataFrame con columnas group_by + resultados
    """
    operations = spec.get("operations", [])
    group_by = spec.get("group_by")

    if not operations:
        return {} if not group_by else pd.DataFrame()

    # --- sin group_by: resolver operación por operación ---
    if not group_by:
        out = {}
        for op_spec in operations:
            op = op_spec["op"]
            alias = op_spec.get("alias", f"{op}_result")
            if op in ("ratio", "weighted_avg"):
                if op == "ratio":
                    num = op_spec["numerator"]
                    den = op_spec["denominator"]
                    num_mask = _mask_from_conditions(df, num.get("conditions", []))
                    den_mask = _mask_from_conditions(df, den.get("conditions", []))
                    num_val = _metric_on_series(pd.to_numeric(df.loc[num_mask, num["column"]], errors="coerce"), "sum")
                    den_val = _metric_on_series(pd.to_numeric(df.loc[den_mask, den["column"]], errors="coerce"), "sum")
                    if den_val == 0:
                        out[alias] = np.nan if op_spec.get("safe_div0") is None else float(op_spec["safe_div0"])
                    else:
                        out[alias] = float(num_val / den_val)
                elif op == "weighted_avg":
                    base_mask = _mask_from_conditions(df, op_spec.get("conditions", []))
                    values = df.loc[base_mask, op_spec["column"]]
                    weights = df.loc[base_mask, op_spec["weights"]]
                    out[alias] = _metric_on_series(values, "weighted_avg", weights=weights, safe_div0=op_spec.get("safe_div0"))
            else:
                # operaciones simples sobre una serie filtrada
                mask = _mask_from_conditions(df, op_spec.get("conditions", []))
                serie = df.loc[mask, op_spec["column"]]
                out[alias] = _metric_on_series(
                    serie, op,
                    count_nulls=op_spec.get("count_nulls", False)
                )
        return out

    # --- con group_by: calcular por grupo ---
    if isinstance(group_by, str):
        group_by = [group_by]
    for g in group_by:
        if g not in df.columns:
            raise ValueError(f"Columna de group_by no existe: {g}")

    # Construimos un DataFrame resultado por grupos
    grupos = df.groupby(group_by, dropna=False)
    filas = []
    for gkey, gdf in grupos:
        fila = {}
        # clave de grupo -> dict
        if isinstance(gkey, tuple):
            for col, val in zip(group_by, gkey):
                fila[col] = val
        else:
            fila[group_by[0]] = gkey

        # aplicar cada op sobre gdf
        for op_spec in operations:
            op = op_spec["op"]
            alias = op_spec.get("alias", f"{op}_result")
            if op == "ratio":
                num = op_spec["numerator"]
                den = op_spec["denominator"]
                num_mask = _mask_from_conditions(gdf, num.get("conditions", []))
                den_mask = _mask_from_conditions(gdf, den.get("conditions", []))
                num_val = _metric_on_series(pd.to_numeric(gdf.loc[num_mask, num["column"]], errors="coerce"), "sum")
                den_val = _metric_on_series(pd.to_numeric(gdf.loc[den_mask, den["column"]], errors="coerce"), "sum")
                if den_val == 0:
                    fila[alias] = np.nan if op_spec.get("safe_div0") is None else float(op_spec["safe_div0"])
                else:
                    fila[alias] = float(num_val / den_val)
            elif op == "weighted_avg":
                base_mask = _mask_from_conditions(gdf, op_spec.get("conditions", []))
                values = gdf.loc[base_mask, op_spec["column"]]
                weights = gdf.loc[base_mask, op_spec["weights"]]
                fila[alias] = _metric_on_series(values, "weighted_avg", weights=weights, safe_div0=op_spec.get("safe_div0"))
            else:
                mask = _mask_from_conditions(gdf, op_spec.get("conditions", []))
                serie = gdf.loc[mask, op_spec["column"]]
                fila[alias] = _metric_on_series(
                    serie, op,
                    count_nulls=op_spec.get("count_nulls", False)
                )
        filas.append(fila)

    return pd.DataFrame(filas)
