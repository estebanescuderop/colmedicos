# --- Standard library ---
import os
import re
import json
import time
import base64
from io import BytesIO
from pathlib import Path
from collections.abc import Iterable
from typing import Any, Dict, List, Tuple, Union, Optional, Callable

# --- Third-party ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib._pylab_helpers import Gcf

# --- Project-local ---
from Colmedicos.registry import register
from Colmedicos.ia import operaciones_gpt5, graficos_gpt5, columns_gpt5, ask_gpt5,columns_batch_gpt5
from Colmedicos.math_ops import ejecutar_operaciones_condicionales
from Colmedicos.charts import plot_from_params


# ----------------------------
# 1) Reutilizamos tus helpers
# ----------------------------
_VAR_REGEX = re.compile(r"\{\{\s*([A-Za-z_][\w\.]*)\s*\}\}")

@register("_render_vars_text")
def _render_vars_text(texto: str, ctx: Optional[Dict[str, Any]], strict: bool) -> str:
    if not isinstance(texto, str) or not ctx:
        return "" if texto is None else str(texto)

    def repl(m: re.Match) -> str:
        key = m.group(1)
        # soporte para dot-paths a futuro (por ahora directo)
        if key in ctx:
            val = ctx[key]
            return "" if val is None else str(val)
        if strict:
            raise KeyError(f"Variable no definida en ctx: '{key}'")
        return ""
    return _VAR_REGEX.sub(repl, texto)

@register("_format_number")
def _format_number(x: Any, float_fmt: str = "{:,.2f}") -> str:
    if isinstance(x, (int, np.integer)):
        return f"{int(x)}"
    if isinstance(x, (float, np.floating)):
        if np.isnan(x):
            return "NaN"
        return float_fmt.format(float(x))
    return str(x)

@register("_format_result_plain")
def _format_result_plain(
    result: Union[pd.DataFrame, Dict[str, Any], Any],
    float_fmt: str = "{:,.2f}",
    max_rows: int = 200,
    include_index: bool = False,
) -> str:
    if isinstance(result, pd.DataFrame):
        df_show = result.copy()
        for c in df_show.columns:
            if pd.api.types.is_numeric_dtype(df_show[c]):
                df_show[c] = df_show[c].apply(lambda v: _format_number(v, float_fmt))
            elif pd.api.types.is_datetime64_any_dtype(df_show[c]):
                df_show[c] = df_show[c].dt.strftime("%Y-%m-%d")
            else:
                df_show[c] = df_show[c].astype(str).fillna("")
        if len(df_show) > max_rows:
            df_show = df_show.head(max_rows)
        return df_show.to_string(index=include_index)

    if isinstance(result, dict):
        lines = []
        for k, v in result.items():
            if isinstance(v, (int, float, np.integer, np.floating)):
                v_str = _format_number(v, float_fmt)
            else:
                v_str = str(v)
            lines.append(f"{k}: {v_str}")
        return "\n".join(lines) if lines else ""

    return _format_number(result, float_fmt)

# ----------------------------
# 2) Limpieza segura (sin romper data URIs)
# ----------------------------

@register("limpiar_texto_simple")
def limpiar_texto_simple(
    texto: str,
    chars_a_quitar: Optional[List[str]] = None,
    quitar_entre_hash: bool = True,
    reemplazo_entre_hash: str = "",
    entre_hash_multilinea: bool = False,
    normalizar_espacios: bool = True
) -> str:
    if texto is None:
        return texto
    if quitar_entre_hash:
        flags = re.DOTALL if entre_hash_multilinea else 0
        texto = re.sub(r'#.*?#', reemplazo_entre_hash, texto, flags=flags)
    if chars_a_quitar is None:
        chars_a_quitar = ['*', '+', '[', ']']
    tabla = str.maketrans({c: "" for c in chars_a_quitar})
    texto = texto.translate(tabla)
    if normalizar_espacios:
        texto = re.sub(r'\s+', ' ', texto).strip()
    return texto

# ----------------------------
# 3) Regex de bloques numerados
# ----------------------------
# ++IA_1: prompt++   o   ++prompt++
_IA_BLOCK_RE = re.compile(
    r"\+\s*(?:(?:IA_(?P<idx>\d+))\s*:\s*)?(?P<prompt>.*?)\s*\+",
    flags=re.DOTALL | re.IGNORECASE
)

# ||DATOS_1: spec||   o   ||spec||
_DATA_BLOCK_RE = re.compile(
    r"\|\|\s*(?:(?:DATOS_(?P<idx>\d+))\s*:\s*)?(?P<prompt>.*?)\s*\|\|",
    flags=re.DOTALL | re.IGNORECASE
)

# #GRAFICA_1#  o  #GRAFICO_2#  o  #GRAFICA#
_PLOT_BLOCK_RE = re.compile(
    r"#GRAFIC[AO](?:_(?P<idx>\d+))?#",
    flags=re.DOTALL
)

# ----------------------------
# 4) Parseadores
# ----------------------------
def _enumerate_blocks(matches, explicit_idx_group: str) -> List[Dict[str, Any]]:
    results = []
    next_auto = 1
    for m in matches:
        idx_str = m.groupdict().get(explicit_idx_group)
        if idx_str is not None:
            idx = int(idx_str)
        else:
            idx = next_auto
            next_auto += 1
        # Con este _PLOT_BLOCK_RE ya no hay grupo "prompt",
        # así que aquí por defecto quedará vacío. Luego lo intentamos
        # poblar en parse_plot_blocks buscando el bloque #...# siguiente.
        results.append({
            "idx": idx,
            "prompt": "",
            "span": (m.start(), m.end()),
        })
    return results


@register("parse_ia_blocks")
def parse_ia_blocks(text: str) -> List[Dict[str, Any]]:
    """
    Detecta bloques de datos del tipo:
      - ||DATOS_1: ...especificación...||
      - ||...especificación...||
    y devuelve: [{'idx': int, 'prompt': str, 'span': (start, end)}, ...]
    """
    results: List[Dict[str, Any]] = []
    next_auto = 1

    for m in _IA_BLOCK_RE.finditer(text):
        idx_str = m.group("idx")
        if idx_str is not None:
            idx = int(idx_str)
        else:
            idx = next_auto
            next_auto += 1
        prompt = (m.group("prompt") or "").strip()
        results.append({
            "idx": idx,
            "prompt": prompt,
            "span": (m.start(), m.end()),
        })

    return results

@register("parse_data_blocks")
def parse_data_blocks(text: str) -> List[Dict[str, Any]]:
    """
    Detecta bloques de datos del tipo:
      - ||DATOS_1: ...especificación...||
      - ||...especificación...||
    y devuelve: [{'idx': int, 'prompt': str, 'span': (start, end)}, ...]
    """
    results: List[Dict[str, Any]] = []
    next_auto = 1

    for m in _DATA_BLOCK_RE.finditer(text):
        idx_str = m.group("idx")
        if idx_str is not None:
            idx = int(idx_str)
        else:
            idx = next_auto
            next_auto += 1

        prompt = (m.group("prompt") or "").strip()
        results.append({
            "idx": idx,
            "prompt": prompt,
            "span": (m.start(), m.end()),
        })

    return results

@register("parse_plot_blocks")
def parse_plot_blocks(text: str) -> List[Dict[str, Any]]:
    """
    Detecta tokens #GRAFICA(_n)?# / #GRAFICO(_n)?#.
    Si inmediatamente después (solo espacios/saltos) hay un bloque '# ... #',
    lo toma como 'prompt' del gráfico correspondiente SIN consumir otros tokens.
    """
    # 1) Matches de los tokens
    matches = list(_PLOT_BLOCK_RE.finditer(text))
    items = _enumerate_blocks(matches, "idx")
    if not items:
        return items

    # 2) Para cada token, intenta capturar un bloque '# ... #' inmediato
    #    Se evalúa sobre el "remainder" que empieza justo tras el token.
    for it in items:
        start, end = it["span"]
        remainder = text[end:]  # texto justo después del token
        # Intento de match ANCLADO al inicio del remainder: \A
        #m_prompt = _PROMPT_HASH_BLOCK_RE.match(remainder)
        #if m_prompt:
        #    inner = m_prompt.group("inner").strip()
            # Evitar confundir otro marcador de gráfico:
            # si el inner empieza por "GRAFIC" estamos frente a otro token, ignorar
        #   if not inner.upper().startswith("GRAFIC"):
        #       it["prompt"] = inner

    return items

# ----------------------------
# 5) Reemplazo por familias
# ----------------------------

@register("process_ia_blocks")
def process_ia_blocks(
    texto: str,
    ask_fn: Callable[[str], str] = ask_gpt5,
    *,
    max_retries: int = 2,
    retry_delay_sec: float = 1.0,
    on_error: str = "raise"  # "raise" | "return_input"
) -> str:
    """
    Envía TODO el texto a la IA para que redacte el informe completo.
    No identifica marcadores ni hace reemplazos locales: la IA (ask_fn)
    es la encargada de procesar únicamente lo que esté entre '+' según tu lógica.

    Parámetros
    ----------
    texto : str
        Documento completo a procesar.
    ask_fn : Callable[[str], str]
        Función IA (p. ej., ask_gpt5) que recibe el texto y devuelve el redactado final.
        Esta función ya debe saber cómo tratar los segmentos entre '+'.
    max_retries : int
        Reintentos ante errores temporales de la IA.
    retry_delay_sec : float
        Pausa entre reintentos.
    on_error : {"raise", "return_input"}
        - "raise": propaga la excepción si la IA falla tras los reintentos.
        - "return_input": devuelve el texto original si falla.

    Retorna
    -------
    str
        Texto final redactado por la IA.
    """
    last_err: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        try:
            respuesta = ask_fn(texto)
            return (respuesta or "").strip()
        except Exception as e:
            last_err = e
            if attempt < max_retries:
                time.sleep(retry_delay_sec)
            else:
                if on_error == "return_input":
                    return texto
                raise


# ----------------------------
# 5) Función que calcula DATOS.
# ----------------------------

#Función validada
def unir_idx_params_con_span_json_data(
    idx_params_any: Union[str, Iterable],
    texto: str,
    ensure_ascii: bool = False,
    indent: Union[int, None] = None,
) -> str:
    """
    Une (idx, params) con los span detectados en `texto` por parse_plot_blocks y
    devuelve un JSON string con objetos: [{"idx": int, "params": {...}, "span": [s,e] | null}, ...].

    Acepta `idx_params_any` como:
      - str JSON: '[{"idx":1,"params":{...}}, [2, {...}], ...]'
      - lista de dicts con 'idx' y 'params'
      - lista de pares/tuplas [idx, params] o (idx, params)
    """

    # --- normalización mínima a [(idx:int, params:dict)]
    def _normalize(x: Union[str, Iterable]) -> List[Tuple[int, Dict[str, Any]]]:
        if isinstance(x, str):
            try:
                x = json.loads(x)
            except json.JSONDecodeError:
                return []
        if not isinstance(x, Iterable):
            return []

        out: List[Tuple[int, Dict[str, Any]]] = []
        for item in x:
            if isinstance(item, dict) and "idx" in item and "params" in item and isinstance(item["params"], dict):
                try:
                    out.append((int(item["idx"]), item["params"]))
                except Exception:
                    pass
            elif isinstance(item, (list, tuple)) and len(item) >= 2 and isinstance(item[1], dict):
                try:
                    out.append((int(item[0]), item[1]))
                except Exception:
                    pass
        return out

    idx_params = _normalize(idx_params_any)
    if not idx_params:
        return json.dumps([], ensure_ascii=ensure_ascii, indent=indent)

    # --- mapa idx -> span usando parse_plot_blocks(texto)
    spans_by_idx: Dict[int, Any] = {}
    for it in (parse_data_blocks(texto) or []):
        if isinstance(it, dict) and "idx" in it and "span" in it:
            try:
                spans_by_idx[int(it["idx"])] = it["span"]
            except Exception:
                pass

    # --- construir salida como lista de objetos JSON-friendly
    result = []
    for idx, params in idx_params:
        span = spans_by_idx.get(idx, None)
        # Normalizar span a lista de 2 enteros si viene como tupla
        if isinstance(span, tuple):
            span = list(span)
        result.append({"idx": idx, "params": params, "span": span})

    return json.dumps(result, ensure_ascii=ensure_ascii, indent=indent)


def aplicar_operaciones_en_texto(texto: str, resultados_ops, formato: str = "texto") -> str:
    """
    resultados_ops: lista de tuplas (idx, params, span_obj, resultado_formateado)
      - span_obj puede ser:
          a) [start, end]
          b) {"span": [start, end]}  # o {"spam": [start, end]} si hubo typo
    Reemplaza en `texto` cada instrucción ||...|| por el resultado formateado.
    """
    reemplazos = []

    for item in resultados_ops:
        # Saltar entradas de error que pudieran venir como dict {"error":..., "params":...}
        if isinstance(item, dict):
            continue

        try:
            idx, params, span_obj, resultado_fmt = item
        except Exception:
            # Estructura inesperada, ignorar
            continue

        # Normalizar span a (start, end)
        if isinstance(span_obj, dict):
            span_list = span_obj.get("span") or span_obj.get("spam")
        else:
            span_list = span_obj

        if not (isinstance(span_list, (list, tuple)) and len(span_list) == 2):
            continue

        start, end = span_list
        if not (isinstance(start, int) and isinstance(end, int) and 0 <= start <= end <= len(texto)):
            continue

        # Reemplazo según formato
        if formato == "html":
            reemplazo = f'<pre style="background:#f9f9f9;padding:10px;border-radius:8px;">{resultado_fmt}</pre>'
        else:
            # 🔹 Versión de texto plano
            reemplazo = f"\n[Resultado operación {idx}]:\n{resultado_fmt}\n"

        reemplazos.append((start, end, reemplazo))

    # Aplicar de derecha a izquierda para preservar offsets
    reemplazos.sort(key=lambda t: t[0], reverse=True)

    out = texto
    for start, end, rep in reemplazos:
        out = out[:start] + rep + out[end:]

    return out


def _to_base64(s: str) -> str:
    return base64.b64encode(s.encode("utf-8")).decode("ascii")


@register("process_data_blocks")
def process_data_blocks(df: pd.DataFrame, texto: str):
    """
    Réplica del pipeline_graficos_gpt5_final pero para OPERACIONES:

    - operaciones_gpt5(df, texto) -> lista JSON de objetos con {idx, params, ...}
    - unir_idx_params_con_span_json(..., texto) -> adjunta el span correcto
    - ejecutar_operaciones_condicionales(df, params) -> ejecuta la operación
    - _format_result_plain(resultado) -> string final a insertar
    - aplicar_operaciones_en_texto(texto, resultados_ops, formato="html") -> reemplaza ||...||
    """
    # 1) Ejecutar operaciones_gpt5 y asegurar conversión a JSON con spans
    out = operaciones_gpt5(df, texto)
    out = unir_idx_params_con_span_json_data(out, texto)  # mantenemos el mismo unificador de spans

    if isinstance(out, str):
        try:
            out = json.loads(out)
        except json.JSONDecodeError as e:
            raise ValueError(f"JSON inválido: {e}") from e

    if not isinstance(out, list):
        raise TypeError("El resultado de operaciones_gpt5 debe ser una lista de objetos JSON.")

    resultados_ops: List[
        Tuple[int, Dict[str, Any], Union[Tuple[int, int], None], str]
    ] = []

    # 2) Ejecutar cada operación y formatear resultado
    for item in out:
        if isinstance(item, dict) and "params" in item:
            idx = item.get("idx")
            params = item.get("params")
            span = item.get("span")

            try:
                resultado = ejecutar_operaciones_condicionales(df, params)
                resultado_fmt = _format_result_plain(resultado)
                resultados_ops.append((idx, params, span, resultado_fmt))
            except Exception as e:
                # Mantener trazabilidad sin romper el tipo (resultado como string legible)
                error_txt = f"[error:{str(e)}]"
                resultados_ops.append((idx, params, span, error_txt))

    # 3) Reemplazar spans por resultados (elige "html" o "texto simple")
    texto_reemplazado = aplicar_operaciones_en_texto(texto, resultados_ops, formato="html")

    return texto_reemplazado


_TOKEN_RE = re.compile(r"#GRAFICA(?:[_\s]*([0-9]+))?#", flags=re.IGNORECASE)

# ----------------------------
# 5) Función que calcula gráficos.
# ----------------------------

def _fig_to_data_uri(fig) -> str:
    """
    Convierte figuras Matplotlib o Plotly en data:image/png;base64.
    Soporta ambos tipos de figura y devuelve una URI base64 lista para incrustar en HTML.
    """
    import base64
    from io import BytesIO

    buf = BytesIO()

    try:
        # Caso 1: Matplotlib
        if hasattr(fig, "savefig"):
            fig.savefig(
                buf,
                format="png",
                dpi=150,
                bbox_inches="tight",
                facecolor="white",
                edgecolor="white"
            )

        # Caso 2: Plotly
        elif hasattr(fig, "to_image"):  # preferido sobre write_image
            img_bytes = fig.to_image(format="png", scale=2)
            buf.write(img_bytes)
            buf.seek(0)
        elif hasattr(fig, "write_image"):  # compatibilidad vieja
            fig.write_image(buf, format="png", engine="kaleido")

        else:
            raise TypeError(f"Tipo de figura no soportado: {type(fig)}")

    except Exception as e:
        # Devuelve un texto codificado si algo falla (útil para depurar)
        msg = f"error:{str(e)}"
        return f"data:text/plain;base64,{base64.b64encode(msg.encode()).decode()}"

    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("ascii")
    return f"data:image/png;base64,{b64}"

def unir_idx_params_con_span_json(
    idx_params_any: Union[str, Iterable],
    texto: str,
    ensure_ascii: bool = False,
    indent: Union[int, None] = None,
) -> str:
    """
    Une (idx, params) con los span detectados en `texto` por parse_plot_blocks y
    devuelve un JSON string con objetos: [{"idx": int, "params": {...}, "span": [s,e] | null}, ...].

    Acepta `idx_params_any` como:
      - str JSON: '[{"idx":1,"params":{...}}, [2, {...}], ...]'
      - lista de dicts con 'idx' y 'params'
      - lista de pares/tuplas [idx, params] o (idx, params)
    """

    # --- normalización mínima a [(idx:int, params:dict)]
    def _normalize(x: Union[str, Iterable]) -> List[Tuple[int, Dict[str, Any]]]:
        if isinstance(x, str):
            try:
                x = json.loads(x)
            except json.JSONDecodeError:
                return []
        if not isinstance(x, Iterable):
            return []

        out: List[Tuple[int, Dict[str, Any]]] = []
        for item in x:
            if isinstance(item, dict) and "idx" in item and "params" in item and isinstance(item["params"], dict):
                try:
                    out.append((int(item["idx"]), item["params"]))
                except Exception:
                    pass
            elif isinstance(item, (list, tuple)) and len(item) >= 2 and isinstance(item[1], dict):
                try:
                    out.append((int(item[0]), item[1]))
                except Exception:
                    pass
        return out

    idx_params = _normalize(idx_params_any)
    if not idx_params:
        return json.dumps([], ensure_ascii=ensure_ascii, indent=indent)

    # --- mapa idx -> span usando parse_plot_blocks(texto)
    spans_by_idx: Dict[int, Any] = {}
    for it in (parse_plot_blocks(texto) or []):
        if isinstance(it, dict) and "idx" in it and "span" in it:
            try:
                spans_by_idx[int(it["idx"])] = it["span"]
            except Exception:
                pass

    # --- construir salida como lista de objetos JSON-friendly
    result = []
    for idx, params in idx_params:
        span = spans_by_idx.get(idx, None)
        # Normalizar span a lista de 2 enteros si viene como tupla
        if isinstance(span, tuple):
            span = list(span)
        result.append({"idx": idx, "params": params, "span": span})

    return json.dumps(result, ensure_ascii=ensure_ascii, indent=indent)

def aplicar_graficos_en_texto(texto: str, resultados_graficos, formato: str = "html") -> str:
    """
    resultados_graficos: lista de tuplas (idx, params, span_obj, data_uri)
      - span_obj puede ser:
          a) [start, end]
          b) {"span": [start, end]}  # o {"spam": [start, end]} si hubo typo
    """
    reemplazos = []

    for item in resultados_graficos:
        # Saltar entradas de error que guardaste como dict {"error":..., "params":...}
        if isinstance(item, dict):
            continue

        try:
            idx, params, span_obj, data_uri = item
        except Exception:
            # Estructura inesperada, ignorar
            continue

        # Normalizar span a (start, end)
        if isinstance(span_obj, dict):
            span_list = span_obj.get("span") or span_obj.get("spam")
        else:
            span_list = span_obj

        if not (isinstance(span_list, (list, tuple)) and len(span_list) == 2):
            # span inválido, ignorar
            continue

        start, end = span_list
        if not (isinstance(start, int) and isinstance(end, int) and 0 <= start <= end <= len(texto)):
            # índices fuera de rango o tipos erróneos
            continue

        # Reemplazo según formato
        if formato == "html":
            # Clases: una genérica y una específica por índice
            css_class = f"grafico grafico-{idx}"
            reemplazo = (
                f'<div class="{css_class}">'
                f'<img src="{data_uri}" alt="grafico {idx}" />'
                f'</div>'
            )
        else:
            reemplazo = data_uri

        reemplazos.append((start, end, reemplazo))

    # Aplicar de derecha a izquierda para preservar offsets
    reemplazos.sort(key=lambda t: t[0], reverse=True)

    out = texto
    for start, end, rep in reemplazos:
        out = out[:start] + rep + out[end:]

    return out



def _to_base64(s: str) -> str:
    return base64.b64encode(s.encode("utf-8")).decode("ascii")


def process_plot_blocks(df: pd.DataFrame, texto: str):
    # Ejecutar graficos_gpt5 y asegurar conversión a JSON
    #texto = parse_plot_blocks(texto)
    out = graficos_gpt5(df, texto)
    out = unir_idx_params_con_span_json(out, texto)
    if isinstance(out, str):
        try:
            out = json.loads(out)
        except json.JSONDecodeError as e:
            raise ValueError(f"JSON inválido: {e}") from e

    # Validar tipo
    if not isinstance(out, list):
        raise TypeError("El resultado de graficos_gpt5 debe ser una lista de objetos JSON.")

    params_list = []
    resultados_graficos: List[
        Tuple[int, Dict[str, Any], Union[Tuple[int, int], None], Union[str, Dict[str, str]]]
    ] = []
    for item in out:
        if isinstance(item, dict) and "params" in item:
            idx, params, span = item["idx"], item["params"], item["span"]
            params_list.append(params)
            try:
                fig, ax = plot_from_params(df, params)
                resultados = _fig_to_data_uri(fig)
                resultados_graficos.append((idx, params, span, resultados))
            except Exception as e:
                # En caso de error, aún registramos la entrada para trazabilidad (sin romper el tipo)
                error_uri = f"data:text/plain;base64,{_to_base64(f'error:{str(e)}')}"
                resultados_graficos.append((idx, params, span, error_uri))

    # 3) Reemplazar spans por data-URIs (elige el formato que prefieras: "uri" | "md" | "html")
    texto_reemplazado = aplicar_graficos_en_texto(texto, resultados_graficos, formato="html")

    return texto_reemplazado

def aplicar_multiples_columnas_gpt5(
    df: pd.DataFrame,
    tareas: List[Dict[str, Any]],
    *,
    replace_existing: bool = True,
    chunk_size: int = 200,
    debug: bool = False
) -> pd.DataFrame:

    import json
    import math

    if not tareas:
        return df

    df_out = df.copy()

    # -------------------------
    # Helpers
    # -------------------------
    def _parse_val(val):
        if isinstance(val, dict):
            return val
        if isinstance(val, str) and val.strip().startswith("{"):
            try:
                j = json.loads(val)
                if isinstance(j, dict):
                    return j
            except:
                pass
        return {"valor": val}

    def _build_registro(row, cols_list):
        if len(cols_list) == 1:
            return _parse_val(row[cols_list[0]])
        return {c: row[c] for c in cols_list}

    # -------------------------
    # Procesar cada tarea
    # -------------------------
    for tarea in tareas:

        criterios      = tarea["criterios"]
        registro_cols  = tarea["registro_cols"]
        nueva_col      = tarea["nueva_columna"]

        fn_ia    = tarea.get("fn_ia", columns_batch_gpt5)
        on_error = tarea.get("on_error", None)

        cols_list = [registro_cols] if isinstance(registro_cols, str) else list(registro_cols)

        if (not replace_existing) and (nueva_col in df_out.columns):
            raise ValueError(f"La columna '{nueva_col}' ya existe.")

        df_out[nueva_col] = None

        total_rows = len(df_out)
        n_chunks = math.ceil(total_rows / chunk_size)

        for i in range(n_chunks):

            start = i * chunk_size
            end   = min(start + chunk_size, total_rows)
            df_chunk = df_out.iloc[start:end]

            registros = []
            for idx_df, row in df_chunk.iterrows():
                registros.append({
                    "idx": int(idx_df),
                    "registro": _build_registro(row, cols_list)
                })

            # Llamado IA
            try:
                respuesta = fn_ia({"Criterios": criterios}, {"Registros": registros})
            except Exception as e:
                if debug:
                    print("ERROR IA:", e)
                for r in registros:
                    df_out.at[r["idx"], nueva_col] = on_error
                continue

            if debug:
                print("RESPUESTA IA:", respuesta)

            # Normalizar respuesta
            resultados = None

            # Caso A → {"resultados": [...]}
            if isinstance(respuesta, dict) and "resultados" in respuesta:
                resultados = respuesta["resultados"]

            # Caso B → lista de items
            elif isinstance(respuesta, list):
                resultados = respuesta

            # fallback
            if resultados is None:
                for r in registros:
                    df_out.at[r["idx"], nueva_col] = on_error
                continue

            # Mapear cada resultado
            for item in resultados:

                idx = item.get("idx", item.get("id"))
                etiqueta = item.get("resultado", item.get("etiqueta", on_error))

                if idx in df_out.index:
                    df_out.at[idx, nueva_col] = etiqueta

    return df_out


# def aplicar_multiples_columnas_gpt5(
#     df: pd.DataFrame,
#     tareas: List[Dict[str, Any]],
#     *,
#     replace_existing: bool = True,
# ) -> pd.DataFrame:
#     """
#     Procesa múltiples columnas nuevas usando GPT en modo batch.
#     Para cada tarea:
#       - toma los registros del DF,
#       - arma un batch JSON con IDs secuenciales,
#       - llama a columns_batch_gpt5(),
#       - recibe un array con {"id": "...", "etiqueta": "..."},
#       - reconstruye la columna en el DF.

#     Mantiene la lógica original, pero optimizando costos y velocidad.
#     """

#     import json

#     # Si no hay tareas → devolver igual
#     if not tareas:
#         return df

#     # Trabajamos sobre una copia
#     out = df.copy()

#     # =====================================================
#     # Helpers compatibles con tu flujo original
#     # =====================================================

#     def _parse_val(val):
#         """Normaliza valores: si es JSON lo convierte en dict."""
#         if isinstance(val, dict):
#             return val
#         if isinstance(val, str) and val.strip().startswith("{"):
#             try:
#                 j = json.loads(val)
#                 if isinstance(j, dict):
#                     return j
#             except Exception:
#                 pass
#         return {"valor": val}

#     def _build_registro(row, cols_list):
#         """Construye un registro compatible con el clasificador."""
#         if len(cols_list) == 1:
#             return _parse_val(row[cols_list[0]])
#         return {c: row[c] for c in cols_list}

#     # =====================================================
#     # Procesar cada tarea de clasificación
#     # =====================================================
#     for tarea in tareas:

#         criterios      = tarea["criterios"]        # texto plano
#         registro_cols  = tarea["registro_cols"]
#         nueva_columna  = tarea["nueva_columna"]

#         fn_ia    = tarea.get("fn_ia", columns_batch_gpt5)
#         on_error = tarea.get("on_error", None)

#         # Convertir registro_cols a lista
#         cols_list = [registro_cols] if isinstance(registro_cols, str) else list(registro_cols)

#         # Reindex temporalmente a 0..N-1 para que coincidan los IDs
#         out = out.reset_index(drop=True)

#         # =====================================================
#         # Construir el batch de registros a enviar a la IA
#         # =====================================================
#         registros_batch = []

#         for i in range(len(out)):
#             reg = _build_registro(out.loc[i], cols_list)
#             registros_batch.append({
#                 "id": str(i),     # El batch trabaja con string IDs, consistente con tu prompt
#                 "registro": reg
#             })

#         # Estructura enviada al modelo
#         payload = {
#             "Criterios": criterios,
#             "Registros": registros_batch,
#         }

#         # =====================================================
#         # Llamada real a la IA en batch
#         # =====================================================
#         try:
#             resultados = fn_ia(payload, payload)   # tu función espera (criterios, registros) aunque sean el mismo dict
#         except Exception:
#             # Si falló el batch → llenar con on_error
#             out[nueva_columna] = on_error
#             continue

#         # =====================================================
#         # Normalizar la respuesta del modelo
#         # =====================================================
#         mapa = {}   # id → etiqueta

#         if isinstance(resultados, list):
#             for item in resultados:
#                 if not isinstance(item, dict):
#                     continue

#                 idx = str(item.get("id"))
#                 etiqueta = item.get("etiqueta")

#                 if idx is not None:
#                     mapa[idx] = etiqueta
#         else:
#             # Caso inesperado: respuesta no es lista
#             out[nueva_columna] = on_error
#             continue

#         # =====================================================
#         # Construir la nueva columna con los resultados
#         # =====================================================
#         nueva_series = []

#         for i in range(len(out)):
#             key = str(i)
#             nueva_series.append(mapa.get(key, on_error))

#         # Validar reemplazo
#         if (not replace_existing) and (nueva_columna in out.columns):
#             raise ValueError(
#                 f"La columna '{nueva_columna}' ya existe. "
#                 f"Activa replace_existing=True para reemplazarla."
#             )

#         out[nueva_columna] = nueva_series

#     # Retornar DF final
#     return out


# def aplicar_multiples_columnas_gpt5(
#     df: pd.DataFrame,
#     tareas: List[Dict[str, Any]],
#     *,
#     replace_existing: bool = True,
# ) -> pd.DataFrame:
#     """
#     Procesa múltiples columnas nuevas usando IA.

#     Si 'tareas' está vacío → devuelve df.
#     Permite reemplazar columnas existentes si replace_existing=True.

#     Cada tarea debe incluir:
#       - 'criterios'
#       - 'registro_cols'
#       - 'nueva_columna'
#       - opcional: 'fn_ia'
#       - opcional: 'on_error'
#     """

#     import json

#     # -------------------------------------
#     # 🟢 Si no hay tareas, devolver el df intacto
#     # -------------------------------------
#     if not tareas:
#         return df

#     out = df.copy()

#     # --------------------------
#     # Helpers compartidos
#     # --------------------------
#     def _parse_val(val):
#         if isinstance(val, dict):
#             return val
#         if isinstance(val, str) and val.strip().startswith("{"):
#             try:
#                 j = json.loads(val)
#                 if isinstance(j, dict):
#                     return j
#             except Exception:
#                 pass
#         return {"valor": val}

#     def _build_registro(row, cols_list):
#         if len(cols_list) == 1:
#             return _parse_val(row[cols_list[0]])
#         return {c: row[c] for c in cols_list}

#     def _key(reg):
#         try:
#             return json.dumps(reg, ensure_ascii=False, sort_keys=True)
#         except Exception:
#             return str(reg)

#     # -----------------------------------------
#     # Procesar cada tarea
#     # -----------------------------------------
#     for tarea in tareas:

#         criterios      = tarea["criterios"]
#         registro_cols  = tarea["registro_cols"]
#         nueva_columna  = tarea["nueva_columna"]

#         fn_ia    = tarea.get("fn_ia", columns_gpt5)
#         on_error = tarea.get("on_error", None)

#         # Normalizar input
#         cols_list = [registro_cols] if isinstance(registro_cols, str) else list(registro_cols)

#         # Cache por tarea
#         cache = {}

#         def clasificar_row(row):
#             reg = _build_registro(row, cols_list)
#             k = _key(reg)

#             if k in cache:
#                 return cache[k]

#             try:
#                 etiqueta = fn_ia({"Criterios": criterios}, {"Registro": reg})
#                 if isinstance(etiqueta, str):
#                     etiqueta = etiqueta.strip()
#             except Exception:
#                 etiqueta = on_error

#             cache[k] = etiqueta
#             return etiqueta

#         # ---------------------------------------------------------
#         # 🟦 Crear o reemplazar la columna según replace_existing
#         # ---------------------------------------------------------
#         if (not replace_existing) and (nueva_columna in out.columns):
#             raise ValueError(
#                 f"La columna '{nueva_columna}' ya existe. "
#                 f"Activa replace_existing=True para reemplazarla."
#             )

#         out[nueva_columna] = out.apply(clasificar_row, axis=1)

#     return out

# ----------------------------
# 7) Exportar a HTML con saltos de página y data URIs.
# ----------------------------

PAGE_BREAK_TOKENS = [
    r"\f",                       # Form feed (salto de página clásico)
    r"\[\[PAGE_BREAK\]\]",       # [[PAGE_BREAK]]
    r"\[\[SALTO_PAGINA\]\]",     # [[SALTO_PAGINA]]
    r"#PAGEBREAK#",              # #PAGEBREAK#
    r"#SALTO#",                  # #SALTO#
    r"<pagebreak\s*/?>",         # <pagebreak/> o <pagebreak>
    r"---\s*PAGEBREAK\s*---",    # --- PAGEBREAK ---
]

# Compila un solo patrón para todos los tokens de salto de página
PAGE_BREAK_PATTERN = re.compile(
    "(" + "|".join(PAGE_BREAK_TOKENS) + ")",
    flags=re.IGNORECASE
)

# Detecta URIs de imagen en base64 (data:image/...;base64,xxxxx)
DATA_IMAGE_PATTERN = re.compile(
    r"(data:image/(?:png|jpeg|jpg|gif|webp);base64,[A-Za-z0-9+/=\s]+)"
)

def _insertar_page_breaks(texto: str) -> str:
    """
    Reemplaza tokens de salto de página por un div con clase 'page-break'.
    """
    return PAGE_BREAK_PATTERN.sub('<div class="page-break"></div>', texto)

def _envolver_data_uri_en_img(texto: str) -> str:
    """
    Si existen URIs base64 de imagen sin estar dentro de <img>, los envuelve.
    Respeta <img src="data:image/..."> ya existentes.
    """
    if re.search(r'<\s*img[^>]+src\s*=\s*"(?:\s*)data:image/', texto, flags=re.IGNORECASE):
        pass

    def wrap_if_not_img(m: re.Match) -> str:
        uri = m.group(1).strip()
        start = m.start()
        context_start = max(0, start - 120)
        contexto = texto[context_start:start]

        if re.search(r'<\s*img[^>]+src\s*=\s*["\']?\s*$', contexto, flags=re.IGNORECASE):
            return uri

        # look limpio por defecto (centrado y más pequeño dentro del texto)
        return (
            f'<img class="img-doc" src="{uri}" alt="imagen" '
            f'style="display:block; margin:12px auto; max-width:58%; height:auto;" />'
        )

    return DATA_IMAGE_PATTERN.sub(wrap_if_not_img, texto)

def _lineas_a_parrafos_y_br(texto: str) -> str:
    """
    Convierte dobles saltos de línea en párrafos <p> y simples en <br>.
    Respeta los <div class="page-break"> insertados.
    """
    # Separar con marcador temporal para no romper page-breaks al hacer split
    marcador = "___PAGE_BREAK___"
    texto = texto.replace('<div class="page-break"></div>', marcador)

    # Normalizar CRLF
    texto = texto.replace("\r\n", "\n\n").replace("\r", "\n\n")

    bloques = texto.split("\n\n")
    html_partes = []
    for bloque in bloques:
        if bloque.strip() == "":
            continue
        # Dentro de cada bloque, un solo \n => <br>
        bloque_html = bloque.replace("\n", "<br>")
        # Si el bloque es exactamente el marcador, no lo envuelvas en <p>
        if bloque_html.strip() == marcador:
            html_partes.append(marcador)
        else:
            html_partes.append(f"<p>{bloque_html}</p>")

    html = "\n".join(html_partes)
    return html.replace(marcador, '<div class="page-break"></div>')

def exportar_output_a_html(
    contenido: str,
    ruta_salida: Optional[str] = None,
    titulo: str = "Reporte",
    estilos_extra: Optional[str] = None,
    crear_nombre_si_directorio: bool = True,
    devolver_html_si_falla_write: bool = False
) -> str:
    """
    Genera HTML a partir de 'contenido' reconociendo:
      - Imágenes en base64 (URIs data:image/...;base64,...) y envolviéndolas con <img> si vienen sueltas.
      - Saltos de página con varios tokens, traducidos a <div class="page-break"></div> imprimible.

    Si 'ruta_salida' es:
      - None => retorna el HTML como string.
      - Archivo => escribe el archivo y retorna la ruta final.
      - Carpeta => si 'crear_nombre_si_directorio' es True, crea un nombre 'reporte_YYYYmmdd_HHMMSS.html' dentro.
                   Si es False, lanza ValueError.

    En caso de error de escritura (PermissionError, etc.):
      - Si 'devolver_html_si_falla_write' es True, retorna el HTML como string.
      - Si es False, relanza la excepción.

    Returns:
        str: ruta del archivo escrito, o el HTML como string si ruta_salida es None o si se decidió devolverlo.
    """

    # 1) Procesamiento del contenido
    contenido = _insertar_page_breaks(contenido)
    contenido = _envolver_data_uri_en_img(contenido)
    contenido = _lineas_a_parrafos_y_br(contenido)

    # 2) HTML final
    estilos_base = """
<style>
  :root{
    --doc-max: 820px;
    --font: -apple-system, Segoe UI, Roboto, Helvetica, Arial, "Noto Sans", Ubuntu, "Apple Color Emoji", "Segoe UI Emoji";
    --text: #1e2329;
    --muted:#667085;
    --accent:#25347a;     /* azul corporativo sugerido */
    --accent-2:#58b12e;   /* verde corporativo sugerido */
    --border:#e6e8ec;
    --bg:#ffffff;
  }
  *{ box-sizing:border-box; }

  html,body{ background:var(--bg); color:var(--text); }
  body{
    font-family: var(--font);
    margin:0;
    -webkit-font-smoothing:antialiased;
    -moz-osx-font-smoothing:grayscale;
  }

  /* Contenedor tipo documento */
  main{
    max-width: var(--doc-max);
    margin: 28px auto;
    padding: 0 20px 40px;
    line-height: 1.65;
    font-size: 15.5px;
  }

  /* Encabezados – sobrios y legibles */
  h1,h2,h3,h4{
    color: var(--accent);
    margin: 22px 0 12px;
    line-height:1.25;
    letter-spacing: .2px;
  }
  h1{ font-size: 28px; border-bottom: 2px solid var(--accent-2); padding-bottom: 8px; }
  h2{ font-size: 22px; margin-top: 28px; }
  h3{ font-size: 18px; }
  h4{ font-size: 16px; color:#2d3440; }

  p{ margin: 10px 0 12px; text-align: justify; text-justify: inter-word; }
  em, i{ color: var(--muted); }

  /* Imágenes y figuras (centradas, más pequeñas por defecto) */
  img{ max-width:100%; height:auto; }
  .img-doc{
    display:block;
    margin: 12px auto;
    max-width: 58%;   /* <- tamaño “más pequeño” dentro del texto */
    height:auto;
  }
  .caption{
    text-align:center;
    font-size: 13px;
    color: var(--muted);
    margin-top: 6px;
  }

  /* Listas y citas */
  ul,ol{ padding-left: 22px; margin: 8px 0 14px; }
  blockquote{
    margin: 14px 0;
    padding: 10px 14px;
    border-left: 3px solid var(--accent-2);
    background: #f7faf7;
    color:#2f3b2f;
  }

  /* Tablas con zebra */
  table{
    width:100%;
    border-collapse:collapse;
    margin: 14px 0 18px;
    font-size: 14px;
  }
  th,td{
    border: 1px solid var(--border);
    padding: 8px 10px;
    vertical-align: top;
  }
  th{
    background:#f7f8fb;
    color:#2d3440;
    font-weight:600;
  }
  tbody tr:nth-child(odd){ background:#fbfcfe; }

  /* Separadores */
  hr{
    border:0; border-top:1px solid var(--border);
    margin: 18px 0;
  }

  /* Page breaks visibles en pantalla, reales en impresión */
  .page-break{
    display:block; height:0; margin: 24px 0; border:0;
    border-top: 2px dashed #ccd2da;
  }
  @media print{
    @page{ margin: 1.8cm; }
    body{ background:#fff; }
    main{ padding:0; }
    .page-break{
      page-break-before: always;
      border:0; margin:0; height:0;
    }
    a[href]:after{ content:""; } /* limpiar urls impresas */
  }

  /* Utilidades por si las necesitas en tu contenido */
  .center{text-align:center;}
  .right{text-align:right;}
  .muted{color:var(--muted);}
  .w-50{max-width:50%;} .w-60{max-width:60%;} .w-70{max-width:70%;}
</style>
"""


    if estilos_extra:
        estilos_base += f"\n<style>\n{estilos_extra}\n</style>"

    html_final = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8" />
<title>{titulo}</title>
{estilos_base}
</head>
<body>
<main>
{contenido}
</main>
</body>
</html>
"""

    # 3) Guardar o devolver
    if ruta_salida is None:
        return html_final

    # Si me pasan una carpeta, genero nombre de archivo dentro
    ruta_final = ruta_salida
    try:
        if os.path.isdir(ruta_salida):
            if not crear_nombre_si_directorio:
                raise ValueError(
                    f"La ruta dada es un directorio y crear_nombre_si_directorio=False: {ruta_salida}"
                )
            nombre = f"reporte_{time.strftime('%Y%m%d_%H%M%S')}.html"
            ruta_final = os.path.join(ruta_salida, nombre)
        else:
            # Si es ruta de archivo, asegurar carpeta
            carpeta = os.path.dirname(ruta_salida)
            if carpeta and not os.path.exists(carpeta):
                os.makedirs(carpeta, exist_ok=True)

        with open(ruta_final, "w", encoding="utf-8") as f:
            f.write(html_final)

        return ruta_final

    except PermissionError as e:
        if devolver_html_si_falla_write:
            return html_final
        raise e


import re
from bs4 import BeautifulSoup

def mejorar_html_informe(html_raw: str) -> str:
    """
    Toma el HTML generado por informe_final y:
    - Detecta portada
    - Identifica títulos principales y secundarios
    - Mejora tablas
    - Inserta CSS corporativo
    - Devuelve HTML final listo para exportar o PDF
    """

    # ============================================================
    # 1. Insertar CSS CORPORATIVO
    # ============================================================
    CSS = """
    <style>
      :root{
        --doc-max: 820px;
        --font: -apple-system, Segoe UI, Roboto, Helvetica, Arial, "Noto Sans";
        --text: #1e2329;
        --muted:#667085;
        --accent:#25347a;
        --accent-2:#58b12e;
        --bg:#ffffff;
        --border:#e1e4e8;
      }
      *{ box-sizing:border-box; }
      html,body{ margin:0; padding:0; background:var(--bg); color:var(--text); }

      body{
        font-family:var(--font);
        line-height:1.65;
      }

      .portada{
        text-align:center;
        padding:90px 40px 120px;
        border-bottom:6px solid var(--accent);
        margin-bottom:40px;
      }
      .portada h1{ font-size:34px; color:var(--accent); margin-bottom:12px; }
      .portada h2{ font-size:22px; margin:0; color:var(--muted); }
      .portada .empresa{ margin-top:25px; font-size:18px; font-weight:600; }

      main{
        max-width:var(--doc-max);
        margin: 0 auto 60px;
        padding: 0 24px;
        font-size:15.5px;
      }

      p{ margin:12px 0; text-align:justify; }

      h1{
        color:var(--accent);
        font-size:28px;
        font-weight:700;
        border-bottom:3px solid var(--accent-2);
        padding-bottom:6px;
        margin-top:40px;
      }

      h2{
        color:var(--accent);
        font-size:23px;
        margin-top:34px;
        font-weight:600;
      }

      h3{
        font-size:18px;
        margin-top:28px;
        font-weight:600;
        color:#2d3440;
      }
      h4{
        background:#eef3ff;
        padding:6px 10px;
        border-left:4px solid var(--accent);
        font-size:16px;
      }

      table{
        width:100%;
        border-collapse:collapse;
        margin:22px 0;
        font-size:14px;
      }
      th{
        background:#f0f3f9;
        font-weight:700;
        border-bottom:2px solid var(--accent);
        padding:10px;
      }
      td{
        padding:8px 10px;
        border-bottom:1px solid var(--border);
      }
      tbody tr:nth-child(odd){ background:#fafbff; }

      .tabla-destacada{
        border:2px solid var(--accent);
      }

      .figura{
        text-align:center;
        margin:20px auto;
      }
      .figura img{
        max-width:60%;
        display:block;
        margin:0 auto;
      }
      .figura .caption{
        font-size:13px;
        color:var(--muted);
        margin-top:6px;
      }

      .page-break{
        page-break-before:always;
        margin-top:40px;
      }
    </style>
    """

    # Insertar CSS al comienzo del HTML
    html_raw = CSS + html_raw

    # ============================================================
    # 2. Parsear con BeautifulSoup
    # ============================================================
    soup = BeautifulSoup(html_raw, "html.parser")

    # ============================================================
    # 3. DETECTAR TÍTULOS AUTOMÁTICAMENTE
    # ============================================================

    patrones_h1 = re.compile(r"^\s*\*?\s*\d+\.\s+.+", re.IGNORECASE)
    patrones_h2 = re.compile(r"^\s*\d+\.\d+\s+.+", re.IGNORECASE)

    for p in soup.find_all("p"):
        txt = p.get_text(strip=True)

        # Título principal (1. INTRODUCCIÓN)
        if patrones_h1.match(txt):
            new_tag = soup.new_tag("h1")
            new_tag.string = txt
            p.replace_with(new_tag)

        # Subtítulo (3.2 Objetivos específicos)
        elif patrones_h2.match(txt):
            new_tag = soup.new_tag("h2")
            new_tag.string = txt
            p.replace_with(new_tag)

        # Títulos escritos en MAYÚSCULAS
        elif txt.isupper() and len(txt) > 8:
            new_tag = soup.new_tag("h2")
            new_tag.string = txt
            p.replace_with(new_tag)

    # ============================================================
    # 4. DETECTAR Y MEJORAR TABLAS
    # ============================================================
    for table in soup.find_all("table"):
        classes = table.get("class", [])
        if "tabla-destacada" not in classes:
            classes.append("tabla-destacada")
        table["class"] = classes

    # ============================================================
    # 5. DETECTAR Y CONSTRUIR PORTADA
    # ============================================================
    # La portada será el contenido previo al primer <h1>
    portada_items = []
    for elem in soup.body.contents:
        if elem.name == "h1":
            break
        portada_items.append(elem)

    if portada_items:
        portada_div = soup.new_tag("div", **{"class": "portada"})
        for e in portada_items:
            portada_div.append(e.extract())

        soup.body.insert(0, portada_div)

    return str(soup)


from bs4 import BeautifulSoup
import re

def estructurar_documento(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    main = soup.find("main")
    if not main:
        return html

    # ============================================================
    # 1. DETECTAR PORTADA (primeros párrafos que no son contenido)
    # ============================================================
    portada_items = []
    contenido_items = []
    
    portada_keywords = [
        "diagnostico", "evaluaciones", "empresa", "resultado",
        "rango de fechas", "institución", "cobertura"
    ]

    def es_portada(p):
        txt = p.get_text().strip().lower()
        return any(k in txt for k in portada_keywords)

    for p in main.find_all("p", recursive=False):
        if es_portada(p):
            portada_items.append(p)
        else:
            contenido_items.append(p)

    # Envolver portada en una sección
    if portada_items:
        portada_section = soup.new_tag("section", **{"class": "portada"})
        for p in portada_items:
            portada_section.append(p)
        main.insert(0, portada_section)

    # ====================================================================
    # 2. DETECTAR TABLA DE CONTENIDO (TOC)
    # ====================================================================
    toc_section = None
    for p in main.find_all("p"):
        if "contenido" in p.get_text().lower():
            toc_section = p
            break

    if toc_section:
        toc_section["class"] = "tabla-contenido"
        # Aquí podrías extender para extraer los ítems
    # ====================================================================
    # 3. DETECTAR TÍTULOS “FALSOS” EN <p> Y CONVERTIRLOS A <h2> O <h3>
    # ====================================================================
    
    patron_titulo = re.compile(r"""^(
        \d+(\.\d+)*\s+.+       |   # Títulos tipo 8.1 Pirámide
        [A-ZÁÉÍÓÚÑ ]{5,}           # Texto TODO EN MAYÚSCULA
    )$""", re.VERBOSE)

    for p in main.find_all("p"):
        txt = p.get_text().strip()

        # identificar títulos reales
        if patron_titulo.match(txt):
            new_h = soup.new_tag("h2")
            new_h.string = txt
            p.replace_with(new_h)

    # ================================================================
    # 4. DEVOLVER HTML MODIFICADO
    # ================================================================
    return str(soup)



#-----------------------------

def columnas_a_texto(df: pd.DataFrame, col1: str, col2: str, *, 
                     sep: str = "\n\n", dropna: bool = True, strip: bool = True) -> str:
    """
    Devuelve un string con los valores de col1 y col2 en orden horizontal por filas:
    fila1: col1, col2; fila2: col1, col2; ... separados por sep (default: salto de línea).
    
    Parámetros:
        df: DataFrame de entrada.
        col1, col2: nombres de columnas a considerar.
        sep: separador entre valores en el texto final (por defecto '\n').
        dropna: si True, omite celdas NaN/None.
        strip: si True, hace strip() a cada valor convertido a string.
    """
    # Validaciones básicas
    for c in (col1, col2):
        if c not in df.columns:
            raise ValueError(f"La columna '{c}' no existe en el DataFrame.")
    
    piezas = []
    # Itera por filas y dentro de cada fila recorre col1 -> col2 (orden horizontal)
    for a, b in df[[col1, col2]].itertuples(index=False, name=None):
        for v in (a, b):
            if v is None or (isinstance(v, float) and pd.isna(v)):
                if dropna:
                    continue
                v = ""  # si no dropna, representa NA como cadena vacía
            s = str(v)
            if strip:
                s = s.strip()
            piezas.append(s)
    
    return sep.join(piezas)


def limpieza_final(texto: str) -> str:
    # eliminar todo lo que esté entre #...#
    texto = re.sub(r"#.*?#", "", texto, flags=re.DOTALL)
    
    # eliminar signos [ ] *
    texto = texto.replace("[", "").replace("]", "").replace("*", "")
    
    # limpiar dobles espacios
    #texto = re.sub(r" {2,}", " ", texto)
    
    # limpiar espacios antes de saltos
    #texto = re.sub(r" +\n", "\n", texto)

    return texto.strip()



def unpivot_df(df: pd.DataFrame,
               columnas_unpivot: list,
               nombre_columna_variable: str = "variable",
               nombre_columna_valor: str = "valor") -> pd.DataFrame:
    """
    Realiza un unpivot (melt) de un conjunto de columnas específicas
    y devuelve un dataframe transformado.

    Parámetros:
    -----------
    df : pd.DataFrame
        DataFrame original.
    columnas_unpivot : list
        Columnas que deseo convertir a dos columnas (variable/valor).
    nombre_columna_variable : str
        Nombre de la columna resultante que tendrá el nombre original de la columna unpivot.
    nombre_columna_valor : str
        Nombre de la columna resultante con el valor correspondiente.

    Retorna:
    --------
    pd.DataFrame transformado
    """
    cols_faltantes = [c for c in columnas_unpivot if c not in df.columns]
    if cols_faltantes:
            raise KeyError(f"Las siguientes columnas no están en el DataFrame: {cols_faltantes}")
    
    # Columnas que NO se unpivotan
    columnas_id = [c for c in df.columns if c not in columnas_unpivot]

    # Melt
    df_unpivot = df.melt(
        id_vars=columnas_id,
        value_vars=columnas_unpivot,
        var_name=nombre_columna_variable,
        value_name=nombre_columna_valor
    )

    return df_unpivot

def dividir_columna_en_dos(
    df: pd.DataFrame,
    columna: str,
    caracter_separador: str = "-",    # caracter especial a buscar
    nombre_col1: str = "parte_1",
    nombre_col2: str = "parte_2",
    eliminar_original: bool = False,
) -> pd.DataFrame:

    # 1. Validación
    if columna not in df.columns:
        raise KeyError(f"La columna '{columna}' no existe en el DataFrame.")

    # 2. Convertir a texto para evitar errores con valores nulos o numéricos
    serie = df[columna].astype(str)

    # 3. Dividir solo en el PRIMER separador encontrado
    partes = serie.str.split(caracter_separador, n=1, expand=True)

    # Si no encuentra el separador, crear columnas sin dividir
    if partes.shape[1] == 1:
        partes[1] = None

    # 4. Asignar columnas limpias
    df[nombre_col1] = partes[0].str.strip()
    df[nombre_col2] = partes[1].str.strip() if partes.shape[1] > 1 else None

    # 5. Eliminar columna original si se requiere
    if eliminar_original:
        df = df.drop(columns=[columna])

    return df


def procesar_codigos_cie10(
    df: pd.DataFrame,
    columna_texto: str = "obs_diagnostico",
    columnas_eliminar: list = "firma"
) -> pd.DataFrame:
    """
    Extrae códigos CIE10 de una columna, los normaliza, los explota en filas
    y devuelve un DataFrame transformado.
    
    Parámetros:
        df: DataFrame de entrada
        columna_texto: Nombre de la columna que contiene los textos con códigos
        columnas_eliminar: Lista opcional de columnas a eliminar después del explode
        
    Retorna:
        DataFrame transformado con 1 fila por código CIE10 encontrado.
    """

    # --- 1. Función interna para extraer códigos ---
    def extraer_codigos(texto):
        if pd.isna(texto):
            return []
        
        texto = str(texto)

        # Busca patrones tipo CIE10|J45:
        codigos = re.findall(r"CIE10\|([A-Z0-9]+)\s*:", texto)

        # Normalizar y eliminar repetidos preservando orden
        codigos = list(dict.fromkeys([c.strip().upper() for c in codigos]))

        return codigos

    # --- 2. Crear columna con lista de códigos ---
    df = df.copy()
    df["__cie10_list"] = df[columna_texto].apply(extraer_codigos)

    # --- 3. Explode: una fila por código ---
    df = df.explode("__cie10_list", ignore_index=True)

    # --- 4. Reemplazar columna original por el código extraído ---
    df[columna_texto] = df["__cie10_list"]

    # --- 5. Eliminar columnas auxiliares ---
    cols_drop = ["__cie10_list"]
    if columnas_eliminar:
        cols_drop.extend(columnas_eliminar)

    df = df.drop(columns=[col for col in cols_drop if col in df.columns])

    return df

def unir_dataframes(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    col_df1: str,
    col_df2: str,
    tipo_union: str = "left"
) -> pd.DataFrame:
    """
    Une dos DataFrames usando diferentes columnas de enlace.

    Parámetros:
        df1: Primer DataFrame (base)
        df2: Segundo DataFrame
        col_df1: Nombre de la columna en df1 para unir
        col_df2: Nombre de la columna en df2 para unir
        tipo_union: Tipo de unión ('left', 'right', 'inner', 'outer')

    Retorna:
        DataFrame unido según los parámetros especificados.
    """

    if col_df1 not in df1.columns:
        raise KeyError(f"La columna {col_df1} no existe en df1")
    if col_df2 not in df2.columns:
        raise KeyError(f"La columna {col_df2} no existe en df2")

    df_merged = df1.merge(
        df2,
        left_on=col_df1,
        right_on=col_df2,
        how=tipo_union
    )

    return df_merged
import pandas as pd
import json

def expand_json_column(df, json_col, fields_to_extract, rename_map=None):
    """
    df: DataFrame original
    json_col: nombre de la columna que contiene el arreglo JSON
    fields_to_extract: lista de claves que se desean extraer
    rename_map: diccionario opcional para renombrar columnas

    Comportamiento:
    - Si hay elementos en laboratorios_incluidos → expande filas
    - Si NO hay elementos → mantiene registro original y agrega columnas vacías
    """
    rows = []

    for _, row in df.iterrows():
        json_value = row[json_col]

        # Convertir string JSON → dict/list si aplica
        if isinstance(json_value, str):
            try:
                json_value = json.loads(json_value)
            except:
                json_value = None

        # Verificar si el JSON tiene la lista
        items = []
        if isinstance(json_value, dict) and "laboratorios_incluidos" in json_value:
            items = json_value["laboratorios_incluidos"] or []

        # 🔹 CASO 1: Hay elementos → expandir filas
        if items:
            for item in items:
                new_row = row.to_dict()
                for f in fields_to_extract:
                    new_row[f] = item.get(f, None)
                rows.append(new_row)

        # 🔹 CASO 2: NO hay elementos → mantener la fila con campos vacíos
        else:
            new_row = row.to_dict()
            for f in fields_to_extract:
                new_row[f] = None
            rows.append(new_row)

    df_new = pd.DataFrame(rows)

    # Renombrar columnas opcional
    if rename_map:
        df_new = df_new.rename(columns=rename_map)

    return df_new
