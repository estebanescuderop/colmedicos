# --- Standard library ---
import os
import re
import json
import time
import base64
from collections.abc import Iterable
from typing import Any, Dict, List, Tuple, Union, Optional, Callable

# --- Third-party ---
import pandas as pd
import numpy as np

# --- Project-local ---
from Colmedicos.registry import register
from Colmedicos.ia import operaciones_gpt5, graficos_gpt5, ask_gpt5,columns_batch_gpt5, apendices_gpt5, titulos_gpt5
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
# 5) Reemplazo por familias
# ----------------------------
import re

def extraer_ia_blocks(texto: str):
    """
    Extrae bloques de IA marcados entre + ... +
    
    Retorna una lista:
    [
      {"idx": 1, "prompt": "...", "span": (inicio, fin)},
      {"idx": 2, "prompt": "...", "span": (inicio, fin)}
    ]
    """
    patron = re.compile(r"\+(.*?)\+", re.DOTALL)
    bloques = []

    for i, match in enumerate(patron.finditer(texto), start=1):
        bloques.append({
            "idx": i,
            "prompt": match.group(1).strip(),
            "span": match.span()   # (inicio, fin)
        })

    return bloques

def aplicar_ia_en_texto(texto: str, resultados_ia, formato: str = "html") -> str:
    """
    resultados_ia: lista de tuplas (idx, prompt, span_obj, resultado_ia)
      - span_obj puede ser:
          a) [start, end]
          b) {"span": [start, end]}
          c) {"spam": [start, end]} (tolerancia)
    Reemplaza en `texto` cada bloque +...+ con el resultado generado por IA.
    """

    reemplazos = []

    for item in resultados_ia:
        # Saltar si llega un dict de error
        if isinstance(item, dict):
            continue

        try:
            idx, prompt, span_obj, resultado_ia = item
        except Exception:
            # estructura inesperada
            continue

        # Normalizar span a (start, end)
        if isinstance(span_obj, dict):
            span_list = span_obj.get("span") or span_obj.get("spam")
        else:
            span_list = span_obj

        if not (isinstance(span_list, (list, tuple)) and len(span_list) == 2):
            continue

        start, end = span_list
        if not (0 <= start <= end <= len(texto)):
            continue

        # --- construir reemplazo ---
        if formato == "html":
            rep = f'<span class="ia-bloque_{idx}">{resultado_ia}</span>'
        else:
            rep = f"[IA bloque {idx}] {resultado_ia}"

        reemplazos.append((start, end, rep))

    # Aplicar de derecha a izquierda (muy importante)
    reemplazos.sort(key=lambda t: t[0], reverse=True)

    out = texto
    for start, end, rep in reemplazos:
        out = out[:start] + rep + out[end:]

    return out


@register("process_ia_blocks")
def process_ia_blocks(
    texto: str,
    *,
    batch_size: int = 10,
    max_workers: int = 4,
    debug: bool = False,
) -> str:
    """
    Versión FINAL optimizada de bloques IA (+ ... +)

    - Batching de instrucciones (optimiza costo)
    - Paralelización de llamadas IA (optimiza tiempo)
    - Orden y spans deterministas
    - Fallback seguro ante errores/cuota
    """

    import concurrent.futures
    import json

    # -------------------------------------------------
    # 1️⃣ Extraer bloques
    # -------------------------------------------------
    extract = extraer_ia_blocks(texto)
    if not extract:
        return texto

    def _chunk_list(lst, size):
        for i in range(0, len(lst), size):
            yield lst[i:i + size]

    # -------------------------------------------------
    # 2️⃣ Crear batches
    # -------------------------------------------------
    batches = list(_chunk_list(extract, batch_size))

    # -------------------------------------------------
    # 3️⃣ Job IA (batch)
    # -------------------------------------------------
    def _procesar_batch(batch):
        try:
            out = ask_gpt5(batch)

            if isinstance(out, str):
                try:
                    out = _json_loads_loose(out)
                except Exception as e:
                    if debug:
                        print("Error parseando IA JSON:", e)
                    return []

        except Exception as e:
            if debug:
                print("Error IA batch:", e)
            return []

        if not isinstance(out, list):
            return []

        return out  # [{idx, params, span}, ...]

    # -------------------------------------------------
    # 4️⃣ Ejecutar IA en paralelo
    # -------------------------------------------------
    resultados_por_idx = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_procesar_batch, batch) for batch in batches]

        for future in concurrent.futures.as_completed(futures):
            try:
                out_batch = future.result()
                for item in out_batch:
                    if isinstance(item, dict) and "params" in item:
                        resultados_por_idx[item["idx"]] = item
            except Exception as e:
                if debug:
                    print("Error future IA:", e)

    # -------------------------------------------------
    # 5️⃣ Reemplazar en texto (SECUENCIAL)
    # -------------------------------------------------
    resultados_ia = []

    for bloque in sorted(extract, key=lambda b: b["idx"]):
        idx = bloque["idx"]
        span = bloque["span"]

        item = resultados_por_idx.get(idx)
        if not item:
            continue

        params = item.get("params")

        try:
            resultados_ia.append((idx, bloque["prompt"], span, params))
        except Exception as e:
            resultados_ia.append((idx, bloque["prompt"], span, f"[error:{str(e)}]"))

    texto_reemplazado = aplicar_ia_en_texto(
        texto,
        resultados_ia,
        formato="html"
    )

    return texto_reemplazado


# @register("process_ia_blocks")
# def process_ia_blocks(
#     texto: str # "raise" | "return_input"
# ) -> str:

#    # 1) Ejecutar operaciones_gpt5 y asegurar conversión a JSON con spans
#     extract = extraer_ia_blocks(texto)
#     out = ask_gpt5(extract)
 
#     if isinstance(out, str):
#         try:
#             out = json.loads(out)
#         except json.JSONDecodeError as e:
#             raise ValueError(f"JSON inválido: {e}") from e

#     if not isinstance(out, list):
#         raise TypeError("El resultado de operaciones_gpt5 debe ser una lista de objetos JSON.")

#     resultados_ops: List[
#         Tuple[int, Dict[str, Any], Union[Tuple[int, int], None], str]
#     ] = []


#     # 2) Ejecutar cada operación y formatear resultado
#     for item in out:
#         if isinstance(item, dict) and "params" in item:
#             idx = item.get("idx")
#             params = item.get("params")
#             span = item.get("span")

#             try:
#                 resultado_fmt = params
#                 resultados_ops.append((idx, params, span, resultado_fmt))
#             except Exception as e:
#                 # Mantener trazabilidad sin romper el tipo (resultado como string legible)
#                 error_txt = f"[error:{str(e)}]"
#                 resultados_ops.append((idx, params, span, error_txt))
    
#     # 3) Reemplazar spans por resultados (elige "html" o "texto simple")
#     texto_reemplazado = aplicar_ia_en_texto(texto, resultados_ops, formato="html")
#     return texto_reemplazado

# ----------------------------
# 5) Función que calcula DATOS.
# ----------------------------
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
           reemplazo = f'<pre>{resultado_fmt}</pre>'
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

@register("extraer_data_blocks")
def extraer_data_blocks(texto: str):
    patron = re.compile(r"\|\|(.*?)\|\|", re.DOTALL)
    bloques = []
    for i, match in enumerate(patron.finditer(texto), start=1):
        bloques.append({
            "idx": i,
            "prompt": match.group(1),
            "span": match.span(),
        })
    return bloques


# @register("process_data_blocks")
# def process_data_blocks(df: pd.DataFrame, texto: str):
#     """
#     Réplica del pipeline_graficos_gpt5_final pero para OPERACIONES:

#     - operaciones_gpt5(df, texto) -> lista JSON de objetos con {idx, params, ...}
#     - ejecutar_operaciones_condicionales(df, params) -> ejecuta la operación
#     - _format_result_plain(resultado) -> string final a insertar
#     - aplicar_operaciones_en_texto(texto, resultados_ops, formato="html") -> reemplaza ||...||
#     """
#     # 1) Ejecutar operaciones_gpt5 y asegurar conversión a JSON con spans
#     extract = extraer_data_blocks(texto)
#     out = operaciones_gpt5(df, extract)
 
#     if isinstance(out, str):
#         try:
#             out = json.loads(out)
#         except json.JSONDecodeError as e:
#             raise ValueError(f"JSON inválido: {e}") from e

#     if not isinstance(out, list):
#         raise TypeError("El resultado de operaciones_gpt5 debe ser una lista de objetos JSON.")

#     resultados_ops: List[
#         Tuple[int, Dict[str, Any], Union[Tuple[int, int], None], str]
#     ] = []

#     # 2) Ejecutar cada operación y formatear resultado
#     for item in out:
#         if isinstance(item, dict) and "params" in item:
#             idx = item.get("idx")
#             params = item.get("params")
#             span = item.get("span")

#             try:
#                 resultado = ejecutar_operaciones_condicionales(df, params)
#                 resultado_fmt = _format_result_plain(resultado)
#                 resultados_ops.append((idx, params, span, resultado_fmt))
#             except Exception as e:
#                 # Mantener trazabilidad sin romper el tipo (resultado como string legible)
#                 error_txt = f"[error:{str(e)}]"
#                 resultados_ops.append((idx, params, span, error_txt))

#     # 3) Reemplazar spans por resultados (elige "html" o "texto simple")
#     texto_reemplazado = aplicar_operaciones_en_texto(texto, resultados_ops, formato="html")

#     return texto_reemplazado

@register("process_data_blocks")
def process_data_blocks(
    df: pd.DataFrame,
    texto: str,
    *,
    batch_size: int = 7,
    max_workers: int = 4,
    debug: bool = False,
):
    """
    Versión FINAL paralelizada del orquestador de OPERACIONES:

    - Batching de bloques ||...|| (optimiza costo IA)
    - Paralelización de operaciones_gpt5 (optimiza tiempo)
    - Ejecución y reemplazo determinista
    """

    import concurrent.futures
    import json

    # -------------------------------------------------
    # 1️⃣ Extraer bloques
    # -------------------------------------------------
    extract = extraer_data_blocks(texto)
    if not extract:
        return texto

    def _chunk_list(lst, size):
        for i in range(0, len(lst), size):
            yield lst[i:i + size]

    # -------------------------------------------------
    # 2️⃣ Crear jobs (batches de bloques)
    # -------------------------------------------------
    batches = list(_chunk_list(extract, batch_size))

    # -------------------------------------------------
    # 3️⃣ Job puro IA (BATCH)
    # -------------------------------------------------
    def _procesar_batch(batch):
        try:
            out = operaciones_gpt5(df, batch)
            if isinstance(out, str):
                out = json.loads(out)
        except Exception as e:
            if debug:
                print("Error IA batch:", e)
            return []

        if not isinstance(out, list):
            return []

        return out  # lista de {idx, params, span}

    # -------------------------------------------------
    # 4️⃣ Ejecutar IA en paralelo
    # -------------------------------------------------
    resultados_por_idx = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_procesar_batch, batch) for batch in batches]

        for future in concurrent.futures.as_completed(futures):
            try:
                out_batch = future.result()
                for item in out_batch:
                    if isinstance(item, dict) and "params" in item:
                        idx = item.get("idx")
                        resultados_por_idx[idx] = item
            except Exception as e:
                if debug:
                    print("Error future batch:", e)

    # -------------------------------------------------
    # 5️⃣ Ejecutar operaciones (SECUENCIAL, RÁPIDO)
    # -------------------------------------------------
    resultados_ops = []

    for bloque in sorted(extract, key=lambda b: b["idx"]):
        idx = bloque["idx"]
        span = bloque["span"]

        item = resultados_por_idx.get(idx)
        if not item:
            continue

        params = item.get("params")

        try:
            resultado = ejecutar_operaciones_condicionales(df, params)
            resultado_fmt = _format_result_plain(resultado)
            resultados_ops.append((idx, params, span, resultado_fmt))
        except Exception as e:
            error_txt = f"[error:{str(e)}]"
            resultados_ops.append((idx, params, span, error_txt))

    # -------------------------------------------------
    # 6️⃣ Reemplazar spans en texto
    # -------------------------------------------------
    texto_reemplazado = aplicar_operaciones_en_texto(
        texto,
        resultados_ops,
        formato="html"
    )

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

import re

@register("extraer_plot_blocks")
def extraer_plot_blocks(texto: str):
    """
    Extrae todo lo que esté entre # ... # en el texto.
    
    Ejemplo:
        #GRAFICA#
        # Instrucción de gráfica #
    
    Devuelve:
    [
        {"idx": 1, "prompt": "GRAFICA", "span": (start,end)},
        {"idx": 2, "prompt": "Instrucción de gráfica", "span": (start,end)}
    ]
    """
    patron = re.compile(r"#\s*(.*?)\s*#", re.DOTALL)
    
    bloques = []
    for i, match in enumerate(patron.finditer(texto), start=1):
        bloques.append({
            "idx": i,
            "prompt": match.group(1).strip(),
            "span": match.span(),
        })
    
    return bloques

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


def process_plot_blocks(
    df: pd.DataFrame,
    texto: str,
    *,
    batch_size: int = 6,
    max_workers: int = 4,
    debug: bool = False,
):
    """
    Versión FINAL paralelizada del orquestador de GRÁFICAS:

    - Batching de bloques #...# (optimiza costo IA)
    - Paralelización de graficos_gpt5 (optimiza tiempo)
    - Render de gráficas y reemplazo determinista
    """

    import concurrent.futures
    import json

    # -------------------------------------------------
    # 1️⃣ Extraer bloques
    # -------------------------------------------------
    extract = extraer_plot_blocks(texto)
    if not extract:
        return texto

    def _chunk_list(lst, size):
        for i in range(0, len(lst), size):
            yield lst[i:i + size]

    # -------------------------------------------------
    # 2️⃣ Crear batches
    # -------------------------------------------------
    batches = list(_chunk_list(extract, batch_size))

    # -------------------------------------------------
    # 3️⃣ Job IA (batch)
    # -------------------------------------------------
    def _procesar_batch(batch):
        try:
            out = graficos_gpt5(df, batch)
            if isinstance(out, str):
                try:
                    out = _json_loads_loose(out)
                except Exception as e:
                    if debug:
                        print("Error parseando JSON gráficos:", e)
                    return []

            if not isinstance(out, list):
                return []
        except Exception as e:
            if debug:
                print("Error IA graficos batch:", e)
            return []

        if not isinstance(out, list):
            return []

        return out  # [{idx, params, span}...]

    # -------------------------------------------------
    # 4️⃣ Ejecutar IA en paralelo
    # -------------------------------------------------
    resultados_por_idx = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_procesar_batch, batch) for batch in batches]

        for future in concurrent.futures.as_completed(futures):
            try:
                out_batch = future.result()
                for item in out_batch:
                    if isinstance(item, dict) and "params" in item:
                        resultados_por_idx[item["idx"]] = item
            except Exception as e:
                if debug:
                    print("Error future gráficos:", e)
    print(resultados_por_idx)
    # -------------------------------------------------
    # 5️⃣ Renderizar gráficas (SECUENCIAL)
    # -------------------------------------------------
    resultados_graficos = []

    for bloque in sorted(extract, key=lambda b: b["idx"]):
        idx = bloque["idx"]
        span = bloque["span"]

        item = resultados_por_idx.get(idx)
        if not item:
            continue

        params = item.get("params")

        try:
            fig, ax = plot_from_params(df, params)
            uri = _fig_to_data_uri(fig)
            resultados_graficos.append((idx, params, span, uri))
        except Exception as e:
            error_uri = f"data:text/plain;base64,{_to_base64(f'error:{str(e)}')}"
            resultados_graficos.append((idx, params, span, error_uri))

    # -------------------------------------------------
    # 6️⃣ Reemplazar en texto
    # -------------------------------------------------
    texto_reemplazado = aplicar_graficos_en_texto(
        texto,
        resultados_graficos,
        formato="html"
    )

    return texto_reemplazado


# def process_plot_blocks(df: pd.DataFrame, texto: str):
#     # Ejecutar graficos_gpt5 y asegurar conversión a JSON
#     extract = extraer_plot_blocks(texto)
#     out = graficos_gpt5(df, extract)
#     if isinstance(out, str):
#         try:
#             out = json.loads(out)
#         except json.JSONDecodeError as e:
#             raise ValueError(f"JSON inválido: {e}") from e

#     # Validar tipo
#     if not isinstance(out, list):
#         raise TypeError("El resultado de graficos_gpt5 debe ser una lista de objetos JSON.")

#     params_list = []
#     resultados_graficos: List[
#         Tuple[int, Dict[str, Any], Union[Tuple[int, int], None], Union[str, Dict[str, str]]]
#     ] = []
#     for item in out:
#         if isinstance(item, dict) and "params" in item:
#             idx, params, span = item["idx"], item["params"], item["span"]
#             params_list.append(params)
#             try:
#                 fig, ax = plot_from_params(df, params)
#                 resultados = _fig_to_data_uri(fig)
#                 resultados_graficos.append((idx, params, span, resultados))
#             except Exception as e:
#                 # En caso de error, aún registramos la entrada para trazabilidad (sin romper el tipo)
#                 error_uri = f"data:text/plain;base64,{_to_base64(f'error:{str(e)}')}"
#                 resultados_graficos.append((idx, params, span, error_uri))

#     # 3) Reemplazar spans por data-URIs (elige el formato que prefieras: "uri" | "md" | "html")
#     texto_reemplazado = aplicar_graficos_en_texto(texto, resultados_graficos, formato="html")

#     return texto_reemplazado


# @register("aplicar_multiples_columnas_gpt5")
# def aplicar_multiples_columnas_gpt5(
#     df: pd.DataFrame,
#     tareas: List[Dict[str, Any]],
#     *,
#     replace_existing: bool = True,
#     chunk_tareas: int = 1,
#     chunk_registros: int = 300,
#     debug: bool = False,
#     fn_ia = columns_batch_gpt5
# ):
#     """
#     Versión optimizada con CLUSTERING:
#     Agrupa registros con los mismos valores en columnas_necesarias
#     para reducir llamadas a IA sin alterar la lógica original.
#     """

#     import json
#     import math
#     from collections import defaultdict

#     if not tareas:
#         return df

#     df_out = df.copy()

#     def _chunk_list(lst, size):
#         for i in range(0, len(lst), size):
#             yield lst[i:i+size]

#     # Inicializar columnas de salida
#     for tarea in tareas:
#         col = tarea["nueva_columna"]
#         if col not in df_out.columns:
#             df_out[col] = None

#     # Procesamiento por batches de tareas
#     for tareas_batch in _chunk_list(tareas, chunk_tareas):

#         # 1️⃣ Determinar qué columnas necesita IA
#         columnas_necesarias = set()
#         for tarea in tareas_batch:
#             cols = tarea["registro_cols"]
#             if isinstance(cols, str):
#                 columnas_necesarias.add(cols)
#             else:
#                 columnas_necesarias.update(cols)
#         columnas_necesarias = list(columnas_necesarias)

#         # 2️⃣ Preparar metadata de tareas para IA
#         tareas_para_ia = []
#         for tarea in tareas_batch:
#             tareas_para_ia.append({
#                 "columna": tarea["nueva_columna"],
#                 "criterios": tarea["criterios"],
#                 "registro_cols": tarea["registro_cols"]
#             })

#         # Procesar registros por batches
#         for reg_indices in _chunk_list(df_out.index.tolist(), chunk_registros):

#             # --- CLUSTERING: agrupar registros idénticos ---
#             clusters = defaultdict(list)   # key → lista de ids
#             registros_unicos = {}          # key → registro dict

#             for idx in reg_indices:
#                 row = df_out.loc[idx]
#                 registro = {col: row[col] for col in columnas_necesarias}

#                 # llave determinista para agrupar
#                 key = tuple(registro[col] for col in columnas_necesarias)

#                 clusters[key].append(idx)

#                 # Guardar solo un ejemplo por grupo
#                 if key not in registros_unicos:
#                     registros_unicos[key] = registro

#             # Construir registros compactados para IA
#             registros_para_ia = []
#             for group_id, (key, registro) in enumerate(registros_unicos.items()):
#                 registros_para_ia.append({
#                     "idx": f"G{group_id}",   # id sintético del grupo
#                     "registro": registro
#                 })

#             # Payload final para IA
#             payload = {
#                 "Tareas": tareas_para_ia,
#                 "Registros": registros_para_ia
#             }

#             # --- Llamar IA ---
#             try:
#                 respuesta = fn_ia(payload)
#             except Exception as e:
#                 print("Error IA:", e)
#                 continue

#             resultados = respuesta.get("Resultados", {})

#             # --- Reasignar resultados a TODOS los ids del grupo ---
#             for colname, lista_items in resultados.items():

#                 # Mapa: id_ia → etiqueta
#                 mapa_respuestas = {}
#                 for item in lista_items:
#                     id_ia = item.get("id", item.get("idx"))
#                     val = item.get("resultado", item.get("etiqueta"))
#                     mapa_respuestas[id_ia] = val

#                 # Reaplicar resultados a cada grupo
#                 for group_id, (key, _) in enumerate(registros_unicos.items()):
#                     etiqueta = mapa_respuestas.get(f"G{group_id}")

#                     if etiqueta is None:
#                         continue

#                     # asignar a todos los registros originales
#                     for rid in clusters[key]:
#                         if rid in df_out.index:
#                             df_out.at[rid, colname] = etiqueta

#     return df_out


@register("aplicar_multiples_columnas_gpt5")
def aplicar_multiples_columnas_gpt5(
    df: pd.DataFrame,
    tareas: List[Dict[str, Any]],
    *,
    replace_existing: bool = True,
    chunk_tareas: int = 1,
    chunk_registros: int = 300,
    max_workers: int = 8,
    debug: bool = False,
    fn_ia=columns_batch_gpt5
):
    """
    Versión FINAL paralelizada.
    - Mantiene clustering de registros
    - Paraleliza llamadas IA (ThreadPoolExecutor)
    - Reinyecta resultados de forma segura
    """

    import concurrent.futures
    from collections import defaultdict
    import itertools

    if not tareas:
        return df

    df_out = df.copy()

    def _chunk_list(lst, size):
        for i in range(0, len(lst), size):
            yield lst[i:i + size]

    # -------------------------------------------------
    # 1️⃣ Inicializar columnas de salida
    # -------------------------------------------------
    for tarea in tareas:
        col = tarea["nueva_columna"]
        if col not in df_out.columns:
            df_out[col] = None

    # -------------------------------------------------
    # 2️⃣ Construcción de JOBS (unidad paralelizable)
    # -------------------------------------------------
    jobs = []

    for tareas_batch in _chunk_list(tareas, chunk_tareas):

        # columnas necesarias para este batch
        columnas_necesarias = set()
        for tarea in tareas_batch:
            cols = tarea["registro_cols"]
            if isinstance(cols, str):
                columnas_necesarias.add(cols)
            else:
                columnas_necesarias.update(cols)
        columnas_necesarias = list(columnas_necesarias)

        # metadata tareas IA
        tareas_para_ia = [
            {
                "columna": tarea["nueva_columna"],
                "criterios": tarea["criterios"],
                "registro_cols": tarea["registro_cols"]
            }
            for tarea in tareas_batch
        ]

        for reg_indices in _chunk_list(df_out.index.tolist(), chunk_registros):
            jobs.append({
                "tareas_para_ia": tareas_para_ia,
                "columnas_necesarias": columnas_necesarias,
                "reg_indices": reg_indices
            })

    # -------------------------------------------------
    # 3️⃣ Función ejecutada en paralelo (PURA)
    # -------------------------------------------------
    def _procesar_job(job):
        clusters = defaultdict(list)
        registros_unicos = {}

        # clustering
        for idx in job["reg_indices"]:
            row = df_out.loc[idx]
            registro = {col: row[col] for col in job["columnas_necesarias"]}
            key = tuple(registro[col] for col in job["columnas_necesarias"])

            clusters[key].append(idx)
            if key not in registros_unicos:
                registros_unicos[key] = registro

        # registros compactados
        registros_para_ia = []
        for group_id, registro in enumerate(registros_unicos.values()):
            registros_para_ia.append({
                "idx": f"G{group_id}",
                "registro": registro
            })

        payload = {
            "Tareas": job["tareas_para_ia"],
            "Registros": registros_para_ia
        }

        try:
            respuesta = fn_ia(payload)
        except Exception as e:
            if debug:
                print("Error IA:", e)
            return []

        resultados = respuesta.get("Resultados", {})
        updates = []

        # reconstrucción resultados → ids reales
        for colname, lista_items in resultados.items():
            mapa_respuestas = {}
            for item in lista_items:
                id_ia = item.get("id", item.get("idx"))
                val = item.get("resultado", item.get("etiqueta"))
                mapa_respuestas[id_ia] = val

            for group_id, key in enumerate(registros_unicos.keys()):
                etiqueta = mapa_respuestas.get(f"G{group_id}")
                if etiqueta is None:
                    continue

                for rid in clusters[key]:
                    updates.append((rid, colname, etiqueta))

        return updates

    # -------------------------------------------------
    # 4️⃣ Ejecución paralela
    # -------------------------------------------------
    all_updates = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_procesar_job, job) for job in jobs]

        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                all_updates.extend(result)
            except Exception as e:
                if debug:
                    print("Error en job:", e)

    # -------------------------------------------------
    # 5️⃣ Aplicar resultados (hilo principal)
    # -------------------------------------------------
    for rid, colname, etiqueta in all_updates:
        if rid in df_out.index:
            df_out.at[rid, colname] = etiqueta

    return df_out




# @register("aplicar_multiples_columnas_gpt5")
# def aplicar_multiples_columnas_gpt5(
#     df: pd.DataFrame,
#     tareas: List[Dict[str, Any]],
#     *,
#     replace_existing: bool = True,
#     chunk_tareas: int = 3,
#     chunk_registros: int = 200,   # ahora sí será suficiente
#     debug: bool = False,
#     fn_ia = columns_batch_gpt5
# ):

#     import json
#     import math

#     if not tareas:
#         return df

#     df_out = df.copy()

#     def _chunk_list(lst, size):
#         for i in range(0, len(lst), size):
#             yield lst[i:i+size]

#     # Inicializar columnas de salida
#     for tarea in tareas:
#         col = tarea["nueva_columna"]
#         if col not in df_out.columns:
#             df_out[col] = None

#     # Procesamiento por batches de tareas
#     for tareas_batch in _chunk_list(tareas, chunk_tareas):

#         # Calcular columnas necesarias del registro (UNIÓN)
#         columnas_necesarias = set()
#         for tarea in tareas_batch:
#             cols = tarea["registro_cols"]
#             if isinstance(cols, str):
#                 columnas_necesarias.add(cols)
#             else:
#                 columnas_necesarias.update(cols)

#         columnas_necesarias = list(columnas_necesarias)

#         # Preparación de tareas
#         tareas_para_ia = []
#         for tarea in tareas_batch:
#             tareas_para_ia.append({
#                 "columna": tarea["nueva_columna"],
#                 "criterios": tarea["criterios"],
#                 "registro_cols": tarea["registro_cols"]
#             })

#         # Chunkear registros
#         for reg_indices in _chunk_list(df_out.index.tolist(), chunk_registros):

#             registros = []
#             for idx in reg_indices:
#                 row = df_out.loc[idx]
#                 registro = {col: row[col] for col in columnas_necesarias}
#                 registros.append({"idx": idx, "registro": registro})

#             payload = {
#                 "Tareas": tareas_para_ia,
#                 "Registros": registros
#             }

#             try:
#                 respuesta = fn_ia(payload)
#             except Exception as e:
#                 print("Error IA:", e)
#                 continue

#             resultados = respuesta.get("Resultados", {})
#             for colname, lista_items in resultados.items():
#                 for item in lista_items:
#                     rid = item.get("id", item.get("idx"))
#                     val = item.get("resultado", item.get("etiqueta"))
#                     if rid in df_out.index:
#                         df_out.at[rid, colname] = val

#     return df_out


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

def columnas_a_texto(df: pd.DataFrame, col1: str, col2: str, *, 
                     sep: str = "\n\n", dropna: bool = True, strip: bool = True) -> str:
    """
    Igual que antes, pero ahora cada par (col1, col2) queda envuelto en:
        <div class="Apendice {idx}"> ... </div>
    donde idx es el número de fila (comenzando en 1).
    """

    # Validaciones
    for c in (col1, col2):
        if c not in df.columns:
            raise ValueError(f"La columna '{c}' no existe en el DataFrame.")

    bloques = []  # aquí se almacenará cada Apendice idx
    apendice_idx = 1

    # Procesar fila por fila
    for a, b in df[[col1, col2]].itertuples(index=False, name=None):

        partes = []  # contenido interno del Apendice

        for idx, v in enumerate((a, b)):
            if v is None or (isinstance(v, float) and pd.isna(v)):
                if dropna:
                    continue
                v = ""

            s = str(v)
            if strip:
                s = s.strip()

            # col1 → Título
            if idx == 0:
                    s = f'<span class="titulo">{s}</span>'

            partes.append(s)

        # Construir el bloque del apéndice solo si tiene contenido
        if partes:
            bloque = f'<div class="Apendice {apendice_idx}">' + sep.join(partes) + '</div>'
            bloques.append(bloque)
            apendice_idx += 1

    return sep.join(bloques)



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


import re

def filtrar_apendices(texto: str, agente_result: dict, *, renumerar=True):
    """
    Filtra bloques <div class="Apendice N">...</div> de acuerdo con la salida
    del agente de clasificación.

    El diccionario agente_result DEBE tener la forma:
    {
        "conservar": [1, 2, ...],
        "borrar": [3, 4, ...]
    }

    Reglas:
    - Si un índice está en 'borrar', SIEMPRE se elimina aunque aparezca en 'conservar'.
    - Si 'conservar' está vacío pero 'borrar' tiene elementos → se conservan los demás.
    - Si ambos están vacíos → no se modifica nada.
    - Si un índice no existe en el texto → se ignora silenciosamente.
    - Después del filtrado, si renumerar=True, los apéndices se renumeran 1..N.
    """

    # Validación mínima de entrada
    if not isinstance(agente_result, dict):
        raise ValueError("agente_result debe ser un diccionario con 'conservar' y 'borrar'.")

    conservar = agente_result.get("conservar", [])
    borrar = agente_result.get("borrar", [])

    # Normalización
    conservar = set(int(x) for x in conservar if isinstance(x, int) or str(x).isdigit())
    borrar = set(int(x) for x in borrar if isinstance(x, int) or str(x).isdigit())

    # Si un índice aparece en ambos → se borra
    borrar = borrar.union(conservar.intersection(borrar))

    # Regex para capturar apéndices completos
    patron = re.compile(
        r'<div class="Apendice\s+(\d+)">(.*?)</div>',
        re.S | re.I
    )

    bloques = patron.findall(texto)  # lista de (idx_str, contenido)

    if not bloques:
        return texto  # no hay apéndices → retornar sin cambios

    nuevos_bloques = []

    for idx_str, contenido in bloques:
        idx = int(idx_str)

        # 1) Prioridad absoluta: si está en BORRAR → se elimina
        if idx in borrar:
            continue

        # 2) Si 'conservar' está vacío → conservar todos los que no estén en borrar
        if not conservar:
            nuevos_bloques.append((idx, contenido))
            continue

        # 3) Si 'conservar' NO está vacío → conservar solo esos índices
        if idx in conservar:
            nuevos_bloques.append((idx, contenido))

    # Si nada queda después del filtrado
    if not nuevos_bloques:
        return ""  # o retorna texto, según tu política

    # Reconstrucción del mensaje final
    salida = []
    if renumerar:
        for nuevo_idx, (_, contenido) in enumerate(nuevos_bloques, start=1):
            salida.append(f'<div class="Apendice {nuevo_idx}">{contenido}</div>')
    else:
        for (idx, contenido) in nuevos_bloques:
            salida.append(f'<div class="Apendice {idx}">{contenido}</div>')

    return "\n\n".join(salida)


import json
import re

def normalizar_salida_agente(salida):
    """
    Normaliza lo que devuelva apendices_gpt5 en un dict seguro:
    {
        "conservar": [...],
        "borrar": [...]
    }
    """

    # ============================
    # 1. SI ES STRING → intentar parse JSON
    # ============================
    if isinstance(salida, str):
        s = salida.strip()

        # Intento 1: JSON directo
        try:
            salida = json.loads(s)
        except Exception:
            # Intento 2: extraer JSON dentro de texto más largo
            match = re.search(r"\{.*\}", s, re.S)
            if match:
                try:
                    salida = json.loads(match.group(0))
                except Exception:
                    salida = {}
            else:
                salida = {}

    # ============================
    # 2. SI NO ES DICT → forzar dict vacío
    # ============================
    if not isinstance(salida, dict):
        salida = {}

    # ============================
    # 3. CREAR LLAVES SI NO EXISTEN
    # ============================
    conservar = salida.get("conservar", [])
    borrar = salida.get("borrar", [])

    # ============================
    # 4. FORZAR LISTAS
    # ============================
    if not isinstance(conservar, list):
        conservar = []
    if not isinstance(borrar, list):
        borrar = []

    # ============================
    # 5. CONVERTIR A ENTEROS VÁLIDOS
    # ============================
    conservar = [int(x) for x in conservar if str(x).isdigit()]
    borrar = [int(x) for x in borrar if str(x).isdigit()]

    return {"conservar": conservar, "borrar": borrar}


def procesar_apendices(texto: str, *, renumerar=True) -> str:
    """
    Orquesta el proceso:
    1. Ejecuta apendices_gpt5(texto)
    2. Normaliza su salida para garantizar dict válido
    3. Aplica filtrar_apendices()
    """

    # ============================================================
    # 1. EJECUTAR AGENTE
    # ============================================================
    try:
        agente_raw = apendices_gpt5(texto)
    except Exception as e:
        raise RuntimeError(f"Error al ejecutar apendices_gpt5: {e}")

    # ============================================================
    # 2. NORMALIZAR SALIDA DEL AGENTE (NUNCA lanza excepción)
    # ============================================================
    agente_result = normalizar_salida_agente(agente_raw)

    # ============================================================
    # 3. FILTRAR APÉNDICES
    # ============================================================
    try:
        texto_filtrado = filtrar_apendices(texto, agente_result, renumerar=renumerar)
    except Exception as e:
        raise RuntimeError(f"Error al filtrar apéndices: {e}")

    return texto_filtrado


@register("extraer_titulos")
def extraer_titulos(texto: str):
    """
    Extrae todos los títulos del tipo:
        <span class="titulo">TEXTO</span>

    Devuelve una lista de dicts:
    [
        {
            "idx": 1,
            "titulo": "INTRODUCCIÓN",
            "span": (start, end)
        },
        ...
    ]
    """

    # Patrón robusto:
    # - Acepta otros atributos dentro del span
    # - Captura cualquier texto interno
    # - Respeta saltos de línea
    patron = re.compile(
        r'<span\s+class=["\']titulo["\'][^>]*>(.*?)</span>',
        re.I | re.S
    )

    bloques = []

    for i, match in enumerate(patron.finditer(texto), start=1):
        titulo = match.group(1).strip()
        bloques.append({
            "idx": i,
            "titulo": titulo,
            "span": match.span()
        })

    return bloques


def aplicar_titulos_numerados(texto: str, titulos_numerados) -> str:
    """
    Reemplaza en `texto` cada <span class="titulo">...</span> por su versión numerada.

    titulos_numerados:
        Lista de diccionarios con estructura:
        {
            "idx": N,
            "titulo": "1.2 Perfil hematológico",  # título ya numerado
            "span": [start, end]                 # span del título original en el HTML
        }

    Funciona igual que aplicar_operaciones_en_texto:
    - Aplica reemplazos de derecha a izquierda.
    - Preserva la integridad del HTML.
    - Acepta spans tipo lista o diccionario.
    """

    reemplazos = []

    for item in titulos_numerados:
        # Validar estructura mínima
        if not isinstance(item, dict):
            continue

        idx = item.get("idx")
        nuevo_titulo = item.get("titulo")
        span_obj = item.get("span")

        # Validación de título
        if not isinstance(nuevo_titulo, str) or not nuevo_titulo.strip():
            continue

        # Normalizar span
        if isinstance(span_obj, dict):
            span_list = span_obj.get("span") or span_obj.get("spam")
        else:
            span_list = span_obj

        if not (isinstance(span_list, (list, tuple)) and len(span_list) == 2):
            continue

        start, end = span_list
        if not (isinstance(start, int) and isinstance(end, int) and 0 <= start <= end <= len(texto)):
            continue

        # Reemplazo final:
        # Sustituimos TODO el <span class="titulo">...</span> por el nuevo número + título
        reemplazo = f'<span class="titulo">{nuevo_titulo}</span>'

        reemplazos.append((start, end, reemplazo))

    # Aplicar reemplazos de derecha a izquierda para NO dañar offsets
    reemplazos.sort(key=lambda t: t[0], reverse=True)

    out = texto
    for start, end, rep in reemplazos:
        out = out[:start] + rep + out[end:]

    return out

def generar_tabla_contenido(titulos_numerados) -> str:
    """
    Construye un bloque de texto plano con la tabla de contenido.

    Parámetro:
        titulos_numerados: lista de dicts con estructura:
        {
            "idx": N,
            "titulo": "1.2 Perfil hematológico",  # título ya numerado
            "span": [inicio, fin]
        }

    Retorno:
        Texto plano con el encabezado 'Tabla de contenido' seguido por los títulos numerados.
    """

    if not isinstance(titulos_numerados, list):
        raise TypeError("titulos_numerados debe ser una lista de dicts.")

    lineas = ["Tabla de contenido"]

    for item in titulos_numerados:
        if not isinstance(item, dict):
            continue

        titulo = item.get("titulo")
        if isinstance(titulo, str) and titulo.strip():
            lineas.append(titulo.strip())

    return "\n".join(lineas)

def process_titulo_blocks(texto: str) -> str:
    """
    1. Extrae títulos con extraer_titulos(texto)
    2. Ejecuta titulos_gpt5 para numerarlos
    3. Normaliza y valida la salida JSON
    4. Aplica aplicar_titulos_numerados() para reemplazar los títulos en el HTML

    Devuelve el texto final con títulos numerados.
    """

    # ============================================================
    # 1) EXTRAER TITULOS
    # ============================================================
    try:
        extract = extraer_titulos(texto)
    except Exception as e:
        raise RuntimeError(f"Error al extraer títulos: {e}")

    # Si no hay títulos, retornar igual
    if not extract:
        return texto

    # ============================================================
    # 2) EJECUTAR AGENTE titulos_gpt5
    # ============================================================
    try:
        out = titulos_gpt5(extract)
    except Exception as e:
        raise RuntimeError(f"Error al ejecutar titulos_gpt5: {e}")

    # ============================================================
    # 3) NORMALIZAR SALIDA DEL AGENTE
    # ============================================================
    if isinstance(out, str):
        out_str = out.strip()
        try:
            out = json.loads(out_str)
        except Exception:
            # intentar extraer json interno
            match = re.search(r"\[.*\]", out_str, re.S)
            if match:
                try:
                    out = json.loads(match.group(0))
                except Exception:
                    raise ValueError(f"titulos_gpt5 devolvió un JSON inválido: {out_str}")
            else:
                raise ValueError(f"titulos_gpt5 no devolvió un JSON válido: {out_str}")

    # ============================================================
    # 4) VALIDAR TIPO
    # ============================================================
    if not isinstance(out, list):
        raise TypeError("El resultado de titulos_gpt5 debe ser una LISTA de objetos JSON.")

    # Validar estructura mínima de cada item
    for item in out:
        if not isinstance(item, dict):
            raise TypeError("Cada elemento del JSON de salida debe ser un dict.")

        if "idx" not in item or "titulo" not in item or "span" not in item:
            raise ValueError(
                "Cada dict debe tener las llaves 'idx', 'titulo' y 'span'."
            )

    tabla_contenido = generar_tabla_contenido(out)

    # ============================================================
    # 5) REEMPLAZAR TITULOS EN EL HTML
    # ============================================================
    try:
        texto_final = aplicar_titulos_numerados(texto, out)
    except Exception as e:
        raise RuntimeError(f"Error al reemplazar títulos numerados: {e}")

    return texto_final, tabla_contenido
