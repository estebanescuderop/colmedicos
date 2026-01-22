# --- Standard library ---
import os
import re
import json
import time
import base64
import hashlib
import concurrent.futures
from collections import defaultdict
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
            rep = f'<div class="análisis_{idx}"><p>{resultado_ia}</p></div>'
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
    
    json_operaciones = json.dumps(resultados_ops, ensure_ascii=False, indent=2)

    # -------------------------------------------------
    # 6️⃣ Reemplazar spans en texto
    # -------------------------------------------------
    texto_reemplazado = aplicar_operaciones_en_texto(
        texto,
        resultados_ops,
        formato="html"
    )

    return texto_reemplazado, json_operaciones


_TOKEN_RE = re.compile(r"#GRAFICA(?:[_\s]*([0-9]+))?#", flags=re.IGNORECASE)

# ----------------------------
# 5) Función que calcula gráficos.
# ----------------------------

def rescale_image(img, *, max_width: int = 1200, max_height: int = 800):
    """
    Reescala una imagen PIL manteniendo aspect ratio (sin deformar).
    Nunca hace upscale (si ya es más pequeña, la deja igual).
    """
    from PIL import Image

    if not isinstance(img, Image.Image):
        raise TypeError(f"rescale_image espera PIL.Image.Image, recibió: {type(img)}")

    orig_w, orig_h = img.size
    scale_w = max_width / orig_w
    scale_h = max_height / orig_h
    scale = min(scale_w, scale_h, 1.0)  # no upscale

    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)

    if (new_w, new_h) == (orig_w, orig_h):
        return img

    return img.resize((new_w, new_h), Image.LANCZOS)


def _fig_to_data_uri(
    fig,
    *,
    max_width: int = 800,
    max_height: int = 500,
    dpi: int = 150,
    plotly_scale: int = 1.5,
) -> str:
    """
    Convierte figuras Matplotlib o Plotly en data:image/png;base64.
    Reescala imagen final manteniendo proporción y CIERRA recursos.
    """
    import base64
    from io import BytesIO
    from PIL import Image

    buf = BytesIO()

    try:
        # 1) Render a PNG bytes
        if hasattr(fig, "savefig"):  # Matplotlib
            fig.savefig(
                buf,
                format="png",
                dpi=dpi,
                bbox_inches="tight",
                facecolor="white",
                edgecolor="white",
            )
            png_bytes = buf.getvalue()

        elif hasattr(fig, "to_image"):  # Plotly (kaleido)
            png_bytes = fig.to_image(format="png", scale=plotly_scale)

        elif hasattr(fig, "write_image"):  # Plotly compat
            fig.write_image(buf, format="png", engine="kaleido")
            png_bytes = buf.getvalue()

        else:
            raise TypeError(f"Tipo de figura no soportado: {type(fig)}")

        # 2) Reescalar con PIL
        img = Image.open(BytesIO(png_bytes))
        try:
            img = rescale_image(img, max_width=max_width, max_height=max_height)

            out = BytesIO()
            try:
                img.save(out, format="PNG", optimize=True)
                out.seek(0)

                b64 = base64.b64encode(out.read()).decode("ascii")
                return f"data:image/png;base64,{b64}"

            finally:
                out.close()

        finally:
            img.close()

    except Exception as e:
        msg = f"error:{str(e)}"
        return f"data:text/plain;base64,{base64.b64encode(msg.encode()).decode()}"

    finally:
        # ✅ Cerrar buffer principal siempre
        buf.close()

        # ✅ Matplotlib: cerrar figura (EVITA fuga de RAM)
        if hasattr(fig, "savefig"):
            try:
                import matplotlib.pyplot as plt
                plt.close(fig)
            except Exception:
                pass


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


def _fig_is_empty(fig) -> bool:
    """
    Determina si una figura salió 'vacía' (sin data visible).
    Soporta Matplotlib y Plotly de forma simple.
    """
    if fig is None:
        return True

    # Matplotlib
    if hasattr(fig, "axes"):
        if not fig.axes:
            return True
        # si ninguna axis tiene data, asumimos vacío
        has_any_data = False
        for ax in fig.axes:
            try:
                if ax.has_data():
                    has_any_data = True
                    break
            except Exception:
                pass
        return not has_any_data

    # Plotly
    if hasattr(fig, "data"):
        try:
            return len(fig.data) == 0
        except Exception:
            return True

    # Si no sabemos identificarlo, asumimos NO vacío
    return False


def validar_params_grafica_por_ejecucion(df, params: dict) -> tuple[bool, str]:
    """
    Valida ejecutando el graficador real.
    Si falla o queda vacío, devuelve (False, motivo).
    """
    try:
        fig_ax = plot_from_params(df, params)

        # plot_from_params suele devolver (fig, ax)
        if isinstance(fig_ax, tuple) and len(fig_ax) >= 1:
            fig = fig_ax[0]
        else:
            fig = fig_ax

        if _fig_is_empty(fig):
            return False, "Figura vacía (sin data)"
        return True, "OK"

    except Exception as e:
        return False, f"Error ejecutando plot_from_params: {type(e).__name__}: {e}"


def generar_params_con_reintento_ia(
    df,
    prompt_grafica: str,
    *,
    max_retries: int = 1,  # por ahora 1 reintento
) -> dict:
    """
    Flujo simplista:
    graficos_gpt5 -> validar (ejecutando plot_from_params)
    si falla o vacío -> graficos_gpt5 otra vez -> validar -> si ok retorna
    """
    last_reason = None

    for attempt in range(max_retries + 1):
        params = graficos_gpt5(df, prompt_grafica)

        ok, reason = validar_params_grafica_por_ejecucion(df, params)
        if ok:
            return params

        last_reason = reason

    # Si ya no hay reintentos
    raise ValueError(
        "No se logró generar una gráfica operable tras reintentos.\n"
        f"Motivo final: {last_reason}"
    )

def process_plot_blocks(
    df: pd.DataFrame,
    texto: str,
    *,
    batch_size: int = 6,
    max_workers: int = 2,
    debug: bool = False,
):
    """
    Versión FINAL paralelizada del orquestador de GRÁFICAS:

    - Batching de bloques #...# (optimiza costo IA)
    - Paralelización de graficos_gpt5 (optimiza tiempo)
    - Render de gráficas y reemplazo determinista

    🔥 NUEVO (simplista):
    - Prevalida por ejecución real:
      plot_from_params(df, params)
      si falla o figura vacía -> reintenta IA 1 vez (solo para ese bloque)
    """

    import concurrent.futures
    import json

    # -------------------------------------------------
    # 0️⃣ Helpers ultra simples (NO cambian tu lógica)
    # -------------------------------------------------
    def _fig_is_empty(fig) -> bool:
        """
        Determina si una figura salió vacía (sin data visible).
        Soporta Matplotlib y Plotly de forma simple.
        """
        if fig is None:
            return True

        # Matplotlib
        if hasattr(fig, "axes"):
            if not fig.axes:
                return True
            has_any_data = False
            for ax in fig.axes:
                try:
                    if ax.has_data():
                        has_any_data = True
                        break
                except Exception:
                    pass
            return not has_any_data

        # Plotly
        if hasattr(fig, "data"):
            try:
                return len(fig.data) == 0
            except Exception:
                return True

        # Si no sabemos identificarlo, asumimos NO vacío
        return False

    def _try_plot(df, params):
        """
        Intenta graficar. Si hay error o figura vacía, levanta excepción.
        """
        fig, ax = plot_from_params(df, params)
        if _fig_is_empty(fig):
            raise ValueError("Figura vacía (sin data)")
        return fig, ax

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

    json_grafos = json.dumps(resultados_por_idx, ensure_ascii=False, indent=2)

    # -------------------------------------------------
    # 5️⃣ Renderizar gráficas (SECUENCIAL)
    #    + REINTENTO IA si falla / vacío (1 vez)
    # -------------------------------------------------
    resultados_graficos = []

    for bloque in sorted(extract, key=lambda b: b["idx"]):
        idx = bloque["idx"]
        span = bloque["span"]

        item = resultados_por_idx.get(idx)
        if not item:
            continue

        params = item.get("params")

        # ---- 1er intento normal ----
        try:
            fig, ax = _try_plot(df, params)
            uri = _fig_to_data_uri(fig)
            resultados_graficos.append((idx, params, span, uri))
            continue

        except Exception as e1:
            # ✅ NUEVO: reintento IA simplista solo para este bloque
            if debug:
                print(f"[plot retry] idx={idx} falló primer intento:", str(e1))

            try:
                # mini-batch con solo este bloque
                out_retry = _procesar_batch([bloque])

                # out_retry debe ser lista de dicts con idx/params
                retry_item = None
                if isinstance(out_retry, list):
                    for it in out_retry:
                        if isinstance(it, dict) and it.get("idx") == idx and "params" in it:
                            retry_item = it
                            break

                if retry_item and isinstance(retry_item.get("params"), dict):
                    # actualizamos params para el render
                    params = retry_item["params"]
                    resultados_por_idx[idx] = retry_item  # mantiene consistencia interna

                    # ---- 2do intento con params nuevos ----
                    fig, ax = _try_plot(df, params)
                    uri = _fig_to_data_uri(fig)
                    resultados_graficos.append((idx, params, span, uri))
                    continue

                else:
                    # no pudimos obtener params válidos del reintento
                    raise ValueError("Reintento IA no devolvió params válidos para esta gráfica")

            except Exception as e2:
                # fallback final: tu lógica original de error_uri
                if debug:
                    print(f"[plot retry] idx={idx} también falló reintento:", str(e2))

                error_uri = f"data:text/plain;base64,{_to_base64(f'error:{str(e2)}')}"
                resultados_graficos.append((idx, params, span, error_uri))

    # -------------------------------------------------
    # 6️⃣ Reemplazar en texto
    # -------------------------------------------------
    texto_reemplazado = aplicar_graficos_en_texto(
        texto,
        resultados_graficos,
        formato="html"
    )

    # ✅ json final actualizado (por si hubo retries que cambiaron params)
    json_grafos = json.dumps(resultados_por_idx, ensure_ascii=False, indent=2)

    return texto_reemplazado, json_grafos


# def process_plot_blocks(
#     df: pd.DataFrame,
#     texto: str,
#     *,
#     batch_size: int = 6,
#     max_workers: int = 2,
#     debug: bool = False,
# ):
#     """
#     Versión FINAL paralelizada del orquestador de GRÁFICAS:

#     - Batching de bloques #...# (optimiza costo IA)
#     - Paralelización de graficos_gpt5 (optimiza tiempo)
#     - Render de gráficas y reemplazo determinista
#     """

#     import concurrent.futures
#     import json

#     # -------------------------------------------------
#     # 1️⃣ Extraer bloques
#     # -------------------------------------------------
#     extract = extraer_plot_blocks(texto)
#     if not extract:
#         return texto

#     def _chunk_list(lst, size):
#         for i in range(0, len(lst), size):
#             yield lst[i:i + size]

#     # -------------------------------------------------
#     # 2️⃣ Crear batches
#     # -------------------------------------------------
#     batches = list(_chunk_list(extract, batch_size))

#     # -------------------------------------------------
#     # 3️⃣ Job IA (batch)
#     # -------------------------------------------------
#     def _procesar_batch(batch):
#         try:
#             out = graficos_gpt5(df, batch)
#             if isinstance(out, str):
#                 try:
#                     out = _json_loads_loose(out)
#                 except Exception as e:
#                     if debug:
#                         print("Error parseando JSON gráficos:", e)
#                     return []

#             if not isinstance(out, list):
#                 return []
#         except Exception as e:
#             if debug:
#                 print("Error IA graficos batch:", e)
#             return []

#         if not isinstance(out, list):
#             return []

#         return out  # [{idx, params, span}...]

#     # -------------------------------------------------
#     # 4️⃣ Ejecutar IA en paralelo
#     # -------------------------------------------------
#     resultados_por_idx = {}

#     with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
#         futures = [executor.submit(_procesar_batch, batch) for batch in batches]

#         for future in concurrent.futures.as_completed(futures):
#             try:
#                 out_batch = future.result()
#                 for item in out_batch:
#                     if isinstance(item, dict) and "params" in item:
#                         resultados_por_idx[item["idx"]] = item
#             except Exception as e:
#                 if debug:
#                     print("Error future gráficos:", e)
#     json_grafos = json.dumps(resultados_por_idx, ensure_ascii=False, indent=2)
#     # -------------------------------------------------
#     # 5️⃣ Renderizar gráficas (SECUENCIAL)
#     # -------------------------------------------------
#     resultados_graficos = []

#     for bloque in sorted(extract, key=lambda b: b["idx"]):
#         idx = bloque["idx"]
#         span = bloque["span"]

#         item = resultados_por_idx.get(idx)
#         if not item:
#             continue

#         params = item.get("params")

#         try:
#             fig, ax = plot_from_params(df, params)
#             uri = _fig_to_data_uri(fig)
#             resultados_graficos.append((idx, params, span, uri))
#         except Exception as e:
#             error_uri = f"data:text/plain;base64,{_to_base64(f'error:{str(e)}')}"
#             resultados_graficos.append((idx, params, span, error_uri))

#     # -------------------------------------------------
#     # 6️⃣ Reemplazar en texto
#     # -------------------------------------------------
#     texto_reemplazado = aplicar_graficos_en_texto(
#         texto,
#         resultados_graficos,
#         formato="html"
#     )

#     return texto_reemplazado, json_grafos


@register("aplicar_multiples_columnas_gpt5")
def aplicar_multiples_columnas_gpt5(
    df: pd.DataFrame,
    tareas: List[Dict[str, Any]],
    *,
    replace_existing: bool = True,
    chunk_tareas: int = 10,      # ✅ OPTIMIZADO: Agrupar más tareas por batch
    chunk_registros: int = 500,  # ✅ OPTIMIZADO: Aumentado de 300 a 500 (menos llamadas)
    max_workers: int = 10,       # ✅ OPTIMIZADO: Aumentado de 8 a 10 (match con semaphore)
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
        import hashlib

        clusters = defaultdict(list)
        registros_unicos = {}

        # OPTIMIZADO: clustering con hash para mejor performance
        def _hash_registro(reg_dict):
            """Genera hash estable para clustering de registros"""
            # Normalizar valores (NaN → None, floats a strings)
            valores = []
            for col in job["columnas_necesarias"]:
                val = reg_dict.get(col)
                if pd.isna(val):
                    valores.append("__NAN__")
                elif isinstance(val, (float, np.floating)):
                    valores.append(f"{val:.6f}")  # Precisión fija
                else:
                    valores.append(str(val))

            # Hash MD5 (rápido, suficiente para clustering)
            clave = "|".join(valores)
            return hashlib.md5(clave.encode('utf-8')).hexdigest()

        # clustering
        for idx in job["reg_indices"]:
            row = df_out.loc[idx]
            registro = {col: row[col] for col in job["columnas_necesarias"]}
            key = _hash_registro(registro)  # ✅ Hash en lugar de tupla

            clusters[key].append(idx)
            if key not in registros_unicos:
                registros_unicos[key] = registro

        # registros compactados
        from datetime import datetime
        fecha_hoy = datetime.now().strftime("%Y-%m-%d")

        registros_para_ia = []
        for group_id, registro in enumerate(registros_unicos.values()):
            # ✅ FIX CRÍTICO: Agregar fecha_hoy al registro
            # El prompt del sistema dice que fecha_hoy debe estar en el registro
            registro_con_fecha = registro.copy()
            registro_con_fecha["fecha_hoy"] = fecha_hoy

            # ✅ FIX: Convertir tipos no serializables a JSON
            for key, val in registro_con_fecha.items():
                # Timestamps → string
                if pd.api.types.is_datetime64_any_dtype(type(val)) or isinstance(val, pd.Timestamp):
                    registro_con_fecha[key] = val.strftime('%Y-%m-%d %H:%M:%S') if not pd.isna(val) else None
                # numpy int/float → Python int/float
                elif isinstance(val, (np.integer, np.floating)):
                    registro_con_fecha[key] = val.item()  # Convierte np.int64/np.float64 a int/float nativo
                # NaN → None
                elif pd.isna(val):
                    registro_con_fecha[key] = None

            registros_para_ia.append({
                "idx": f"G{group_id}",
                "registro": registro_con_fecha
            })

        payload = {
            "Tareas": job["tareas_para_ia"],
            "Registros": registros_para_ia
        }

        try:
            respuesta = fn_ia(payload, fecha_hoy=fecha_hoy)
        except Exception as e:
            if debug:
                print("Error IA:", e)
            return []

        resultados = respuesta.get("Resultados", {})
        updates = []

        # ===== DIAGNÓSTICO =====
        import os
        DEBUG_DIAGNOSTICO = os.getenv("DEBUG_DIAGNOSTICO", "0") == "1"

        if DEBUG_DIAGNOSTICO:
            columnas_esperadas = [t["columna"] for t in job["tareas_para_ia"]]
            print(f"\n[DEBUG io_utils_remaster] Procesando respuesta de IA:")
            print(f"  Columnas ESPERADAS: {columnas_esperadas}")
            print(f"  Columnas RECIBIDAS: {list(resultados.keys())}")
            for colname, items in resultados.items():
                en_esperadas = "✅" if colname in columnas_esperadas else "❌ NO ESPERADA"
                print(f"  {colname}: {len(items)} items {en_esperadas}")
        # =======================

        # reconstrucción resultados → ids reales
        for colname, lista_items in resultados.items():
            mapa_respuestas = {}
            for item in lista_items:
                id_ia = item.get("id", item.get("idx"))
                val = item.get("resultado", item.get("etiqueta"))
                # ✅ FIX: Normalizar ID a string para matching consistente
                mapa_respuestas[str(id_ia)] = val

            if DEBUG_DIAGNOSTICO:
                print(f"  {colname} - mapa_respuestas: {mapa_respuestas}")

            matched_count = 0
            for group_id, key in enumerate(registros_unicos.keys()):
                # ✅ FIX: Intentar "G{n}" primero, luego fallback a "{n}" (igual que V2)
                etiqueta = mapa_respuestas.get(f"G{group_id}")
                if etiqueta is None:
                    # Fallback: IA a veces devuelve ID sin prefijo "G"
                    etiqueta = mapa_respuestas.get(str(group_id))
                if etiqueta is None:
                    if DEBUG_DIAGNOSTICO:
                        print(f"    [WARN] G{group_id} no tiene etiqueta en respuesta IA. Keys: {list(mapa_respuestas.keys())[:5]}")
                    continue

                matched_count += 1
                for rid in clusters[key]:
                    updates.append((rid, colname, etiqueta))

            if DEBUG_DIAGNOSTICO:
                print(f"  {colname} - Matched {matched_count}/{len(registros_unicos)} grupos")
                print(f"  {colname} - Generados {len([u for u in updates if u[1] == colname])} updates")

        if DEBUG_DIAGNOSTICO:
            print(f"[DEBUG io_utils_remaster] Total updates generados: {len(updates)}\n")

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
    import os
    DEBUG_DIAGNOSTICO = os.getenv("DEBUG_DIAGNOSTICO", "0") == "1"

    if DEBUG_DIAGNOSTICO:
        print(f"\n[DEBUG io_utils_remaster] Aplicando updates:")
        print(f"  Total updates: {len(all_updates)}")

        # Resumen por columna
        from collections import Counter
        col_counts = Counter(colname for _, colname, _ in all_updates)
        for col, cnt in col_counts.items():
            en_df = "✅" if col in df_out.columns else "❌ NO EXISTE EN DF"
            print(f"  {col}: {cnt} updates {en_df}")

    applied_count = 0
    skipped_idx = 0
    skipped_col = 0

    for rid, colname, etiqueta in all_updates:
        if rid not in df_out.index:
            skipped_idx += 1
            continue
        if colname not in df_out.columns:
            skipped_col += 1
            continue
        df_out.at[rid, colname] = etiqueta
        applied_count += 1

    if DEBUG_DIAGNOSTICO:
        print(f"  Updates aplicados: {applied_count}")
        print(f"  Skipped (índice no existe): {skipped_idx}")
        print(f"  Skipped (columna no existe): {skipped_col}")

    # -------------------------------------------------
    # 6️⃣ Convertir columnas a tipo numérico si es posible
    # -------------------------------------------------
    # Después de aplicar los updates, intentar convertir columnas a numérico
    # Esto preserva int/float en lugar de mantener todo como object
    for tarea in tareas:
        col = tarea["nueva_columna"]
        if col in df_out.columns:
            # Intentar convertir a numérico (errors='ignore' mantiene original si falla)
            df_out[col] = pd.to_numeric(df_out[col], errors='ignore')

    return df_out



@register("aplicar_multiples_columnas_gpt5_ultra_v2")
def aplicar_multiples_columnas_gpt5_ultra_v2(
    df: pd.DataFrame,
    tareas: List[Dict[str, Any]],
    *,
    replace_existing: bool = True,
    chunk_registros_unicos: int = 500,
    max_workers: int = 10,
    batch_timeout: int = 180,
    max_retries: int = 2,
    float_precision: int = 6,
    fn_ia=columns_batch_gpt5,
):
    """
    Versión HÍBRIDA OPTIMIZADA para generación de columnas con IA.

    Combina paralelismo a nivel de tareas con clustering por tarea para
    maximizar velocidad y minimizar tokens enviados a la API.

    Parámetros:
    -----------
    df : pd.DataFrame
        DataFrame con los datos a procesar.
    tareas : List[Dict]
        Lista de tareas. Cada tarea debe tener:
        - "nueva_columna": nombre de la columna a crear
        - "criterios": dict con reglas de clasificación/cálculo
        - "registro_cols": columnas necesarias para la tarea
    replace_existing : bool, default=True
        Si True, reemplaza valores existentes en las columnas.
    chunk_registros_unicos : int, default=500
        Registros únicos por llamada IA (no usado actualmente, reservado).
    max_workers : int, default=10
        Workers paralelos para ejecutar tareas.
    batch_timeout : int, default=180
        Timeout en segundos por tarea.
    max_retries : int, default=2
        Reintentos por tarea si falla.
    float_precision : int, default=6
        Decimales para redondeo en hashing.
    fn_ia : callable, default=columns_batch_gpt5
        Función de IA a utilizar.

    Retorna:
    --------
    pd.DataFrame
        DataFrame con las nuevas columnas agregadas.
    """
    from typing import Tuple
    from datetime import datetime

    if df is None or df.empty or not tareas:
        return df

    df_out = df.copy()

    def _safe_isna(v) -> bool:
        try:
            return pd.isna(v)
        except Exception:
            return v is None

    def _normalize_for_hash(val: Any) -> str:
        if _safe_isna(val):
            return "__NAN__"
        if isinstance(val, (float, np.floating)):
            return f"{float(val):.{float_precision}f}"
        return str(val)

    def _hash_registro(registro: dict, cols: List[str]) -> str:
        clave = "|".join(_normalize_for_hash(registro.get(c)) for c in cols)
        return hashlib.md5(clave.encode("utf-8")).hexdigest()

    # Inicializar columnas de salida
    for tarea in tareas:
        col = tarea.get("nueva_columna")
        if col and col not in df_out.columns:
            df_out[col] = None

    def _procesar_tarea(tarea: Dict[str, Any], _: int) -> List[Tuple[Any, str, Any]]:
        """Procesa una tarea completa con clustering."""
        col_out = tarea.get("nueva_columna")
        criterios = tarea.get("criterios")
        reg_cols = tarea.get("registro_cols", [])

        if not col_out or criterios is None:
            return []

        if isinstance(reg_cols, str):
            columnas_tarea = [reg_cols]
        else:
            columnas_tarea = list(reg_cols)

        columnas_tarea = [c for c in columnas_tarea if c in df_out.columns]
        if not columnas_tarea:
            return []

        # Clustering para esta tarea
        clusters = defaultdict(list)
        registros_unicos = {}

        for idx in df_out.index:
            row = df_out.loc[idx]
            registro = {c: row[c] for c in columnas_tarea}
            key = _hash_registro(registro, columnas_tarea)
            clusters[key].append(idx)
            if key not in registros_unicos:
                registros_unicos[key] = registro

        # Preparar payload
        fecha_hoy = datetime.now().strftime("%Y-%m-%d")
        registros_para_ia = []
        unique_keys = list(registros_unicos.keys())

        for group_id, key in enumerate(unique_keys):
            registro = registros_unicos[key].copy()
            registro["fecha_hoy"] = fecha_hoy

            for k, v in registro.items():
                if pd.api.types.is_datetime64_any_dtype(type(v)) or isinstance(v, pd.Timestamp):
                    registro[k] = v.strftime('%Y-%m-%d %H:%M:%S') if not _safe_isna(v) else None
                elif isinstance(v, (np.integer, np.floating)):
                    registro[k] = v.item()
                elif _safe_isna(v):
                    registro[k] = None

            registros_para_ia.append({
                "idx": f"G{group_id}",
                "registro": registro
            })

        payload = {
            "Tareas": [{
                "columna": col_out,
                "criterios": criterios,
                "registro_cols": reg_cols
            }],
            "Registros": registros_para_ia
        }

        # Llamar IA con reintentos
        respuesta = None
        for _ in range(max_retries + 1):
            try:
                respuesta = fn_ia(payload, fecha_hoy=fecha_hoy)
                if respuesta and isinstance(respuesta, dict) and "Resultados" in respuesta:
                    break
            except Exception:
                pass

        if respuesta is None or "Resultados" not in respuesta:
            return []

        # Parsear respuesta y expandir a filas reales
        resultados = respuesta.get("Resultados", {})
        items = resultados.get(col_out, [])

        if not items:
            return []

        mapa_respuestas = {}
        for item in items:
            if not isinstance(item, dict):
                continue
            id_ia = item.get("id", item.get("idx"))
            val = item.get("resultado", item.get("etiqueta"))
            if id_ia is not None:
                mapa_respuestas[str(id_ia)] = val

        updates = []
        for group_id, key in enumerate(unique_keys):
            etiqueta = mapa_respuestas.get(f"G{group_id}")
            if etiqueta is None:
                etiqueta = mapa_respuestas.get(str(group_id))
            if etiqueta is None:
                continue
            for rid in clusters[key]:
                updates.append((rid, col_out, etiqueta))

        return updates

    # Ejecutar todas las tareas en paralelo
    all_updates = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_procesar_tarea, tarea, idx): (idx, tarea.get("nueva_columna"))
            for idx, tarea in enumerate(tareas)
        }

        for fut in concurrent.futures.as_completed(futures, timeout=batch_timeout * len(tareas)):
            try:
                updates = fut.result(timeout=batch_timeout)
                all_updates.extend(updates)
            except Exception:
                pass

    # Aplicar updates al DataFrame
    updates_by_col = defaultdict(dict)
    for rid, colname, etiqueta in all_updates:
        if rid in df_out.index and colname in df_out.columns:
            updates_by_col[colname][rid] = etiqueta

    for colname, updates_dict in updates_by_col.items():
        if updates_dict:
            updates_series = pd.Series(updates_dict)
            if replace_existing:
                df_out.loc[updates_series.index, colname] = updates_series.values
            else:
                current = df_out.loc[updates_series.index, colname]
                mask = current.isna() | (current == "")
                indices_to_update = mask[mask].index
                if len(indices_to_update) > 0:
                    df_out.loc[indices_to_update, colname] = updates_series[indices_to_update].values

    # Convertir columnas a tipo numérico si es posible
    for tarea in tareas:
        col = tarea.get("nueva_columna")
        if col and col in df_out.columns:
            df_out[col] = pd.to_numeric(df_out[col], errors='ignore')

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
    --font: -apple-system, Segoe UI, Roboto, Helvetica, Arial, "Noto Sans", Ubuntu;
    --text: #1e2329;
    --muted:#667085;
    --accent:#25347a;
    --accent-2:#58b12e;
    --border:#e6e8ec;
    --bg:#ffffff;
  }
  *{ box-sizing:border-box; }

  body{
    font-family: var(--font);
    color: var(--text);
    margin:0;
    background:#fff;
    line-height:1.65;
    font-size:15px;
  }

  main{
    max-width: var(--doc-max);
    margin: 24px auto;
    padding: 0 18px 32px;
  }

  h1,h2,h3{
    color: var(--accent);
    line-height:1.25;
  }
  h1{
    font-size:26px;
    margin-top:40px;
    border-bottom:2px solid var(--accent-2);
    padding-bottom:6px;
    page-break-before: always;
  }
  h2{ font-size:20px; margin-top:28px; }
  h3{ font-size:16px; margin-top:20px; }

  p{ margin:10px 0; text-align:justify; }

  section{ page-break-inside: avoid; }

  /* Figuras */
  figure{
    margin:16px auto 20px;
    text-align:center;
    page-break-inside: avoid;
  }
  figure img{
    max-width:100%;
    max-height:14cm;
    object-fit:contain;
  }
  figcaption{
    font-size:12px;
    color:var(--muted);
    margin-top:6px;
  }

  /* Bloques de análisis */
  .analysis{
    background:#f5f7fa;
    border-left:4px solid var(--accent);
    padding:12px 14px;
    margin:16px 0;
    text-align: justify;              /* ✅ fuerza justificado dentro del bloque */
    text-justify: inter-word;
  }

  /* Tabla de contenido */
  .toc ol{ padding-left:20px; }

  @media print{
    @page{ size:A4; margin:2cm; }
    main{ padding:0; }
  }
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
import unicodedata
import pandas as pd


# def columnas_a_texto(
#     df: pd.DataFrame,
#     col1: str,
#     col2: str,
#     *,
#     sep: str = "\n\n",
#     dropna: bool = True,
#     strip: bool = True
# ) -> str:
#     """
#     Toma dos columnas (col1 = título, col2 = contenido) y genera un HTML donde
#     cada fila se convierte en un apéndice:

#         <section class="toc Apendice N" id="apendice-N-slug-del-titulo">
#             <h1>TÍTULO</h1>
#             ...contenido...
#         </section>

#     - N empieza en 1.
#     - El id se construye a partir del título normalizado.
#     """

#     # --- Validaciones de columnas ---
#     for c in (col1, col2):
#         if c not in df.columns:
#             raise ValueError(f"La columna '{c}' no existe en el DataFrame.")

#     # --- Helper interno para generar slugs a partir del título ---
#     def _slugify(text: str) -> str:
#         if text is None:
#             return "apendice"
#         s = str(text)
#         # Normalizar tildes
#         s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
#         s = s.lower()
#         # Reemplazar todo lo que no sea alfanumérico por guiones
#         s = re.sub(r"[^a-z0-9]+", "-", s)
#         s = s.strip("-")
#         return s or "apendice"

#     bloques: list[str] = []
#     apendice_idx = 1

#     # --- Procesar fila por fila ---
#     for a, b in df[[col1, col2]].itertuples(index=False, name=None):
#         partes: list[str] = []
#         titulo_plano: str | None = None  # para usar en el id

#         for idx, v in enumerate((a, b)):
#             if v is None or (isinstance(v, float) and pd.isna(v)):
#                 if dropna:
#                     continue
#                 v = ""

#             s = str(v)
#             if strip:
#                 s = s.strip()

#             # col1 → Título (primer valor)
#             if idx == 0:
#                 titulo_plano = s
#                 s = f"<h1>{s}</h1>"

#             partes.append(s)

#         # Construir el bloque del apéndice solo si tiene contenido
#         if partes:
#             slug = _slugify(titulo_plano)
#             id_attr = f"apendice-{apendice_idx}-{slug}"

#             # Mantiene la numeración en la clase, añade 'toc' y el id
#             bloque = (
#                 f'<section class="Apendice {apendice_idx}" id="{id_attr}">'
#                 + sep.join(partes)
#                 + "</section>"
#             )
#             bloques.append(bloque)
#             apendice_idx += 1

#     return sep.join(bloques)

def columnas_a_texto(
    df: pd.DataFrame,
    col1: str,        # título
    col2: str,        # contenido
    col_level: str,   # nivel del título (1–4)
    *,
    sep: str = "\n\n",
    dropna: bool = True,
    strip: bool = True
) -> str:
    """
    Toma tres columnas:
      - col1: título
      - col2: contenido
      - col_level: nivel del título (1,2,3,4)

    y genera HTML con encabezados dinámicos (<h1>–<h4>).
    """

    # --- Validaciones de columnas ---
    for c in (col1, col2, col_level):
        if c not in df.columns:
            raise ValueError(f"La columna '{c}' no existe en el DataFrame.")

    # --- Helper interno para generar slugs ---
    def _slugify(text: str) -> str:
        if text is None:
            return "apendice"
        s = str(text)
        s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
        s = s.lower()
        s = re.sub(r"[^a-z0-9]+", "-", s)
        s = s.strip("-")
        return s or "apendice"

    bloques: list[str] = []
    apendice_idx = 1

    # --- Procesar fila por fila ---
    for titulo, contenido, level_raw in df[[col1, col2, col_level]].itertuples(index=False, name=None):
        partes: list[str] = []

        # --- Normalizar nivel ---
        try:
            level = int(level_raw)
        except Exception:
            level = 1

        if level not in (1, 2, 3, 4):
            level = 1

        h_tag = f"h{level}"

        # --- Título ---
        if titulo is None or (isinstance(titulo, float) and pd.isna(titulo)):
            if dropna:
                continue
            titulo_str = ""
        else:
            titulo_str = str(titulo).strip() if strip else str(titulo)

        slug = _slugify(titulo_str)
        id_attr = f"apendice-{apendice_idx}-{slug}"

        partes.append(
            f'<{h_tag} class="titulo-apendice title-level-{level}">'
            f'{titulo_str}'
            f'</{h_tag}>'
        )

        # --- Contenido ---
        if contenido is not None and not (isinstance(contenido, float) and pd.isna(contenido)):
            cont_str = str(contenido).strip() if strip else str(contenido)
            partes.append(cont_str)
        elif not dropna:
            partes.append("")

        # --- Construir bloque ---
        bloque = (
            f'<section class="Apendice {apendice_idx}" id="{id_attr}">'
            + sep.join(partes)
            + "</section>"
        )

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
        texto = re.sub(r"\|{2,}", "|", texto)
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

def _actualizar_indice_apendice_en_tag(open_tag: str, nuevo_idx: int) -> str:
    """
    Dado un <section ... class="... Apendice 3 ...">,
    reemplaza solo el '3' por nuevo_idx, preservando id y demás atributos.
    """
    def _repl(m: re.Match) -> str:
        # m.group(1) = 'class="... Apendice '
        # m.group(2) = número actual
        # m.group(3) = resto hasta el cierre de comillas
        return f'{m.group(1)}{nuevo_idx}{m.group(3)}'

    return re.sub(
        r'(class=["\'][^"\']*Apendice\s+)(\d+)([^"\']*["\'])',
        _repl,
        open_tag,
        flags=re.I
    )


def filtrar_apendices(texto: str, agente_result: dict, *, renumerar: bool = True) -> str:
    """
    Filtra bloques <section ... class="Apendice N" ...>...</section> de acuerdo con
    la salida del agente de clasificación.

    agente_result:
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
    - Se preservan id="..." y demás atributos del <section>.
    """

    if not isinstance(agente_result, dict):
        raise ValueError("agente_result debe ser un diccionario con 'conservar' y 'borrar'.")

    conservar = agente_result.get("conservar", [])
    borrar    = agente_result.get("borrar", [])

    # Normalización a enteros
    conservar = set(int(x) for x in conservar if isinstance(x, int) or str(x).isdigit())
    borrar    = set(int(x) for x in borrar    if isinstance(x, int) or str(x).isdigit())

    # Si un índice aparece en ambos → se borra
    borrar = borrar.union(conservar.intersection(borrar))

    # Capturamos:
    # 1) el tag de apertura completo (<section ...>)
    # 2) el índice de Apendice (\d+)
    # 3) el contenido interno (.*?)
    patron = re.compile(
        r'(<section[^>]*class=["\'][^"\']*Apendice\s+(\d+)[^"\']*["\'][^>]*>)(.*?)</section\s*>',
        re.S | re.I
    )

    matches = list(patron.finditer(texto))
    if not matches:
        return texto  # no hay apéndices → nada que hacer

    nuevos_bloques = []

    for m in matches:
        open_tag   = m.group(1)  # <section ...>
        idx_str    = m.group(2)  # número dentro de 'Apendice N'
        contenido  = m.group(3)  # HTML interno

        idx = int(idx_str)

        # 1) Prioridad: si está en BORRAR → se elimina
        if idx in borrar:
            continue

        # 2) Si 'conservar' está vacío → conservar todo lo que no esté en borrar
        if not conservar:
            nuevos_bloques.append((idx, open_tag, contenido))
            continue

        # 3) Si 'conservar' NO está vacío → conservar solo esos índices
        if idx in conservar:
            nuevos_bloques.append((idx, open_tag, contenido))

    if not nuevos_bloques:
        return ""

    # Reconstrucción
    salida = []
    if renumerar:
        for nuevo_idx, (_, open_tag, contenido) in enumerate(nuevos_bloques, start=1):
            # Cambiamos solo el número de 'Apendice N' en la clase, pero
            # mantenemos id="..." y demás atributos igual.
            new_open_tag = _actualizar_indice_apendice_en_tag(open_tag, nuevo_idx)
            salida.append(f'{new_open_tag}{contenido}</section>')
    else:
        # Mantenemos el tag original tal cual, incluyendo id e índice de apéndice
        for idx, open_tag, contenido in nuevos_bloques:
            salida.append(f'{open_tag}{contenido}</section>')

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
    Extrae todos los títulos HTML <h1> a <h4>.

    Devuelve:
    [
        {
            "idx": 1,
            "titulo": "Introducción",
            "level": 1,
            "tag": "h1",
            "span": (start, end)
        },
        ...
    ]
    """

    patron = re.compile(
        r'<(h[1-4])\b[^>]*>(.*?)</\1>',
        re.I | re.S
    )

    bloques = []

    for i, match in enumerate(patron.finditer(texto), start=1):
        tag = match.group(1).lower()      # h1, h2, h3, h4
        level = int(tag[1])               # 1,2,3,4
        titulo = match.group(2).strip()

        bloques.append({
            "idx": i,
            "titulo": titulo,
            "level": level,
            "tag": tag,
            "span": match.span()
        })

    return bloques


def aplicar_titulos_numerados(texto: str, titulos_numerados) -> str:
    """
    Reemplaza títulos <h1>-<h4> por versiones numeradas,
    respetando el nivel original.
    """

    reemplazos = []

    for item in titulos_numerados:
        if not isinstance(item, dict):
            continue

        nuevo_titulo = item.get("titulo")
        level = item.get("level")
        span_obj = item.get("span")

        if not nuevo_titulo or not isinstance(level, int):
            continue
        if level not in (1, 2, 3, 4):
            continue

        # Normalizar span
        if isinstance(span_obj, dict):
            span_list = span_obj.get("span") or span_obj.get("spam")
        else:
            span_list = span_obj

        if not (isinstance(span_list, (list, tuple)) and len(span_list) == 2):
            continue

        start, end = span_list
        if not (0 <= start <= end <= len(texto)):
            continue

        tag = f"h{level}"

        reemplazo = (
            f'<{tag} class="titulo-apendice title-level-{level}">'
            f'{nuevo_titulo}'
            f'</{tag}>'
        )

        reemplazos.append((start, end, reemplazo))

    # ⚠️ Muy importante: derecha a izquierda
    reemplazos.sort(key=lambda x: x[0], reverse=True)

    out = texto
    for start, end, rep in reemplazos:
        out = out[:start] + rep + out[end:]

    return out

from html import escape

def generar_tabla_contenido(titulos_numerados) -> str:
    """
    Construye un bloque HTML con la tabla de contenido.

    Parámetro:
        titulos_numerados: lista de dicts con estructura:
        {
            "idx": N,
            "titulo": "1.2 Perfil hematológico",  # título ya numerado
            "span": [inicio, fin]
        }

    Retorno:
        HTML con la estructura:

        <section class="toc">
          <h1>Tabla de contenido</h1>
          <ol>
            <li>...</li>
            ...
          </ol>
        </section>
    """

    if not isinstance(titulos_numerados, list):
        raise TypeError("titulos_numerados debe ser una lista de dicts.")

    # Extraer y limpiar títulos válidos
    titulos_limpios = []
    for item in titulos_numerados:
        if not isinstance(item, dict):
            continue

        titulo = item.get("titulo")
        if isinstance(titulo, str) and titulo.strip():
            titulos_limpios.append(titulo.strip())

    # Si no hay títulos, opcional: devolver cadena vacía
    if not titulos_limpios:
        return ""

    # Construir HTML
    partes = [
        '<section class="toc">',
        '  <h1>Tabla de contenido</h1>',
        '  <ol>',
    ]

    for titulo in titulos_limpios:
        partes.append(f"    <li>{escape(titulo)}</li>")

    partes.extend([
        '  </ol>',
        '</section>',
    ])

    return "\n".join(partes)


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


import re

def remover_contenedores_apendice(html: str) -> str:
    """
    Elimina únicamente las etiquetas <div class="Apendice N"> y </div> 
    sin afectar el contenido, incluso cuando las etiquetas están
    mal anidadas dentro de <p>.
    """

    # 1. Eliminar aperturas de div tipo: <div class="Apendice 7"> o <div class='Apendice 3'>
    html = re.sub(
        r'<div\s+class=["\']Apendice\s+\d+["\'][^>]*>',
        '',
        html,
        flags=re.IGNORECASE
    )

    # 2. Eliminar cualquier cierre </div> (solo esos div se usan en tu informe)
    html = re.sub(
        r'</div\s*>',
        '',
        html,
        flags=re.IGNORECASE
    )

    return html


import pandas as pd
from typing import Dict, List, Tuple, Optional


import unicodedata
import pandas as pd

def _normalizar_texto_base(valor: object) -> str:
    if valor is None or (isinstance(valor, float) and pd.isna(valor)):
        return ""
    texto = str(valor).strip()
    if not texto:
        return ""
    # quitar tildes / diacríticos
    texto = unicodedata.normalize("NFKD", texto)
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    # Normal -> "Normal"
    texto = texto.lower()
    return texto.capitalize()

def normalizar_columna(
    df: pd.DataFrame,
    col: str,
    *,
    output_col: str | None = None,
    copy: bool = True
) -> pd.DataFrame:
    """
    Retorna un DataFrame con la columna normalizada.
    - Si output_col es None: reemplaza df[col]
    - Si output_col tiene valor: crea df[output_col]
    - copy=True: no muta el df original
    """
    if col not in df.columns:
        raise ValueError(f"La columna '{col}' no existe en el DataFrame. Columnas: {list(df.columns)}")

    out = df.copy() if copy else df
    target_col = output_col or col
    out[target_col] = out[col].apply(_normalizar_texto_base)
    return out


def reemplazar_textos(
    df: pd.DataFrame,
    columna: str = "Resultado",
    reglas: Optional[Dict[str, str]] = None,
    *,
    rellenar_vacios_con: Optional[str] = None,  # <-- NUEVO
    considerar_vacio_str: bool = True,          # "" o "   " cuentan como vacío
    ignorar_mayusculas: bool = True,
    solo_valor_completo: bool = True,
) -> pd.DataFrame:
    if columna not in df.columns:
        raise ValueError(f"La columna '{columna}' no existe en el DataFrame.")

    out = df.copy()
    reglas = reglas or {}

    # 1) Rellenar vacíos (None/NaN y opcionalmente "" / "   ")
    if rellenar_vacios_con is not None:
        s = out[columna]

        # NaN / None
        s = s.fillna(rellenar_vacios_con)

        # "" o solo espacios
        if considerar_vacio_str:
            s_str = s.astype("string")
            mask_vacios = s_str.str.strip().eq("")  # True para "" o "   "
            s = s_str.mask(mask_vacios, rellenar_vacios_con)

        out[columna] = s

    # 2) Reglas de reemplazo
    if reglas:
        if solo_valor_completo:
            if ignorar_mayusculas:
                mapa_norm = {str(k).strip().lower(): v for k, v in reglas.items()}

                def _map_cell(x):
                    if pd.isna(x):
                        return x
                    key = str(x).strip().lower()
                    return mapa_norm.get(key, x)

                out[columna] = out[columna].map(_map_cell)
            else:
                out[columna] = out[columna].replace(reglas)
        else:
            s = out[columna].astype("string")
            for src, dst in reglas.items():
                s = s.str.replace(str(src), str(dst), case=not ignorar_mayusculas, regex=False)
            out[columna] = s

    return out

import pandas as pd
from typing import Dict, List


def crear_resultado_agregado(
    df: pd.DataFrame,
    col_documento: str = "Documento",
    col_tipo_prueba: str = "Tipo prueba",
    col_resultado: str = "Resultado",
    nueva_columna: str = "Resultado agregado",
    reglas_por_tipo: Dict[str, Dict[str, List[str]]] = None,
    valor_sin_dato: str = "Sin dato",
    valor_sin_regla: str = "Sin clasificar",
) -> pd.DataFrame:
    """
    Crea un resultado agregado por (Documento, Tipo prueba)
    usando reglas específicas por tipo de prueba.
    """

    if reglas_por_tipo is None:
        raise ValueError("Debe proporcionar reglas_por_tipo")

    out = df.copy()

    def evaluar_grupo(valores: pd.Series) -> str:
        # valores es la serie de Resultados del grupo
        resultados = valores.dropna().astype(str)

        if resultados.empty:
            return valor_sin_dato

        # Obtenemos el tipo de prueba desde el índice del grupo
        tipo = valores.name[1]  # (documento, tipo_prueba)

        reglas_tipo = reglas_por_tipo.get(tipo)
        if not reglas_tipo:
            return valor_sin_regla

        for resultado_final, disparadores in reglas_tipo.items():
            if resultados.isin(disparadores).any():
                return resultado_final

        return valor_sin_regla

    out[nueva_columna] = (
        out
        .groupby([col_documento, col_tipo_prueba])[col_resultado]
        .transform(evaluar_grupo)
    )

    return out




def eliminar_duplicados_ultimo(
    df: pd.DataFrame,
    col_unica: Union[str, List[str]]
) -> pd.DataFrame:
  
    if df is None or df.empty:
        return df

    if isinstance(col_unica, str):
        col_unica = [col_unica]

    # Mantiene el último registro por cada duplicado, respetando el orden actual del DataFrame
    df_final = df.drop_duplicates(subset=col_unica, keep="last").reset_index(drop=True)

    return df_final
