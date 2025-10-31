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
from Colmedicos.math_ops import ejecutar_operaciones_condicionales
from Colmedicos.ia import operaciones_gpt5, graficos_gpt5
from Colmedicos.charts import plot_from_params



_VAR_REGEX = re.compile(r"\{\{\s*([A-Za-z_][\w\.]*)\s*\}\}")

@register("_render_vars")
def _render_vars(texto: str, ctx: Optional[Dict[str, Any]], strict: bool) -> str:

    if not isinstance(texto, str) or not ctx:
        return "" if texto is None else str(texto)

    def repl(m: re.Match) -> str:
        key = m.group(1)
        if key in ctx:
            val = ctx[key]
            return "" if val is None else str(val)
        if strict:
            raise KeyError(f"Variable no definida en ctx: '{key}'")
        return ""  # modo tolerante

    return _VAR_REGEX.sub(repl, texto)

@register("generar_output")
def generar_output(
    df: pd.DataFrame,
    col_texto: str = "Titulo",          # usa "Texto" si tu DF la tiene así
    col_contenido: str = "Contenido",
    plantilla: str = "{texto}\n\n{contenido}",
    crear_columna: bool = True,
    unir_todos: bool = False,
    separador_unido: str = "\n\n---\n\n",
    ctx: Optional[Dict[str, Any]] = None,  # ← variables para {{ ... }}
    strict_ctx: bool = False               # ← si True, error si falta una variable
):
    """
    Combina por fila 'col_texto' + 'col_contenido' en un Output, con soporte de variables {{ var }}.

    - Procesa TODAS las filas (ya no depende de 'Tipo').
    - Reemplaza variables {{ nombre_variable }} en 'col_texto' y 'col_contenido' usando 'ctx'.
    - 'plantilla' debe usar {texto} y {contenido} para componer el resultado final.

    Retorna:
      - df_out (con columna 'Output' si crear_columna=True)
      - y opcionalmente (si unir_todos=True) un único string unido con 'separador_unido'.
    """

    # Validaciones de columnas
    faltantes = [c for c in [col_texto, col_contenido] if c not in df.columns]
    if faltantes:
        raise ValueError(f"Faltan columnas en el DataFrame: {faltantes}")

    df_out = df.copy()

    def _combinar(texto_raw, contenido_raw):
        # Resolver variables en ambos campos
        t0 = "" if pd.isna(texto_raw) else str(texto_raw)
        c0 = "" if pd.isna(contenido_raw) else str(contenido_raw)

        t = _render_vars(t0, ctx, strict_ctx)
        c = _render_vars(c0, ctx, strict_ctx)

        # Armar con la plantilla
        return plantilla.format(texto=t.strip(), contenido=c.strip()).strip()

    outputs = [
        _combinar(tx, ct) for tx, ct in zip(df_out[col_texto], df_out[col_contenido])
    ]

    if crear_columna:
        df_out["Output"] = outputs

    if unir_todos:
        lista = [o for o in outputs if o and o.strip()]
        unidos = separador_unido.join(lista) if lista else ""
        return df_out, unidos

    return df_out

@register("aplicar_ia_por_tipo")
def aplicar_ia_por_tipo(
    df: pd.DataFrame,
    ask_fn: Callable[[str], str],           # p.ej. ask_gpt5
    col_tipo: str = "Tipo",
    col_output: str = "Output",
    valor_tipo_objetivo: str = "Fijo con IA",
    overwrite: bool = True,                 # True = sobrescribe Output; False = crea col_salida
    col_salida: str = "Output_IA",          # Sólo se usa si overwrite=False
    on_error: Literal["keep", "mark"] = "keep"  # "keep" conserva texto original si falla IA; "mark" escribe [ERROR IA] ...
) -> pd.DataFrame:
    """
    Para filas donde Tipo == 'Fijo con IA', envía df[Output] a ask_fn y reemplaza por la respuesta.
    En las demás filas, mantiene el texto original.

    - df: DataFrame que contiene al menos las columnas 'Tipo' y 'Output'
    - ask_fn: función IA que recibe prompt (str) y devuelve respuesta (str)
    - on_error: 'keep' conserva el texto original si hay error; 'mark' escribe un mensaje de error
    """
    # Validaciones mínimas
    faltan = [c for c in [col_tipo, col_output] if c not in df.columns]
    if faltan:
        raise ValueError(f"Faltan columnas en el DataFrame: {faltan}")

    # Trabajamos sobre copia
    df2 = df.copy()

    # Determinar dónde escribir
    destino = col_output if overwrite else col_salida
    if not overwrite and destino in df2.columns:
        # evitar sobreescritura accidental
        raise ValueError(f"La columna destino '{destino}' ya existe. Elige otro nombre o usa overwrite=True.")

    # Normalización de comparación (tolerante a espacios y mayúsculas)
    tipo_norm = df2[col_tipo].astype(str).str.strip().str.casefold()
    objetivo_norm = str(valor_tipo_objetivo).strip().casefold()

    # Procesar fila a fila
    nuevos = []
    for idx, (tipo_val, texto) in enumerate(zip(tipo_norm, df2[col_output].astype(str))):
        if tipo_val == objetivo_norm:
            try:
                resp = ask_fn(texto.strip())
                nuevos.append((resp or "").strip())
            except Exception as e:
                if on_error == "keep":
                    nuevos.append(texto)
                else:
                    nuevos.append(f"[ERROR IA] {e}")
        else:
            nuevos.append(texto)

    # Escribir resultados
    df2[destino] = nuevos
    return df2

@register("exportar_ouput_a_html")
def exportar_output_a_html(
    df: pd.DataFrame,
    col_output: str = "Output",
    archivo_html: str = "salida.html",
    titulo: str = "Documento",
    escapar_html: bool = True,
    separar_por_dobles_saltos: bool = True,
    unir_con: str = "\n\n"
) -> str:
    # Import local con alias para evitar shadowing
    import html as _html

    if col_output not in df.columns:
        raise ValueError(f"No existe la columna '{col_output}' en el DataFrame.")

    bloques = df[col_output].fillna("").astype(str).tolist()
    texto = unir_con.join([b for b in bloques if b.strip() != ""]).strip()

    if separar_por_dobles_saltos:
        parrafos = [p.strip() for p in texto.split("\n\n") if p.strip()]
        def a_html_parrafo(p):
            p_html = (_html.escape(p).replace("\n", "<br>")
                      if escapar_html else p.replace("\n", "<br>"))
            return f"<p>{p_html}</p>"
        cuerpo = "\n".join(a_html_parrafo(p) for p in parrafos) if parrafos else "<p></p>"
    else:
        cuerpo_texto = _html.escape(texto) if escapar_html else texto
        cuerpo = f"<p>{cuerpo_texto.replace('\n', '<br>')}</p>"

    html_doc = f"""<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{_html.escape(titulo)}</title>
  <style>
    body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin: 24px; line-height: 1.6; color: #222; }}
    h1 {{ font-size: 1.6rem; margin-bottom: 1rem; }}
    p {{ margin: 0 0 1rem 0; }}
    .container {{ max-width: 980px; margin: 0 auto; }}
  </style>
</head>
<body>
  <div class="container">
    <h1>{_html.escape(titulo)}</h1>
    {cuerpo}
  </div>
</body>
</html>"""

    Path(archivo_html).write_text(html_doc, encoding="utf-8")
    return html_doc

@register("mostrar_html")
def mostrar_html(html_content):
    """
    Crea un archivo temporal con contenido HTML y lo abre automáticamente en el navegador.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".html", mode="w", encoding="utf-8") as f:
        f.write(html_content)
        temp_path = f.name

    # Abrir el archivo en el navegador predeterminado
    webbrowser.open(f"file://{temp_path}")


@register("fig_to_data_uri")
def fig_to_data_uri(obj):
    import io as _io
    import base64 as _b64

    fig = obj.figure if hasattr(obj, "figure") else obj
    if fig is None:
        raise ValueError("Se recibió None en lugar de Figure/Axes.")
    buf = _io.BytesIO()
    # Fondo blanco para que no quede transparente/blanco “vacío”
    fig.savefig(buf, format="png", bbox_inches="tight",
                facecolor="white", edgecolor="white")
    b64 = _b64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"

@register("_render_plot_and_get_base64")
def _render_plot_and_get_base64(plot_callable) -> str:
    before_ids = {m.num for m in Gcf.get_all_fig_managers()}
    plot_callable()
    after_managers = Gcf.get_all_fig_managers()
    new_managers = [m for m in after_managers if m.num not in before_ids]
    fig = (new_managers[-1].canvas.figure if new_managers else plt.gcf())
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("ascii")
    plt.close(fig)
    return b64

@register("aplicar_plot_por_tipo_desde_output")
def aplicar_plot_por_tipo_desde_output(
    df: pd.DataFrame,
    df_datos: pd.DataFrame,
    col_tipo: str = "Tipo",
    col_output: str = "Output",
    valor_tipo_objetivo: str = "Fijo con IA",
    verbose: bool = True,
    reemplazar_en_html: bool = True,  # <img src="data:image/png;base64,..."/>
    token_reemplazo: str = "#GRAFICA#",
    replace_all: bool = True,         # True: reemplaza todas las ocurrencias; False: solo la primera
    inplace: bool = True              # True: muta df; False: trabaja sobre copia y la retorna
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """
    Filtra filas con Tipo == valor_tipo_objetivo, usa df[col_output] como prompt,
    obtiene params con graficos_gpt5, genera la gráfica, captura PNG->Base64 y
    reemplaza 'token_reemplazo' en el texto del Output por <img ...> o base64.
    Devuelve: (df_actualizado, resultados)
    """
    # Validaciones mínimas
    faltan = [c for c in (col_tipo, col_output) if c not in df.columns]
    if faltan:
        raise ValueError(f"Faltan columnas en df: {faltan}")

    # Trabajar sobre copia si no es inplace
    df_out = df if inplace else df.copy()

    resultados: List[Dict[str, Any]] = []
    tipo_norm = df_out[col_tipo].astype(str).str.strip().str.casefold()
    objetivo_norm = valor_tipo_objetivo.strip().casefold()

    for ridx, (tval, prompt_raw) in zip(df_out.index, zip(tipo_norm, df_out[col_output])):
        if tval != objetivo_norm:
            continue

        prompt = "" if pd.isna(prompt_raw) else str(prompt_raw).strip()
        if not prompt:
            if verbose:
                print(f"[aplicar_plot_por_tipo_desde_output] Fila {ridx}: prompt vacío en '{col_output}', se omite.")
            resultados.append({"index": ridx, "ok": False, "error": "prompt vacío", "params": None, "prompt": prompt, "base64_len": None})
            continue

        try:
            # 1) Obtener parámetros (IA)
            params = graficos_gpt5(df_datos, prompt)

            # 2) Normalizaciones mínimas
            if isinstance(params.get("y"), list) and params.get("function_name") == "graficar_torta":
                params["y"] = params["y"][0] if params["y"] else None
            if not params.get("agg"):
                params["agg"] = "sum"

            # 3) Generar y capturar la gráfica en Base64
            fig, ax = plot_from_params(df_datos, params)   # <- obtiene Figure y Axes
            b64 = fig_to_data_uri(fig)                      # <- pasa la Figure (o (fig, ax))
            plt.close(fig)                                  # <- cierra la Figure

            data_uri = f"{b64}"
            reemplazo = f'<img src="{data_uri}" alt="Grafico" />' if reemplazar_en_html else data_uri
            # 4) Reemplazo en el texto
            texto_original = df_out.loc[ridx, col_output]
            if replace_all:
                nuevo_texto = (texto_original or "").replace(token_reemplazo, reemplazo)
            else:
                nuevo_texto = (texto_original or "").replace(token_reemplazo, reemplazo, 1)

            df_out.loc[ridx, col_output] = nuevo_texto

            resultados.append({
                "index": ridx,
                "ok": True,
                "error": None,
                "params": params,
                "prompt": prompt,
                "base64_len": len(b64)
            })

        except Exception as e:
            if verbose:
                print(f"[aplicar_plot_por_tipo_desde_output] Fila {ridx}: error -> {e}")
            resultados.append({
                "index": ridx,
                "ok": False,
                "error": str(e),
                "params": None,
                "prompt": prompt,
                "base64_len": None
            })

    if verbose and not resultados:
        print("[aplicar_plot_por_tipo_desde_output] No se encontraron filas con el tipo objetivo.")

    # ← Devolvemos el DataFrame completo y el log de resultados
    return df_out, resultados

import pandas as pd
import numpy as np
from typing import Any, Dict, List, Tuple, Callable, Union

# ----------------------------
# Helpers de formateo a texto
# ----------------------------
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
    """
    Convierte el resultado devuelto por ejecutar_operaciones_condicionales
    (DataFrame | dict | escalar) a texto plano.
    """
    if isinstance(result, pd.DataFrame):
        df_show = result.copy()
        # Formatear numéricos
        for c in df_show.columns:
            if pd.api.types.is_numeric_dtype(df_show[c]):
                df_show[c] = df_show[c].apply(lambda v: _format_number(v, float_fmt))
            elif pd.api.types.is_datetime64_any_dtype(df_show[c]):
                df_show[c] = df_show[c].dt.strftime("%Y-%m-%d")
            else:
                df_show[c] = df_show[c].astype(str).fillna("")
        # Limitar filas si es grande
        if len(df_show) > max_rows:
            df_show = df_show.head(max_rows)
        return df_show.to_string(index=include_index)

    if isinstance(result, dict):
        # key: value por línea
        lines = []
        for k, v in result.items():
            if isinstance(v, (int, float, np.integer, np.floating)):
                v_str = _format_number(v, float_fmt)
            else:
                v_str = str(v)
            lines.append(f"{k}: {v_str}")
        return "\n".join(lines) if lines else ""

    # Escalar (número, string, etc.)
    return _format_number(result, float_fmt)

# ----------------------------------------------------------------
# Función principal: reemplaza ||DATOS|| por el texto plano generado
# ----------------------------------------------------------------
@register("aplicar_data_por_tipo_desde_output")
def aplicar_data_por_tipo_desde_output(
    df: pd.DataFrame,
    df_datos: pd.DataFrame,
    col_tipo: str = "Tipo",
    col_output: str = "Output",
    valor_tipo_objetivo: str = "Fijo con IA",
    verbose: bool = True,
    token_reemplazo: str = "||DATOS||",
    replace_all: bool = True,
    inplace: bool = True,
    # NUEVOS parámetros:
    # parse_ops_fn: Callable[[pd.DataFrame, str], Dict[str, Any]] = None,  # función que parsea el prompt → spec JSON
    # ejecutar_fn: Callable[[pd.DataFrame, Dict[str, Any]], Union[pd.DataFrame, Dict[str, Any], Any]] = None,
    float_fmt: str = "{:,.2f}",
    max_rows: int = 200,
    include_index: bool = False,
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """
    Filtra filas con Tipo == valor_tipo_objetivo, usa df[col_output] como "prompt",
    obtiene la especificación de operaciones (parse_ops_fn), ejecuta con ejecutar_fn
    (p.ej. ejecutar_operaciones_condicionales) y reemplaza 'token_reemplazo' en el
    texto del Output por el resultado en TEXTO PLANO.

    Retorna: (df_actualizado, resultados_log)
    """
    # Validaciones mínimas
    for c in (col_tipo, col_output):
        if c not in df.columns:
            raise ValueError(f"Falta la columna '{c}' en df.")

    # if parse_ops_fn is None:
    #     raise ValueError("Debes proporcionar 'parse_ops_fn' que lea el prompt y devuelva la especificación JSON.")
    # if ejecutar_fn is None:
    #     raise ValueError("Debes proporcionar 'ejecutar_fn' que ejecute las operaciones y devuelva DataFrame o dict.")

    # Trabajar sobre copia si no es inplace
    df_out = df if inplace else df.copy()

    resultados: List[Dict[str, Any]] = []
    tipo_norm = df_out[col_tipo].astype(str).str.strip().str.casefold()
    objetivo_norm = valor_tipo_objetivo.strip().casefold()

    for ridx, (tval, prompt_raw) in zip(df_out.index, zip(tipo_norm, df_out[col_output])):
        if tval != objetivo_norm:
            continue

        prompt = "" if pd.isna(prompt_raw) else str(prompt_raw).strip()
        if not prompt:
            if verbose:
                print(f"[aplicar_data_por_tipo_desde_output] Fila {ridx}: prompt vacío en '{col_output}', se omite.")
            resultados.append({"index": ridx, "ok": False, "error": "prompt vacío", "spec": None, "prompt": prompt})
            continue

        try:
            # 1) Parsear la instrucción a SPEC (nuevo subprompt multi-op)
            spec = operaciones_gpt5(df_datos, prompt)

            # 2) Ejecutar operaciones
            result = ejecutar_operaciones_condicionales(df_datos, spec)

            # 3) Formatear a TEXTO PLANO
            texto_datos = _format_result_plain(
                result, float_fmt=float_fmt, max_rows=max_rows, include_index=include_index
            )

            # 4) Reemplazo en el texto original
            texto_original = df_out.loc[ridx, col_output]
            if replace_all:
                nuevo_texto = (texto_original or "").replace(token_reemplazo, texto_datos)
            else:
                nuevo_texto = (texto_original or "").replace(token_reemplazo, texto_datos, 1)

            df_out.loc[ridx, col_output] = nuevo_texto

            resultados.append({
                "index": ridx,
                "ok": True,
                "error": None,
                "spec": spec,
                "prompt": prompt,
                "result_type": type(result).__name__,
                "texto_len": len(texto_datos),
            })

        except Exception as e:
            if verbose:
                print(f"[aplicar_data_por_tipo_desde_output] Fila {ridx}: error -> {e}")
            resultados.append({
                "index": ridx,
                "ok": False,
                "error": str(e),
                "spec": None,
                "prompt": prompt,
            })

    if verbose and not resultados:
        print("[aplicar_data_por_tipo_desde_output] No se encontraron filas con el tipo objetivo.")

    return df_out

import re
import pandas as pd
from typing import Optional, Iterable, Tuple, Dict, List

# --- Tu limpiador base (ajustable) ---
@register("limpiar_texto")
def limpiar_texto(
    texto: str,
    chars_a_quitar: Optional[Iterable[str]] = None,
    quitar_entre_hash: bool = True,
    reemplazo_entre_hash: str = "",
    entre_hash_multilinea: bool = False,
    normalizar_espacios: bool = True
) -> str:
    if texto is None:
        return texto

    # 1) Eliminar / reemplazar bloques entre #...#
    if quitar_entre_hash:
        flags = re.DOTALL if entre_hash_multilinea else 0
        texto = re.sub(r'#.*?#', reemplazo_entre_hash, texto, flags=flags)

    # 2) Quitar caracteres especiales dados
    if chars_a_quitar is None:
        chars_a_quitar = ['*', '+', '[', ']']

    tabla = str.maketrans({c: "" for c in chars_a_quitar})
    texto = texto.translate(tabla)

    # 3) Normalizar espacios (opcional)
    if normalizar_espacios:
        texto = re.sub(r'\s+', ' ', texto).strip()

    return texto


# --- Protección de data URIs (para no romper Base64) ---
_DATA_URI_RE = re.compile(r'data:image/[a-zA-Z]+;base64,[A-Za-z0-9+/=]+')

@register("_proteger_data_uris")
def _proteger_data_uris(texto: str) -> Tuple[str, Dict[str, str]]:
    """
    Reemplaza cada data URI por un token temporal __DATAURI_i__.
    Devuelve (texto_protegido, mapa_tokens->valor_original).
    """
    if not texto or not isinstance(texto, str):
        return texto, {}

    mapping: Dict[str, str] = {}
    i = 0

    def _sub(m):
        nonlocal i, mapping
        token = f"__DATAURI_{i}__"
        mapping[token] = m.group(0)
        i += 1
        return token

    protegido = _DATA_URI_RE.sub(_sub, texto)
    return protegido, mapping

@register("_restaurar_data_uris")
def _restaurar_data_uris(texto: str, mapping: Dict[str, str]) -> str:
    if not mapping:
        return texto
    for token, val in mapping.items():
        texto = texto.replace(token, val)
    return texto

@register("limpiar_output_dataframe")
# --- Limpieza sobre la columna Output del DataFrame ---
def limpiar_output_dataframe(
    df: pd.DataFrame,
    col_output: str = "Output",
    *,
    chars_a_quitar: Optional[Iterable[str]] = None,
    quitar_entre_hash: bool = True,
    reemplazo_entre_hash: str = "",
    entre_hash_multilinea: bool = False,
    normalizar_espacios: bool = True,
    inplace: bool = True
) -> pd.DataFrame:
    """
    Aplica la limpieza sobre df[col_output] protegiendo data URIs (Base64) para no dañarlas.
    - Si inplace=True, muta df y lo devuelve.
    - Si inplace=False, trabaja sobre copia y devuelve la copia.
    """
    if col_output not in df.columns:
        raise ValueError(f"No existe la columna '{col_output}' en el DataFrame.")

    target = df if inplace else df.copy()

    def _limpiar_celda(x):
        if not isinstance(x, str) or x.strip() == "":
            return x
        # 1) Proteger data URIs
        protegido, mapa = _proteger_data_uris(x)
        # 2) Limpiar sobre el texto protegido (no toca Base64)
        limpio = limpiar_texto(
            protegido,
            chars_a_quitar=chars_a_quitar,
            quitar_entre_hash=quitar_entre_hash,
            reemplazo_entre_hash=reemplazo_entre_hash,
            entre_hash_multilinea=entre_hash_multilinea,
            normalizar_espacios=normalizar_espacios
        )
        # 3) Restaurar data URIs
        return _restaurar_data_uris(limpio, mapa)

    target[col_output] = target[col_output].map(_limpiar_celda)
    return target
