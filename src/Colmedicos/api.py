
from Colmedicos.ia import ask_gpt5, portada_gpt5
from Colmedicos.io_utils import generar_output, aplicar_data_por_tipo_desde_output, aplicar_ia_por_tipo, aplicar_plot_por_tipo_desde_output, exportar_output_a_html, mostrar_html, limpiar_output_dataframe
from Colmedicos.registry import register
from Colmedicos.io_utils_remaster import process_ia_blocks, process_data_blocks, process_plot_blocks, _render_vars_text, parse_plot_blocks, parse_ia_blocks, parse_data_blocks, exportar_output_a_html, _fig_to_data_uri, _format_result_plain, columnas_a_texto,aplicar_multiples_columnas_gpt5, limpieza_final,  unpivot_df, dividir_columna_en_dos, procesar_codigos_cie10, unir_dataframes, mejorar_html_informe, estructurar_documento, expand_json_column
import pandas as pd

# Colmedicos/api.py
import time
import pandas as pd
from typing import Any, Dict, List, Tuple, Union, Optional, Callable

def informe_final(
    df: pd.DataFrame,
    df_datos: pd.DataFrame,
    ctx: dict,
    tareas: List[Dict[str, Any]] = [],
    salida_html: str = r"C:\Users\EstebanEscuderoPuert\Downloads\informe_final.html",
    escribir_archivo: bool = True,
    modo_rapido_plots: bool = True,
    generar_portada: bool = True,
    aplicar_cie10: bool = False,
    aplicar_union: bool = False,
    aplicar_unpivot: bool = False,
    aplicar_split: bool = False,
    aplicar_expansion_json: bool = False,
    # parámetros CIE10
    col_texto_cie10: str = None,
    col_df1: str = None,
    col_df2: str = "Code",
    # parámetros unpivot
    columnas_unpivot: list = None,
    col_split: str = None,
    sep_split: str = None,
    nombre_col1: str = None,
    nombre_col2: str = "Resultado",
    eliminar_original_split: bool = True,
    # Parámetros de expansión JSON
    json_columna: str = None,
    campos_a_extraer: list = None,
    renombrar_campos: dict = None
) -> Tuple[str, Dict[str, Any]]:
    """
    Versión optimizada con:
      - early-exit por tokens
      - micro-perfilado por etapa
      - opción para evitar E/S a disco
    """

    df_out = df_datos.copy()

    # Manipulacion columnas
    df_out = aplicar_multiples_columnas_gpt5(df_out, tareas)
    print("Columnas creadas con éxito")

    # Expansión columna JSON
    if aplicar_expansion_json:
        if not json_columna or not campos_a_extraer:
            raise ValueError("Debe especificar json_columna y campos_a_extraer cuando aplicar_expansion_json=True")
        df_out = expand_json_column(
            df_out,
            json_columna,
            fields_to_extract=campos_a_extraer,
            rename_map=renombrar_campos
        )
        print("✔ Expansión de columna JSON aplicada")

    # 1) PROCESAR CIE10 ----------------------------------
    import json
    with open("src\Colmedicos\cie10.json", "r", encoding="utf-8") as f:
        maestro_cie10 = json.load(f)
    df_union = pd.DataFrame(maestro_cie10)
    if aplicar_cie10:
        if not col_texto_cie10:
            raise ValueError("Debe especificar col_texto_cie10 cuando aplicar_cie10=True")
        df_out = procesar_codigos_cie10(df_out, columna_texto=col_texto_cie10)
        print("✔ Transformación CIE10 aplicada")

    # 2) UNION DE DATAFRAMES ------------------------------
    if aplicar_union:
        if df_union is None or not col_df1 or not col_df2:
            raise ValueError("Debe especificar df_union, col_df1 y col_df2 cuando aplicar_union=True")
        df_out = unir_dataframes(df_out, df_union, col_df1, col_df2)
        print("✔ Unión de DataFrames aplicada")

    # 3) UNPIVOT -------------------------------------------
    if aplicar_unpivot:
        if columnas_unpivot is None:
            raise ValueError("Debe especificar columnas_unpivot cuando aplicar_unpivot=True")
        df_out = unpivot_df(df_out, columnas_unpivot=columnas_unpivot)
        print("✔ Unpivot aplicado")

    # 4) DIVIDIR COLUMNA EN DOS ---------------------------
    if aplicar_split:
        if not col_split or not sep_split or not nombre_col1 or not nombre_col2:
            raise ValueError("Debe especificar parámetros del split cuando aplicar_split=True")

        df_out = dividir_columna_en_dos(
            df_out,
            columna=col_split,
            caracter_separador=sep_split,
            nombre_col1=nombre_col1,
            nombre_col2=nombre_col2,
            eliminar_original=eliminar_original_split
        )
        print("✔ División de columna aplicada")

    df_datos = df_out
    import time, re

    t0 = time.perf_counter()
    logs = []
    meta_detalle = {}

    # --- utilidades de tokens (ajústalas a tus marcadores reales) ---
    # data: ||...||
    _re_data = re.compile(r"\|\|.*?\|\|", re.S)
    # ia: +IA_  (al inicio de línea o en medio)
    _re_ia   = re.compile(r"\+", re.I)
    # plots: #GRAFICA# o #GRAFICO# o tus tags internos
    _re_plot = re.compile(r"#GRAFIC[AO]#", re.I)

    try:
        # 1) Render de variables
        t1 = time.perf_counter()
        texto_completo = columnas_a_texto(df, "Titulo", "Contenido")  # asumes que ya existe
        text = _render_vars_text(texto_completo, ctx=ctx, strict=False)
        meta_detalle["len_texto_render"] = len(text)
        logs.append("Render de variables: OK")
        meta_detalle["t_render_vars"] = round(time.perf_counter() - t1, 4)

        # Detectar tokens de cada módulo antes de llamar nada costoso
        hay_data = bool(_re_data.search(text))
        hay_ia   = bool(_re_ia.search(text))
        hay_plot = bool(_re_plot.search(text))

        # 2) Data blocks (solo si hay ||...|| en el texto)
        if hay_data:
            t2 = time.perf_counter()
            uot = process_data_blocks(df_datos, text)
            logs.append("Procesamiento de datos: OK")
            meta_detalle["t_data_blocks"] = round(time.perf_counter() - t2, 4)
            text_for_next = uot
        else:
            logs.append("Procesamiento de datos: SKIP (sin tokens)")
            text_for_next = text

        # 3) IA blocks (solo si hay +IA_)
        if hay_ia:
            t3 = time.perf_counter()
            out_ia = process_ia_blocks(text_for_next, ask_fn=ask_gpt5)
            logs.append("Análisis IA: OK")
            meta_detalle["t_ia_blocks"] = round(time.perf_counter() - t3, 4)
            text_for_next = out_ia
        else:
            logs.append("Análisis IA: SKIP (sin tokens)")
            # text_for_next se mantiene
                # 3.5) Portada y tabla de contenido opcional

        
        if generar_portada:
            t35 = time.perf_counter()
            try:
                portada = portada_gpt5(text_for_next)
                text_con_portada = portada + "\n\n" + text_for_next
                logs.append("Generación de portada y TOC: OK")
                meta_detalle["t_portada"] = round(time.perf_counter() - t35, 4)
                text_for_next = text_con_portada
            except Exception as e_port:
                logs.append(f"Generación de portada y TOC: ERROR → {e_port}")
                meta_detalle["t_portada"] = round(time.perf_counter() - t35, 4)

        
        # 4) Plot blocks (solo si hay #GRAFICA#)
        if hay_plot:
            t4 = time.perf_counter()
            # Si tu process_plot_blocks acepta kwargs tipo "fast=True"/"dpi=96"/"tight=False", pásalos aquí:
            if modo_rapido_plots:
                try:
                    out_plot = process_plot_blocks(df_datos, text_for_next, fast=True, dpi=96, tight=False)
                except TypeError:
                    # si no acepta kwargs extra, llama normal
                    out_plot = process_plot_blocks(df_datos, text_for_next)
            else:
                out_plot = process_plot_blocks(df_datos, text_for_next)

            logs.append("Procesamiento de gráficas: OK")
            meta_detalle["t_plot_blocks"] = round(time.perf_counter() - t4, 4)
            text_for_next = out_plot
        else:
            logs.append("Procesamiento de gráficas: SKIP (sin tokens)")


        text_for_next = limpieza_final(text_for_next)
        
        # 5) Exportar HTML (evitar E/S si no se requiere)
        t5 = time.perf_counter()
        if escribir_archivo:
            html_final = exportar_output_a_html(text_for_next, salida_html)
            #html_final = estructurar_documento(html_final)
            #html_final = mejorar_html_informe(html_final)
            logs.append(f"Exportación HTML final: OK → {salida_html}")
        else:
            # muchas implementaciones de exportar_output_a_html ya devuelven el string;
            # si la tuya siempre escribe a disco, crea un exportador in-memory alterno.
            html_final = exportar_output_a_html(text_for_next, None)
            #html_final = estructurar_documento(html_final)
            #html_final = mejorar_html_informe(html_final)
              # si tu función permite None para solo string
            logs.append("Exportación HTML final: OK (sin escribir a disco)")
        meta_detalle["t_export_html"] = round(time.perf_counter() - t5, 4)

        duracion = round(time.perf_counter() - t0, 4)
        meta = {
            "status": "OK",
            "duracion_seg": duracion,
            "logs": logs,
            "detalle": meta_detalle,
            "tokens_detectados": {
                "data_blocks": hay_data,
                "ia_blocks": hay_ia,
                "plot_blocks": hay_plot
            }
        }
        return html_final, meta

    except Exception as e:
        duracion = round(time.perf_counter() - t0, 4)
        logs.append(f"ERROR general: {e}")
        meta = {"status": "ERROR", "error": str(e), "duracion_seg": duracion, "logs": logs, "detalle": meta_detalle}
        return "", meta
