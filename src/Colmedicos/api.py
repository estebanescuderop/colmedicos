
from Colmedicos.ia import ask_gpt5
from Colmedicos.io_utils import generar_output, aplicar_data_por_tipo_desde_output, aplicar_ia_por_tipo, aplicar_plot_por_tipo_desde_output, exportar_output_a_html, mostrar_html, limpiar_output_dataframe
from Colmedicos.registry import register

# Colmedicos/api.py
import time
import pandas as pd
from typing import Any, Dict, Tuple

#from Colmedicos.ia import ask_gpt5
#from Colmedicos.math_ops import aplicar_plot_por_tipo_desde_output
#from Colmedicos.charts import mostrar_html
#from Colmedicos.io_utils import exportar_output_a_html
#from Colmedicos.registry import (
#    generar_output,
#    aplicar_data_por_tipo_desde_output,
#    aplicar_ia_por_tipo,
#)

@register("informe_final")
def informe_final(
    df: pd.DataFrame,
    df_datos: pd.DataFrame,
    ctx: dict,
    valor_tipo_objetivo: str = "Fijo con IA",
    reemplazar_en_html: bool = True,
    token_reemplazo: str = "#GRAFICA#",
) -> Tuple[str, Dict[str, Any]]:
    """
    Procesa df (resumen) y df_datos (detalle) con el contexto ctx y genera:
      - html_final (str): documento HTML con <img ...> embebidos listo para entregar
      - meta (dict): resumen técnico del pipeline

    Steps (alto nivel):
      1. generar_output -> df_out base con columnas Titulo / Contenido -> Output inicial
      2. aplicar_data_por_tipo_desde_output -> mezcla df_out con df_datos
      3. aplicar_ia_por_tipo -> aplica IA (ask_gpt5) sobre cierto tipo (ej. "Fijo con IA")
      4. aplicar_plot_por_tipo_desde_output -> inserta <img src="data:image/png;base64,..."/>
      5. exportar_output_a_html -> arma el HTML concatenando Output
    """

    t0 = time.time()
    logs: list[str] = []
    ia_stats = {
        "total_intentos_ia": 0,
        "total_errores_ia": 0,
    }
    plot_stats = {
        "graficas_insertadas": 0,
    }

    # --- Paso 1: generar_output ---
    # df_out: DataFrame con columnas Tipo / Output / etc.
    try:
        df_out = generar_output(
            df,
            col_texto="Titulo",
            col_contenido="Contenido",
            ctx=ctx,
            strict_ctx=False  # si falta alguna variable en ctx, no revientes
        )
        logs.append("generar_output: OK")
    except Exception as e:
        logs.append(f"generar_output: ERROR {e}")
        raise

    # --- Paso 2: aplicar_data_por_tipo_desde_output ---
    # Combina/enriquece df_out usando df_datos
    try:
        df_enriquecido = aplicar_data_por_tipo_desde_output(
            df_out,
            df_datos
        )
        logs.append("aplicar_data_por_tipo_desde_output: OK")
    except Exception as e:
        logs.append(f"aplicar_data_por_tipo_desde_output: ERROR {e}")
        # si esto falla, ya no podemos seguir coherentemente
        raise

    # --- Paso 3: aplicar_ia_por_tipo ---
    # Corre IA para filas con cierto tipo (ej. "Fijo con IA")
    # Necesitamos capturar cuántas veces falló la IA.
    try:
        # asumo que aplicar_ia_por_tipo MODIFICA textos en la col "Output"
        # y que internamente ya maneja on_error="keep"
        df_ai = aplicar_ia_por_tipo(
            df_enriquecido,
            ask_fn=ask_gpt5,
            col_tipo="Tipo",
            col_output="Output",
            valor_tipo_objetivo=valor_tipo_objetivo,
            overwrite=True,    # sobrescribe Output con el texto generado
            on_error="keep"    # si la IA falla, deja el original
        )
        logs.append("aplicar_ia_por_tipo: OK")

        # Aquí podemos inferir estadísticas de IA si tu función deja alguna marca,
        # pero como no tenemos ese detalle exacto, vamos a estimar:
        #   total_intentos_ia = número de filas cuyo Tipo == valor_tipo_objetivo
        #   total_errores_ia  = conteo de filas donde Output contiene "[ERROR LLM"
        # (esto asume que el fallback de ask_gpt5 mete esa marca cuando hay 429 / sin cuota)
        mask_objetivo = (df_enriquecido["Tipo"] == valor_tipo_objetivo)
        ia_stats["total_intentos_ia"] = int(mask_objetivo.sum())

        if "Output" in df_ai.columns:
            ia_stats["total_errores_ia"] = int(
                df_ai["Output"]
                .astype(str)
                .str.contains("[ERROR LLM", regex=False)
                .sum()
            )

    except Exception as e:
        # Si por algún motivo aplicar_ia_por_tipo truena globalmente,
        # no matamos todo el pipeline: seguimos sin IA.
        logs.append(f"aplicar_ia_por_tipo: ERROR {e} (continuando sin IA)")
        df_ai = df_enriquecido.copy()
        # Stats IA: intentos = filas objetivo, errores = todos ellos
        mask_objetivo = (df_enriquecido["Tipo"] == valor_tipo_objetivo)
        ia_stats["total_intentos_ia"] = int(mask_objetivo.sum())
        ia_stats["total_errores_ia"] = int(mask_objetivo.sum())

    # --- Paso 4: aplicar_plot_por_tipo_desde_output ---
    # Inserta <img ... base64> dentro de Output reemplazando un token
    try:
        # según lo que mostraste antes, aplicar_plot_por_tipo_desde_output
        # devuelve (df_plot, info_graficas)
        df_plot, info_plots = aplicar_plot_por_tipo_desde_output(
            df_ai,
            df_datos,
            col_tipo="Tipo",
            col_output="Output",
            valor_tipo_objetivo=valor_tipo_objetivo,
            verbose=True,
            reemplazar_en_html=True,
            token_reemplazo=token_reemplazo,
            replace_all=True,
            inplace=False  # mejor no mutar df_ai original, devolveme copia
        )
        logs.append("aplicar_plot_por_tipo_desde_output: OK")

        # plot_stats: cuántas gráficas se insertaron
        if isinstance(info_plots, list):
            plot_stats["graficas_insertadas"] = len(info_plots)
        else:
            plot_stats["graficas_insertadas"] = 0

    except Exception as e:
        logs.append(f"aplicar_plot_por_tipo_desde_output: ERROR {e} (continuando sin plots)")
        df_plot = df_ai.copy()
        df_plot = limpiar_output_dataframe(df_plot)
        plot_stats["graficas_insertadas"] = 0

    # --- Paso 5: exportar_output_a_html ---
    # Convierte df_plot["Output"] en un documento HTML final concatenado
    try:
        html_str = exportar_output_a_html(
            df_plot,
            col_output="Output",
            archivo_html="salida.html",   # si escribes a disco internamente, ok
            titulo="Documento",
            escapar_html=False,           # dejamos <img src="...">
            separar_por_dobles_saltos=True
        )
        logs.append("exportar_output_a_html: OK")
    except Exception as e:
        logs.append(f"exportar_output_a_html: ERROR {e}")
        # si falla exportar_output_a_html, no hay informe
        raise

    # --- Paso 6: mostrar_html (si es necesario postprocesar/renderizar)
    try:
        html_renderizado = html_str
        logs.append("mostrar_html: OK")
    except Exception as e:
        logs.append(f"mostrar_html: ERROR {e} (usando html_str sin render)")
        html_renderizado = html_str

    # --- Paso 7: construir meta final ---
    duracion_seg = time.time() - t0

    meta: Dict[str, Any] = {
        "cliente": {
            "nombre_cliente": ctx.get("nombre_cliente"),
            "nit_cliente": ctx.get("nit_cliente"),
            "fecha_inicio": ctx.get("fecha_inicio"),
            "fecha_fin": ctx.get("fecha_fin"),
            "numero_personas": ctx.get("numero_personas"),
            "totales": ctx.get("totales"),
        },
        "tamaño_entrada": {
            "df_rows": int(len(df)),
            "df_cols": list(df.columns),
            "df_datos_rows": int(len(df_datos)),
            "df_datos_cols": list(df_datos.columns),
        },
        "tamaño_salida": {
            "df_plot_rows": int(len(df_plot)),
            "df_plot_cols": list(df_plot.columns),
        },
        "ia_stats": ia_stats,
        "plot_stats": plot_stats,
        "pipeline": {
            "logs": logs,
            "duracion_seg": duracion_seg,
            "valor_tipo_objetivo": valor_tipo_objetivo,
            "token_reemplazo": token_reemplazo,
            "reemplazar_en_html": reemplazar_en_html,
        },
    }

    # devolvemos (html_final, meta)
    return html_renderizado, meta


def informe_final_test(
    df: pd.DataFrame,
    df_datos: pd.DataFrame,
    ctx: dict,
    valor_tipo_objetivo: str = "Fijo con IA",
    reemplazar_en_html: bool = True,
    token_reemplazo: str = "#GRAFICA#",
) -> Tuple[str, Dict[str, Any]]:
    """
    Procesa df (resumen) y df_datos (detalle) con el contexto ctx y genera:
      - html_final (str): documento HTML con <img ...> embebidos listo para entregar
      - meta (dict): resumen técnico del pipeline

    Steps (alto nivel):
      1. generar_output -> df_out base con columnas Titulo / Contenido -> Output inicial
      2. aplicar_data_por_tipo_desde_output -> mezcla df_out con df_datos
      3. aplicar_ia_por_tipo -> aplica IA (ask_gpt5) sobre cierto tipo (ej. "Fijo con IA")
      4. aplicar_plot_por_tipo_desde_output -> inserta <img src="data:image/png;base64,..."/>
      5. exportar_output_a_html -> arma el HTML concatenando Output
    """

    t0 = time.time()
    logs: list[str] = []
    ia_stats = {
        "total_intentos_ia": 0,
        "total_errores_ia": 0,
    }
    plot_stats = {
        "graficas_insertadas": 0,
    }

    # --- Paso 1: generar_output ---
    # df_out: DataFrame con columnas Tipo / Output / etc.
    try:
        df_out = generar_output(
            df,
            col_texto="Titulo",
            col_contenido="Contenido",
            ctx=ctx,
            strict_ctx=False  # si falta alguna variable en ctx, no revientes
        )
        logs.append("generar_output: OK")
    except Exception as e:
        logs.append(f"generar_output: ERROR {e}")
        raise

    # --- Paso 2: aplicar_data_por_tipo_desde_output ---
    # Combina/enriquece df_out usando df_datos
    try:
        df_enriquecido = aplicar_data_por_tipo_desde_output(
            df_out,
            df_datos
        )
        logs.append("aplicar_data_por_tipo_desde_output: OK")
    except Exception as e:
        logs.append(f"aplicar_data_por_tipo_desde_output: ERROR {e}")
        # si esto falla, ya no podemos seguir coherentemente
        raise

    # --- Paso 3: aplicar_ia_por_tipo ---
    # Corre IA para filas con cierto tipo (ej. "Fijo con IA")
    # Necesitamos capturar cuántas veces falló la IA.
    try:
        # asumo que aplicar_ia_por_tipo MODIFICA textos en la col "Output"
        # y que internamente ya maneja on_error="keep"
        df_ai = aplicar_ia_por_tipo(
            df_enriquecido,
            ask_fn=ask_gpt5,
            col_tipo="Tipo",
            col_output="Output",
            valor_tipo_objetivo=valor_tipo_objetivo,
            overwrite=True,    # sobrescribe Output con el texto generado
            on_error="keep"    # si la IA falla, deja el original
        )
        logs.append("aplicar_ia_por_tipo: OK")

        # Aquí podemos inferir estadísticas de IA si tu función deja alguna marca,
        # pero como no tenemos ese detalle exacto, vamos a estimar:
        #   total_intentos_ia = número de filas cuyo Tipo == valor_tipo_objetivo
        #   total_errores_ia  = conteo de filas donde Output contiene "[ERROR LLM"
        # (esto asume que el fallback de ask_gpt5 mete esa marca cuando hay 429 / sin cuota)
        mask_objetivo = (df_enriquecido["Tipo"] == valor_tipo_objetivo)
        ia_stats["total_intentos_ia"] = int(mask_objetivo.sum())

        if "Output" in df_ai.columns:
            ia_stats["total_errores_ia"] = int(
                df_ai["Output"]
                .astype(str)
                .str.contains("[ERROR LLM", regex=False)
                .sum()
            )

    except Exception as e:
        # Si por algún motivo aplicar_ia_por_tipo truena globalmente,
        # no matamos todo el pipeline: seguimos sin IA.
        logs.append(f"aplicar_ia_por_tipo: ERROR {e} (continuando sin IA)")
        df_ai = df_enriquecido.copy()
        # Stats IA: intentos = filas objetivo, errores = todos ellos
        mask_objetivo = (df_enriquecido["Tipo"] == valor_tipo_objetivo)
        ia_stats["total_intentos_ia"] = int(mask_objetivo.sum())
        ia_stats["total_errores_ia"] = int(mask_objetivo.sum())

    # --- Paso 4: aplicar_plot_por_tipo_desde_output ---
    # Inserta <img ... base64> dentro de Output reemplazando un token
    try:
        # según lo que mostraste antes, aplicar_plot_por_tipo_desde_output
        # devuelve (df_plot, info_graficas)
        df_plot, info_plots = aplicar_plot_por_tipo_desde_output(
            df_ai,
            df_datos,
            col_tipo="Tipo",
            col_output="Output",
            valor_tipo_objetivo=valor_tipo_objetivo,
            verbose=True,
            reemplazar_en_html=True,
            token_reemplazo=token_reemplazo,
            replace_all=True,
            inplace=False  # mejor no mutar df_ai original, devolveme copia
        )
        logs.append("aplicar_plot_por_tipo_desde_output: OK")

        # plot_stats: cuántas gráficas se insertaron
        if isinstance(info_plots, list):
            plot_stats["graficas_insertadas"] = len(info_plots)
        else:
            plot_stats["graficas_insertadas"] = 0

    except Exception as e:
        logs.append(f"aplicar_plot_por_tipo_desde_output: ERROR {e} (continuando sin plots)")
        df_plot = df_ai.copy()
        df_plot = limpiar_output_dataframe(df_plot)
        plot_stats["graficas_insertadas"] = 0

    # --- Paso 5: exportar_output_a_html ---
    # Convierte df_plot["Output"] en un documento HTML final concatenado
    try:
        html_str = exportar_output_a_html(
            df_plot,
            col_output="Output",
            archivo_html="salida.html",   # si escribes a disco internamente, ok
            titulo="Documento",
            escapar_html=False,           # dejamos <img src="...">
            separar_por_dobles_saltos=True
        )
        logs.append("exportar_output_a_html: OK")
    except Exception as e:
        logs.append(f"exportar_output_a_html: ERROR {e}")
        # si falla exportar_output_a_html, no hay informe
        raise

    # --- Paso 6: mostrar_html (si es necesario postprocesar/renderizar)
    try:
        html_renderizado = mostrar_html(html_str)
        logs.append("mostrar_html: OK")
    except Exception as e:
        logs.append(f"mostrar_html: ERROR {e} (usando html_str sin render)")
        html_renderizado = mostrar_html(html_str)

    # --- Paso 7: construir meta final ---
    duracion_seg = time.time() - t0
    print(info_plots)
    # devolvemos (html_final, meta)
    return html_renderizado
