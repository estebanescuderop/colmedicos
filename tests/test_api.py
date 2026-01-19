from Colmedicos.api import informe_final
import pandas as pd
from Colmedicos.ia import ask_gpt5
from Colmedicos.io_utils import aplicar_plot_por_tipo_desde_output, aplicar_ia_por_tipo, generar_output, exportar_output_a_html, mostrar_html

from sympy import false
from Colmedicos.io_utils_remaster import process_ia_blocks, process_data_blocks, process_plot_blocks, _render_vars_text, exportar_output_a_html, _fig_to_data_uri, _format_result_plain, columnas_a_texto, aplicar_multiples_columnas_gpt5, unpivot_df, dividir_columna_en_dos, procesar_codigos_cie10, unir_dataframes, expand_json_column, procesar_apendices, filtrar_apendices,process_titulo_blocks
import pandas as pd
from Colmedicos.ia import ask_gpt5, operaciones_gpt5, graficos_gpt5, columns_batch_gpt5, apendices_gpt5
from Colmedicos.io_utils import aplicar_plot_por_tipo_desde_output, aplicar_ia_por_tipo, generar_output, mostrar_html
from Colmedicos.charts import plot_from_params
from Colmedicos.math_ops import ejecutar_operaciones_condicionales
from Colmedicos.api import informe_final


#df_out = generar_output(df)


#df_ia = aplicar_ia_por_tipo(df_out,df_datos)

#df_final = aplicar_plot_por_tipo_desde_output(df_ia,df_datos)


#mostrar_html(exportar_output_a_html(df_final))



ruta_archivos = r"C:\Users\EstebanEscuderoPuert\Downloads\output_.xlsx"
df_datos = pd.read_excel(ruta_archivos)

# from Colmedicos.io_utils_remaster import normalizar_columna
# df_datos = normalizar_columna(df_datos, "Resultado")
# textico = """Se presenta la distribución de los diagnosticos identificados durante las valoraciones medico-ocupacionales, agrupados por sistemas organicos

#   # Gráfica de tabla llamado 'Diagnostico por sistema' con un conteo de registros únicos por columna de identificación con base a la columna de grupo#
# Tabla 10. Distribución absoluta y porcentual por sistemas afectados de patologías encontradas en la poblacion evaluada.

# Se describe la cantidad de trabajadores que presentan al menos un diagnostico en cada sistema evaluado. Es importante aclarar que la cantidad total de diagnosticos puede diferir del numero total de personas evaluadas, ya que un mismo trabajador puede presentar uno o varios diagnosticos en uno o mas organos o sistemas.

# Asimismo, se incluyen dentro del analisis a los trabajadores que no presentan alteraciones de salud, (incluir diagnosticos de pacientes sanos),  ya que esta información tambien es elevante para el analisis glibal del estado de salud de la población evaluada.

# Durante la valoracion medica, los diagnosticos reportados corresponden tanto a enfermedades actuales como previas, con diagnostico confirmado o impresiones diagnosticas clinicas identificadas durante la consulta.

# En todos los casos donde se identifico algun hallazgo clinico relevante, el medico evaluador estableció, cuando fue necesario, recomendaciones o restricciones individuales, estas forman parte del manejo integral del trabajador y deben ser objeto de seguimiento por parte del empleador, en cumplimiento de lo establecido en la Resolución 0312 de 2019, como parte del Sistema de gestion de seguridad y salud en el trabajo (SG-SST)."""

# from Colmedicos.io_utils_remaster import graficos_gpt5, plot_from_params, extraer_plot_blocks, _fig_to_data_uri
# textico = extraer_plot_blocks(textico)
# out = graficos_gpt5(df_datos,textico)
# print(out)
# import json
# config_list = json.loads(out)
# first = config_list[0] if isinstance(config_list, list) else next(iter(config_list.values()))
# if isinstance(first, str):
#     first = json.loads(first)

# params = first["params"] if "params" in first else first
# print(params)
# variable  = """{
#       "chart_type": "tabla",
#       "function_name": "graficar_tabla",
#       "title": "Espirometría",
#       "render": "html",
#       "xlabel": "espirometria",
#       "y": "documento",
#       "agg": "distinct_count",
#       "distinct_on": "documento",
#       "drop_dupes_before_sum": false,
#       "unique_by": null,
#       "conditions_all": [
#         [
#           "espirometria",
#           "!=",
#           "NO REALIZADA."
#         ]
#       ],
#       "conditions_any": [],
#       "binning": null,
#       "stack_columns": null,
#       "color": null,
#       "colors_by_category": null,
#       "legend_col": null,
#       "show_legend": false,
#       "show_values": null,
#       "sort": null,
#       "limit_categories": null,
#       "needs_disambiguation": false,
#       "candidates": {
#         "xlabel": [],
#         "y": []
#       },
#       "percentage_of": "Número trabajadores",
#       "percentage_colname": "Porcentaje",
#       "extra_measures": null,
#       "hide_main_measure": null,
#       "add_total_row": false,
#       "add_total_column": false
#     }"""
# # variable2 = params

# #variable2 = [{'chart_type': 'tabla', 'function_name': 'graficar_tabla', 'title': 'Tipo de riesgo', 'render': 'html', 'xlabel': 'categoria_cargo', 'y': 'documento', 'agg': 'distinct_count', 'distinct_on': 'documento', 'drop_dupes_before_sum': False, 'unique_by': None, 'conditions_all': [], 'conditions_any': [], 'binning': None, 'stack_columns': None, 'color': None, 'colors_by_category': None, 'legend_col': None, 'show_legend': None, 'show_values': None, 'sort': None, 'limit_categories': None, 'needs_disambiguation': False, 'candidates': {'xlabel': [], 'y': []}, 'percentage_of': None, 'percentage_colname': None, 'extra_measures': [{'name': 'Riesgo ergonomico', 'conditions_all': [['riesgo_ergonomico', '==', 'Si']], 'conditions_any': [], 'agg': 'distinct_count', 'distinct_on': 'documento', 'drop_dupes_before_sum': False}, {'name': 'Riesgo quimico', 'conditions_all': [['riesgo_quimico', '==', 'Si']], 'conditions_any': [], 'agg': 'distinct_count', 'distinct_on': 'documento', 'drop_dupes_before_sum': False}, {'name': 'Riesgo psicosocial', 'conditions_all': [['riesgo_psicosocial', '==', 'Si']], 'conditions_any': [], 'agg': 'distinct_count', 'distinct_on': 'documento', 'drop_dupes_before_sum': False}, {'name': 'Riesgo biomecanico', 'conditions_all': [['riesgo_biomecanico', '==', 'Si']], 'conditions_any': [], 'agg': 'distinct_count', 'distinct_on': 'documento', 'drop_dupes_before_sum': False}], 'hide_main_measure': False, 'add_total_row': False, 'add_total_column': False}]
variable2 = [{
      "chart_type": "tabla",
      "function_name": "graficar_tabla",
      "title": "espirometria",
      "render": "html",
      "xlabel": "espirometria",
      "y": "documento",
      "agg": "distinct_count",
      "distinct_on": "documento",
      "drop_dupes_before_sum": False,
      "unique_by": None,
      "conditions_all": [],
      "conditions_any": [],
      "binning": None,
      "stack_columns": None,
      "color": None,
      "colors_by_category": None,
      "legend_col": None,
      "show_legend": False,
      "show_values": False,
      "sort": None,
      "limit_categories": None,
      "needs_disambiguation": False,
      "candidates": {
        "xlabel": [],
        "y": []
      },
      "percentage_of": "Número trabajadores",
      "percentage_colname": "Porcentaje",
      "extra_measures": None,
      "hide_main_measure": None,
      "add_total_row": False,
      "add_total_column": False
    }]
fig, ax = plot_from_params(df_datos,variable2[0])
print(fig)

out = _fig_to_data_uri(fig)
print(out)

with open(r"C:\Users\EstebanEscuderoPuert\Downloads\output_.txt", "w", encoding="utf-8") as f:
  f.write(out)
