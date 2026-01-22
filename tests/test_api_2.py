
from sympy import false
from Colmedicos.io_utils_remaster import process_ia_blocks, process_data_blocks, process_plot_blocks, _render_vars_text, exportar_output_a_html, _fig_to_data_uri, _format_result_plain, columnas_a_texto, aplicar_multiples_columnas_gpt5, unpivot_df, dividir_columna_en_dos, procesar_codigos_cie10, unir_dataframes, expand_json_column, procesar_apendices, filtrar_apendices,process_titulo_blocks
import pandas as pd
from Colmedicos.ia import ask_gpt5, operaciones_gpt5, graficos_gpt5, columns_batch_gpt5, apendices_gpt5
from Colmedicos.io_utils import aplicar_plot_por_tipo_desde_output, aplicar_ia_por_tipo, generar_output, mostrar_html
from Colmedicos.charts import plot_from_params
from Colmedicos.math_ops import ejecutar_operaciones_condicionales
from Colmedicos.api import informe_final


otra_ruta = r"C:\Users\EstebanEscuderoPuert\Downloads\muestra_plantilla.xlsx"
# Ruta del archivo Excel
ruta_archivo = r"C:\Users\EstebanEscuderoPuert\Downloads\Plantilla.xlsx"
# Lee el archivo Excel (por defecto lee la primera hoja)
df = pd.read_excel(ruta_archivo)

ctx = {
    "nombre_cliente": "TCC S.A.S.",
    "nit_cliente": "860016640-4",
    "fecha_inicio": "2025-05-01",
    "fecha_fin": "2025-09-31",
    "numero_personas": 51,
}
# Ruta del archivo Excel
ruta_archivos = r"C:\Users\EstebanEscuderoPuert\Downloads\prueba 4.xlsx"
df_datos = pd.read_excel(ruta_archivos)
maes_cie10 = r"C:\Users\EstebanEscuderoPuert\Downloads\cie10_maestro_limpio.xlsx"
df_maestro = pd.read_excel(maes_cie10)


columnas = [
    "Pruebas Infecciosas (coprologico)-Blastocistis Hominis",
    "Pruebas Infecciosas (coprologico)-Helmintos",
    "Pruebas Infecciosas (coprologico)-Leucocitos",
    "Pruebas Infecciosas (coprologico)-Protozoos",
    "Pruebas Infecciosas-Frotis De Uñas",
    "Pruebas Infecciosas-Frotis Faringeo",
    "Hemograma-Hematocrito",
    "Hemograma-Hemoglobina",
    "Hemograma-Linfocitos",
    "Hemograma-Leucocitos",
    "Hemograma-Mxd # (Eosinofilos, Basofilos, Monocitos)",
    "Hemograma-Plaquetas (Plt)",
    "Perfil Renal-Nitrogeno Ureico",
    "Perfil Renal-Creatinina En Suero",
    "Perfil Metabolico-Colesterol Hdl",
    "Perfil Metabolico-Colesterol Ldl",
    "Perfil Metabolico-Colesterol Total",
    "Perfil Metabolico-Trigliceridos",
    "Perfil Metabolico-Glicemia Simple",
    "Perfil Metabolico-Tsh Hormona Estimulante Tiroides",
    "Perfil Hepatico-Transaminasa Glutamico Oxalacetina (Tgo-Ast)",
    "Perfil Hepatico-Transaminasa Glutamico Piruvica (Tgp-Alt)",
    "Alcohol y sustancias psicoactivas-Anfetaminas",
    "Alcohol y sustancias psicoactivas-Benzodiacepinas",
    "Alcohol y sustancias psicoactivas-Alcohol",
    "Alcohol y sustancias psicoactivas-Cocaina",
    "Alcohol y sustancias psicoactivas-Marihuana",
    "Alcohol y sustancias psicoactivas-Morfina (Opiaceos)",
    "Citoquimico de Orina-Densidad (Orina)",
    "Citoquimico de Orina-Ph(Orina)",
    "Citoquimico de Orina-Nitritos(Orina)",
    "Citoquimico de Orina-Leucocitos(Orina)",
    "Citoquimico de Orina-Proteinas(Orina)",
    "Citoquimico de Orina-Sangre En Orina",
    "Citoquimico de Orina-Sedimento(Orina)",
    "Hemoclasificacion-Hemoclasificacion ( Grupo)",
    "Hemoclasificacion-Hemoclasificacion ( Rh )",
    "NA-Plomo en Sangre",
    "NA-Cromo en orina",
    "NA-Niquel en orina"
]


tareas = [
    {
      "criterios": {
        "No caso": "Se usa si trabajadores que no presentan síntomas...",
        "Sintomático": "Se usa si trabajadores que refieren molestias...",
        "Caso confirmado": "Se usa si trabajadores que presentan síntomas persistentes..."
      },
      "registro_cols": "obs_osteomuscular",
      "nueva_columna": "Clas_osteomuscular"
    },
    {
  "criterios": {
    "factores": [
      {"nombre":"imc_alto","condicion":"imc >= 25"},
      {"nombre":"presion_alta","condicion":"(presion startswith '13' OR presion startswith '14' OR presion startswith '15')"},
      {"nombre":"talla_riesgo","condicion":"(genero == 'M' AND cintura > 102) OR (genero == 'F' AND cintura > 88)"}
    ],
    "conteo": "conteo_factores",
    "Riesgo Bajo": "conteo_factores == 0",
    "Riesgo Moderado": "conteo_factores == 1",
    "Riesgo Alto": "conteo_factores >= 2"
  },
  "registro_cols": ["cintura", "imc", "presion", "genero"],
  "nueva_columna": "tipo_riesgo_cardiovascular"
    },
    {
      "criterios": {
        "SI": "Se usa si los trabajadores reportan o manifiestan síntomas...",
        "NO": "Se usa si los trabajadores NO reportan síntomas..."
      },
      "registro_cols": "obs_revsistemas",
      "nueva_columna": "Reporte sintomatologia"
    },

    {
      "criterios": {
        "No refiere antecedentes patologicos ocupacionales": "Se usa si se manifiesta que no refiere antecedentes...",
        "Si refiere antecedentes patologicos ocupacionales": "Se usa en todos los demás casos..."
      },
      "registro_cols": "obs_antecedpatocupacional",
      "nueva_columna": "Antecedentes Patologicos"
    },

    {
      "criterios": {
        "No": "Se usa si no hay ningun valor, manifiesta que no practica deporte o es exdeportista",
        "Si": "Si manifiesta periodicidad"
      },
      "registro_cols": "habitos_deportes1",
      "nueva_columna": "Practica deporte regularmente"
    },

    {
      "criterios": {
        "No": "Se usa si es un exbebedor o está vacío o manifiesta que no bebe",
        "Si": "Se usa si manifiesta que bebe"
      },
      "registro_cols": "habitos_licor1",
      "nueva_columna": "Consume licor regularmente"
    },

    {
      "criterios": {
        "NO REALIZADA.": "Se clasifica como NO REALIZADA cuando el texto indica ausencia de prueba o que no aplica, incluyendo expresiones como 'no realizada', 'no aplica', con o sin puntuación.",
        "REALIZADA.": "Se clasifica como REALIZADA cuando la prueba fue efectivamente realizada, independientemente de si se describe resultado o no.",
        "Su capacidad visual es adecuada para la ocupación.": "Se usa cuando se describe capacidad visual adecuada, normal o sin alteraciones relevantes para el desempeño laboral habitual.",
        "Su capacidad visual actual es adecuada, con el uso de la corrección formulada.": "Se usa cuando la visión es adecuada gracias al uso de lentes o corrección óptica formulada.",
        "Su capacidad visual es deficiente pero no le genera restricciones para la ocupación. Requiere ser corregida.": "Se usa cuando la capacidad visual presenta disminución o deficiencia leve que puede corregirse y no genera restricciones laborales.",
        "Su capacidad visual es insuficiente y requiere evaluación por especialista para establecer posibilidad de mejoramiento.": "Se usa cuando la visión está disminuida de forma significativa y requiere valoración por oftalmología u optometría especializada.",
        "Tiene una pérdida de su capacidad visual por un ojo y no es posible su corrección o mejoramiento.": "Se usa cuando existe pérdida visual irreversible en un ojo sin posibilidad de mejoría."
      },
      "registro_cols": "visiometria",
      "nueva_columna": "visiometria"
    },

    {
      "criterios": {
        "NO REALIZADA.": "Se usa cuando la prueba no fue realizada o se reporta como no aplica, incluyendo variaciones en puntuación o mayúsculas.",
        "REALIZADA.": "Se usa cuando la espirometría fue efectivamente realizada.",
        "Su capacidad respiratoria es adecuada para la ocupación.": "Se usa cuando la capacidad respiratoria es normal o adecuada para el trabajo, sin limitaciones clínicamente relevantes.",
        "Su capacidad respiratoria está ligeramente alterada.": "Se usa cuando se describe una alteración leve, sin repercusión ocupacional significativa.",
        "Su capacidad respiratoria está disminuida. No se recomienda que labore en ambientes con factores de riesgo respiratorio.": "Se usa cuando la prueba indica disminución respiratoria moderada o marcada que contraindica la exposición a ambientes con riesgo respiratorio.",
        "Su capacidad respiratoria está muy disminuida y le genera restricción para laborar a ambientes con factores de riesgo respiratorio.": "Se usa en casos de disminución severa o grave con clara limitación funcional."
      },
      "registro_cols": "espirometria",
      "nueva_columna": "espirometria"
    },
    {
      "criterios": {
        "NO REALIZADA.": "Se usa cuando la prueba no fue realizada o se indica que no aplica, independientemente de la variación en la redacción.",
        "REALIZADA.": "Se asigna cuando se confirma que la audiometría fue realizada.",
        "Su capacidad auditiva es adecuada para la ocupación.": "Se usa cuando se describe audición normal, adecuada o sin alteración relevante para el trabajo.",
        "Su capacidad auditiva está disminuida unilateralmente.": "Se usa cuando se describe disminución auditiva en un solo oído, sin indicar imposibilidad total ni bilateralidad.",
        "Su capacidad auditiva está disminuída y le genera restricciones para exponerse a ruido.": "Se usa cuando se reporta disminución auditiva significativa que implica restricción o precaución para exposición a ruido.",
        "Su capacidad auditiva está muy deteriorada y no debe exponerse a ruido bajo ninguna circunstancia.": "Se usa cuando la audición está altamente comprometida y requiere evitar totalmente la exposición a ruido.",
        "Su capacidad auditiva está disminuida, pero puede exponerse a ruido, con el uso permanente de la protección adecuada y el seguimiento necesario.": "Se usa cuando existe disminución auditiva, pero la persona puede exponerse a ruido bajo condiciones estrictas de protección y seguimiento.",
        "Su capacidad auditiva está disminuida, pero con relación a la Audiometría de Base está estable.": "Se usa cuando hay disminución auditiva pero se mantiene estable respecto a la audiometría de referencia."
      },
      "registro_cols": "audiometria",
      "nueva_columna": "audiometria"
    },

    {
      "criterios": {
        "NO REALIZADA.": "Se clasifica como NO REALIZADA si el texto indica ausencia de prueba, incluyendo variantes como 'no aplica', 'no realizada', con o sin puntos, mayúsculas o duplicaciones.",
        "REALIZADA.": "Se clasifica como REALIZADA si el texto indica explícitamente realización de la prueba.",
        "Su capacidad visual actual es adecuada para la ocupación.": "Se usa si el texto expresa que la capacidad visual actual es adecuada o normal para la ocupación.",
        "Su capacidad visual actual es adecuada, con el uso de la corrección formulada.": "Se usa si el texto indica que la capacidad visual es adecuada gracias a corrección óptica.",
        "Su capacidad visual actual se encuentra disminuida por pérdida de la visión por un ojo que no es posible mejorar.": "Se usa si se identifica disminución permanente por pérdida visual en un ojo.",
        "Su capacidad visual actual se encuentra disminuida y requiere corrección óptica.": "Se usa si existe disminución corregible mediante fórmula óptica.",
        "Su capacidad visual actual se encuentra disminuida y debe actualizar su corrección óptica.": "Se usa cuando el texto indica que debe actualizar fórmula óptica.",
        "Su capacidad visual actual se encuentra alterada.": "Se usa para alteración general sin clasificación específica.",
        "Su capacidad visual actual es adecuada con la presencia de una alteración cromática.": "Se usa si existe alteración cromática con visión adecuada.",
        "Su capacidad visual actual es adecuada, con la presencia de una alteración en percepción de la profundidad.": "Se usa si se menciona alteración de percepción de profundidad.",
        "Su capacidad visual actual se encuentra disminuida y debe ser evaluada por el Médico Oftalmólogo o Especialista para establecer posibilidad de mejoramiento.": "Se usa cuando requiere evaluación especializada."
      },
      "registro_cols": "optometria",
      "nueva_columna": "optometria"
    },

    {
      "criterios": {
        "Si": "Se clasifica como Si cuando en el numeral 1 del texto se menciona un factor de riesgo de tipo ergonómico.",
        "No": "Se clasifica como No cuando no se identifica referencia a riesgo ergonómico."
      },
      "registro_cols": "obs_antecedocupacional",
      "nueva_columna": "riesgo_ergonomico"
    },

    {
      "criterios": {
        "Si": "Se clasifica como Si cuando en el numeral 1 del registro se menciona un factor de riesgo de tipo biomecánico.",
        "No": "Se clasifica como No cuando no se identifica referencia a riesgo biomecánico."
      },
      "registro_cols": "obs_antecedocupacional",
      "nueva_columna": "riesgo_biomecanico"
    },

    {
      "criterios": {
        "Si": "Se clasifica como Si cuando en el numeral 1 del registro se menciona un factor de riesgo psicosocial.",
        "No": "Se clasifica como No cuando no se identifica referencia a riesgo psicosocial."
      },
      "registro_cols": "obs_antecedocupacional",
      "nueva_columna": "riesgo_psicosocial"
    },

    {
      "criterios": {
        "Si": "Se clasifica como Si cuando en el numeral 1 del registro se menciona un factor de riesgo químico.",
        "No": "Se clasifica como No cuando no se identifica referencia a riesgo químico."
      },
      "registro_cols": "obs_antecedocupacional",
      "nueva_columna": "riesgo_quimico"
    },    
    {
  "criterios": {
    "resultado_años": "Calculo: (fecha_hoy - fecha_ingreso) / 365.25",
    "sin_datos": "Si fecha_ingreso es nula, retornar 0"
  },
  "registro_cols": "fecha_ingreso",
  "nueva_columna": "antiguedad"
    },
    {"criterios": {
        "AUXILIAR": "Se usa si el cargo del trabajador es AUXILIAR o similares",
        "OPERARIO": "Se usa si el cargo del trabajador es OPERARIO o similares",
        "ADMINISTRATIVO": "Se usa si el cargo del trabajador es ADMINISTRATIVO o similares",
        "INGENIERO": "Se usa si el cargo del trabajador es INGENIERO o similares",
        "TECNICO": "Se usa si el cargo del trabajador es TECNICO o similares",
        "ANALISTA": "Se usa si el cargo del trabajador es ANALISTA o similares",
        "VENDEDOR": "Se usa si el cargo del trabajador es VENDEDOR o similares",
        "COORDINADOR": "Se usa si el cargo del trabajador es COORDINADOR o similares",
        "GERENTE": "Se usa si el cargo del trabajador es GERENTE o similares",
        "PUBLICISTA": "Se usa si el cargo del trabajador es PUBLICISTA o similares",
        "DIRECTOR": "Se usa si el cargo del trabajador es DIRECTOR o similares",
        "OTRO": "Se usa para todos los demás cargos no clasificados en las categorías anteriores"
    },
      "registro_cols": "ocupacion",
      "nueva_columna": "categoria_cargo"
    },
    {"criterios": {
        "No": "Se usa si es un exfumador, está vacío o manifiesta que no fuma",
        "Si": "Se usa si manifiesta que fuma"
      },
      "registro_cols": "habitos_tabaquismo1",
      "nueva_columna": "Fuma regularmente"
    },
    {"criterios":{"Es satisfactorio": "Se usa cuando el concepto médico indica explicitamente que es satisfactorios o positivo, además todo lo que no cabe en las otras definiciones", 
                  "Es necesario expedir recomendaciones medicas en el trabajo": "Se usa cuando en el concepto se hace referencia a dejar algunas recomendaciones", 
                  "Presenta alteración en su estado de salud que no le impide desempeñar su trabajo habitual": "Se clasifica en esta, cuando de forma explicita indica que presenta alteraciones en su salud pero que no le impide desempeñar sus funciones"
      },
      "registro_cols": "concepto_aptitud",
      "nueva_columna": "concepto_aptitud"
    }
]

campos = ["lab_grupo", "lab_item_unificado", "lab_Resultado_global"]
renombres = {
    "lab_grupo": "Tipo prueba",
    "lab_item_unificado": "Prueba",
    "lab_Resultado_global": "Resultado"
}

inf, meta, df_datico, json_op, json_grafos = informe_final(df,
                          df_datos,
                          ctx,
                          tareas=tareas,
                          salida_html=r"C:\Users\EstebanEscuderoPuert\Downloads\Informe pruebas colmedicos\informes\informe_prueba1.html",
                          aplicar_cie10=True,
                          aplicar_union=True, 
                          aplicar_expansion_json=True,
                          col_texto_cie10="obs_diagnostico", 
                          col_df1="obs_diagnostico", 
                          col_df2="Code", 
                          json_columna="laboratorios_incluidos",
                          campos_a_extraer=campos,
                          renombrar_campos=renombres
                          )


print(meta)
df_datico.to_excel(r"C:\Users\EstebanEscuderoPuert\Downloads\output_.xlsx", index=False, engine="openpyxl")

with open(r"C:\Users\EstebanEscuderoPuert\Downloads\output_json_op.txt", "w", encoding="utf-8") as f:
    f.write(json_op)

with open(r"C:\Users\EstebanEscuderoPuert\Downloads\output_json_grafos.txt", "w", encoding="utf-8") as f:
    f.write(json_grafos)

#---------------------
# # # Aplicar las tareas definidas al otro DataFrame
#---------------------

# ruta_archivos = r"C:\Users\EstebanEscuderoPuert\Downloads\MatrizInformesDx - TCC (1).xlsx"
# df_otro = pd.read_excel(ruta_archivos)
# out = aplicar_multiples_columnas_gpt5(df_otro, tareas=tareas)
# df_otro = procesar_codigos_cie10(out, columna_texto="obs_diagnostico")
# df_otro = unir_dataframes(df_otro,df_maestro,col_df1="obs_diagnostico",col_df2="Code")
# df_otro = expand_json_column(df_otro,"laboratorios_incluidos",fields_to_extract=campos,rename_map=renombres)
# #df_otro = unpivot_df(df_otro,columnas_unpivot=columnas)
# #df_otro = dividir_columna_en_dos(df_otro,columna="variable",caracter_separador="-",nombre_col1="Tipo prueba",nombre_col2="Prueba", eliminar_original=True)
#df_otro.to_excel(r"C:\Users\EstebanEscuderoPuert\Downloads\output_.xlsx", index=False, engine="openpyxl")
# print(df_otro.head(10))

#out.to_excel(r"C:\Users\EstebanEscuderoPuert\Downloads\output_.xlsx", index=False, engine="openpyxl")

#instruccion = """ "No caso": no se presentan sintomas , "Sintomatico": se presentan sintomas o "Observado": fue identificado durante el examen."""
#df_datos = aplicar_columnas_gpt5(df_datos, instruccion,
#                                 columnas=["obs_osteomuscular"],
#                                 nueva_columna="analisis_osteomuscular")

#df_datos.to_excel(r"C:\Users\EstebanEscuderoPuert\Downloads\output_.xlsx", index=False, engine="openpyxl")

# texto_completo = columnas_a_texto(df,"Titulo","Contenido")


# out2 = process_data_blocks(df_otro,texto_completo)
# # out2 = procesar_apendices(out2)
# out2, tabla = process_titulo_blocks(out2)
# out2 = tabla + out2
# filtro = {
#   "conservar": [6, 7, 11, 12, 13, 14, 17, 18, 19, 20, 21, 23, 25, 26, 27, 29, 30, 31, 32, 34, 36, 37, 38, 39, 44, 45],
#   "borrar": [1, 2, 3, 4, 5, 8, 9, 10, 15, 16, 22, 24, 28, 33, 35, 40, 41, 42, 43]
# }
# out2 = filtrar_apendices(texto_completo,filtro)
#out2 = procesar_apendices(out2)
# # out2 = process_ia_blocks(out2)
# out2 = process_plot_blocks(df_otro, texto_completo)

# from Colmedicos.io_utils_remaster import extraer_data_blocks, aplicar_operaciones_en_texto
# out = extraer_data_blocks(texto_completo)

# out2 = operaciones_gpt5(df_otro, out)
# print(out)
# print(out2)
# for c in df_otro.columns:
#     print(repr(c))

# from typing import Any, Dict, List, Tuple, Union, Optional, Callable
# resultados_ops: List[
#     Tuple[int, Dict[str, Any], Union[Tuple[int, int], None], str]
# ] = []

# for item in out2:
#     if isinstance(item, dict) and "params" in item:
#         idx = item.get("idx")
#         params = item.get("params")
#         span = item.get("span")

#         try:
#             resultado = ejecutar_operaciones_condicionales(df_otro, params)
#             resultado_fmt = _format_result_plain(resultado)
#             resultados_ops.append((idx, params, span, resultado_fmt))
#         except Exception as e:
#             # Mantener trazabilidad sin romper el tipo (resultado como string legible)
#             error_txt = f"[error:{str(e)}]"
#             resultados_ops.append((idx, params, span, error_txt))
# print(resultados_ops)
# out2 = aplicar_operaciones_en_texto(texto_completo, resultados_ops, formato="html")
#print(out2)

# with open(r"C:\Users\EstebanEscuderoPuert\Downloads\output_1.txt", "w", encoding="utf-8") as f:
#     f.write(out2)
# out1 = operaciones_gpt5(df_otro,texto_completo)


#with open(r"C:\Users\EstebanEscuderoPuert\Downloads\output_.txt", "w", encoding="utf-8") as f:
#    json.dump(out2, f, indent=4, ensure_ascii=False)

#with open(r"C:\Users\EstebanEscuderoPuert\Downloads\output_1.txt", "w", encoding="utf-8") as f:
#    f.write(out2)
# print(out1)


# rutica_archivos = r"C:\Users\EstebanEscuderoPuert\Downloads\Informe pruebas colmedicos\Emtelco_colmedicos.xlsx"
# df_dato = pd.read_excel(rutica_archivos)
#out1 = operaciones_gpt5(df_otro,textico)
#out1 = unir_idx_params_con_span_json_data(out1, textico)
#import json
#config_list = json.loads(out)
#params = out1[0]["params"]
# print(out1)
# variable = [{'operations': [{'op': 'distinct_count', 'alias': 'conteo_unico_personas_por_prueba_resultado', 'column': 'documento', 'conditions': [['Tipo de prueba', '==', 'Perfil Metabolico']], 'conditions_logic': 'AND', 'condition_groups': [], 'dedupe_by': None, 'count_nulls': False, 'numerator': None, 'denominator': None, 'weights': None, 'safe_div0': None}], 'group_by': ['Prueba', 'Resultado'], 'needs_disambiguation': False, 'candidates': {'columns': [], 'group_by': [], 'by_operation': []}}]
# var = variable[0]
# out2 = ejecutar_operaciones_condicionales(df_dato,var)
# print(out2)
# out1 = process_data_blocks(df_dato,textico)
# print(out1)
# out1 = process_ia_blocks(out1)
# print(out1)
#out = portada_gpt5(texto_completo)
#out = texto_completo + "\n\n" + out



# print(out)
# import json
# config_list = json.loads(out)
# params = config_list[0]["params"]
# fig, ax = plot_from_params(df_otro,params)
# out = _fig_to_data_uri(fig)
# #print(out)


#with open(r"C:\Users\EstebanEscuderoPuert\Downloads\output_.txt", "w", encoding="utf-8") as f:
#    f.write(df_datos)





# for item in out:
#     if isinstance(item, dict) and "params" in item:
#         idx, params, span = item["idx"], item["params"], item["span"]
#         params_list.append(params)
#         try:
#         fig, ax = plot_from_params(df, params)
# print(texto_completo)
# out = process_ia_blocks(texto_completo,ask_gpt5)
# print(out)
# out = ask_gpt5(texto_completo)
# print(out)
# df_final, res = procesar_df_una_fila(
#    df,
#    col_output="Output",
#    ctx=ctx,
#    df_datos=df_datos,
#    ask_fn=ask_gpt5,          # si necesitas IA
#    export_html=True,
#    titulo_html="Informe generado"
# )

