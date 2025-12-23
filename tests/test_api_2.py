
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


# df_otro = procesar_codigos_cie10(df_otro, columna_texto="obs_diagnostico")
# df_otro = unir_dataframes(df_otro,df_maestro,col_df1="obs_diagnostico",col_df2="Code")
# df_otro = unpivot_df(df_otro,columnas_unpivot=columnas)
# df_otro = dividir_columna_en_dos(df_otro,columna="variable",caracter_separador="-",nombre_col1="Tipo prueba",nombre_col2="Prueba", eliminar_original=True)
# df_otro.to_excel(r"C:\Users\EstebanEscuderoPuert\Downloads\output_.xlsx", index=False, engine="openpyxl")
# print(df_otro.head(10))



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
        "Riesgo Bajo": "Se usa si no hay ningún factor presente...",
        "Riesgo Moderado": "Se usa si se existe al menos 1 factor...",
        "Riesgo Alto": "Se usa si se cumplen 2 o más factores..."
      },
      "registro_cols": ["talla", "imc", "presion", "genero"],
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
       "antiguedad": "Calculo: años_antiguedad = fecha_hoy - fecha_ingreso en años."
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

inf, meta, df_datico = informe_final(df,
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


# # # # # Ruta del archivo Excel
# # ruta_archivos = r"C:\Users\EstebanEscuderoPuert\Downloads\Informe pruebas colmedicos\Prueba_mult_registros.xlsx"
# # df_date = pd.read_excel(ruta_archivos)

print(meta)
df_datico.to_excel(r"C:\Users\EstebanEscuderoPuert\Downloads\output_.xlsx", index=False, engine="openpyxl")


# out = aplicar_multiples_columnas_gpt5(df_date, tareas)

# criterios = {
#     "No caso": "Se usa si trabajadores que no presentan síntomas ni hallazgos clínicos compatibles con desórdenes musculoesqueléticos en el momento de la evaluación",
#     "Sintomático": "Se usa si trabajadores que refieren molestias musculoesqueléticas (como dolor, rigidez o fatiga muscular), pero sin evidencia clínica o funcional suficiente para confirmar un diagnostico ocupacional.",
#     "Caso confirmado": "Se usa si trabajadores que presentan síntomas persistentes acompañados de hallazgos físicos, antecedentes y pruebas clínicas"
# }
# registros = {
#   "Registros": [
#     { "idx": 0, "registro": { "habitos_tabaquismo1": "No Fuma" }},
#     { "idx": 1, "registro": { "habitos_tabaquismo1": "No Fuma" }},
#     { "idx": 2, "registro": { "habitos_tabaquismo1": "No Fuma" }},
#     { "idx": 3, "registro": { "habitos_tabaquismo1": "No Fuma" }},
#     { "idx": 4, "registro": { "habitos_tabaquismo1": "Fumador" }}
#   ]
# }

# tarea_u = {
#   "Criterios": {
#     "No": "Se usa si es un exfumador, está vacío o manifiesta que no fuma",
#     "Si": "Se usa si manifiesta que fuma"
#   }
# }

# valores = columns_batch_gpt5(criterios=tarea_u, registros=registros)
# print(valores)

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

# #df_datos.to_excel(r"C:\Users\EstebanEscuderoPuert\Downloads\output_.xlsx", index=False, engine="openpyxl")
# ruta_archivos = r"C:\Users\EstebanEscuderoPuert\Downloads\output_.xlsx"
# df_otro = pd.read_excel(ruta_archivos)

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

import json

#with open(r"C:\Users\EstebanEscuderoPuert\Downloads\output_.txt", "w", encoding="utf-8") as f:
#    json.dump(out2, f, indent=4, ensure_ascii=False)

#with open(r"C:\Users\EstebanEscuderoPuert\Downloads\output_1.txt", "w", encoding="utf-8") as f:
#    f.write(out2)
# print(out1)



# rutica_archivos = r"C:\Users\EstebanEscuderoPuert\Downloads\Informe pruebas colmedicos\Emtelco_colmedicos.xlsx"
# df_dato = pd.read_excel(rutica_archivos)

textico = """La exposición laboral actual hace referencia a la presencia de factores de riesgo ocupacionales a los cuales se encuentran expuestos los trabajadores en el desarrollo de sus funciones, de manera continua o intermitente, en su entorno laboral actual. Estos factores pueden ser de origen físico, químico, biológico, ergonómico, psicosocial, mecánico, etc, y su identificación se realiza con base en la información recolectada durante las valoraciones médico-ocupacionales, los relatos de los trabajadores y las condiciones observadas.

Conforme al artículo 20 de la Resolución 1843 de 2025, esta información se consigna diferenciando la exposición por área de trabajo, proceso productivo u oficio específico, lo cual permite una caracterización más precisa del riesgo y sirve para orientar adecuadamente las acciones de vigilancia en salud.

Exposición Laboral
#GRAFICA# #Gráfica de Tablas llamada 'Tipo de riesgo' con un conteo de registros únicos de identificación donde incluya en xlabel  de forma obligatoria la categoria de cargos laborales y con una leyenda donde riesgo_ergonomico=Si o  riesgo_quimico=Si o riesgo_psicosocial=Si o riesgo_biomecanico=Si#

+Redacta dos párrafos de aproximadamente 60 palabras cada uno, analizando los tipos de riesgos de los trabajadores por cada oficio o cargo y su relación con determinantes sociales de la salud, barreras de acceso y patrones de vulnerabilidad, con base en: ||DATOS realiza el cálculo del conteo único de personas en identificación agrupado por cargo y donde riesgo_ergonomico=Si, el mismo conteo único y columna cargo pero con filtro riesgo_quimico=Si, el mismo conteo único y misma columna x pero filtrando riesgo_psicosocial=Si y por ultimo mismo conteo único por cargo u oficio con filtro riesgo_biomecanico=Si|| +"""
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


# out = graficos_gpt5(df_otro,textico)
# print(out)
# import json
# config_list = json.loads(out)
# params = config_list[0]["params"]
# fig, ax = plot_from_params(df_otro,params)
# out = _fig_to_data_uri(fig)
# #print(out)


# with open(r"C:\Users\EstebanEscuderoPuert\Downloads\output_.txt", "w", encoding="utf-8") as f:
#     f.write(out)

variable = """{
      "chart_type": "barras_horizontal",
      "function_name": "graficar_barras_horizontal",
      "title": "Personas por grupos etarios",
      "xlabel": "grupo_edad",
      "y": "documento",
      "agg": "distinct_count",
      "distinct_on": "documento",
      "drop_dupes_before_sum": false,
      "unique_by": null,
      "conditions_all": [],
      "conditions_any": [],
      "binning": {
        "column": "edad",
        "bins": [
          "-inf",
          6,
          12,
          19,
          27,
          60,
          "+inf"
        ],
        "labels": [
          "0-5",
          "6-11",
          "12-18",
          "19-26",
          "27-59",
          "60+"
        ],
        "output_col": "grupo_edad"
      },
      "stack_columns": null,
      "color": null,
      "colors_by_category": null,
      "legend_col": "genero",
      "show_legend": true,
      "show_values": null,
      "sort": null,
      "limit_categories": null,
      "needs_disambiguation": false,
      "candidates": {
        "xlabel": [],
        "y": []
      },
      "porcentage_of": null,
      "percentage_colname": null
    }
 """

variable2 = [{
  "chart_type": "barras_horizontal",
  "function_name": "graficar_barras_horizontal",
  "title": "Tipo de riesgo",
  "xlabel": "tipo_riesgo",
  "y": "documento",
  "agg": "distinct_count",
  "distinct_on": "documento",

  "drop_dupes_before_sum": False,
  "unique_by": None,
  "conditions_all": [],
  "conditions_any": [],
  "binning": None,
  "stack_columns": {
    "columns": ["riesgo_ergonomico", "riesgo_quimico", "riesgo_psicosocial", "riesgo_biomecanico"],
    "output_col": "tipo_riesgo",
    "value_col": "valor",          # 👈 mejor que None
    "keep_value": "si",
    "label_map": None
  },
  "color": None,
  "colors_by_category": None,      # o un dict si quieres colores por sexo
  "show_legend": True,
  "legend_col": "genero",
  "show_values": False,
  "sort": None,
  "limit_categories": None,
  "needs_disambiguation": False,
  "candidates": {
    "xlabel": [],
    "y": []
  }
}
]


# # out = _fig_to_data_uri(out)
# # print(out)
# # for item in out:
# #     if isinstance(item, dict) and "params" in item:
# #         idx, params, span = item["idx"], item["params"], item["span"]
# #         params_list.append(params)
# #         try:
# #         fig, ax = plot_from_params(df, params)
# #print(texto_completo)
# #out = process_ia_blocks(texto_completo,ask_gpt5)
# #print(out)
# #out = ask_gpt5(texto_completo)
# #print(out)
# #df_final, res = procesar_df_una_fila(
# #    df,
# #    col_output="Output",
# #    ctx=ctx,
# #    df_datos=df_datos,
# #    ask_fn=ask_gpt5,          # si necesitas IA
# #    export_html=True,
# #    titulo_html="Informe generado"
# #)



# TEXTO = """*El informe de condiciones de salud de la población trabajadora es un instrumento documental que consolida la información resultante del análisis de un conjunto de variables objetivas y subjetivas relacionadas con el estado de salud de los trabajadores. Su propósito principal es servir como herramienta evaluativa para el seguimiento de los objetivos establecidos por la organización, permitiendo identificar necesidades en materia de salud y bienestar, con el fin de proponer acciones correctivas, preventivas y de mejora, a través de programas de medicina preventiva y del trabajo. *
# *COLMÉDICOS S.A.S., como institución prestadora de servicios de salud en Seguridad y Salud en el Trabajo, y en cumplimiento del marco normativo legal vigente, en lo particular lo dispuesto en el artículo 20 de la Resolución 1843 de 2025, presenta el siguiente informe. *
# *Este informe le permitirá al empleador dar cumplimiento a las actividades contempladas en los programas de Medicina Preventiva y del Trabajo, cuyo objetivo es la promoción, prevención y control del estado de salud de los trabajadores, protegiéndolos frente a los factores de riesgo ocupacionales, favoreciendo así el mantenimiento de su capacidad productiva y el bienestar en el entorno laboral. *
# *Este informe se elabora en cumplimiento de las disposiciones legales y reglamentarias vigentes en Colombia que rigen la vigilancia de la salud de los trabajadores y la prevención de enfermedades laborales, en especial: *
# *Ley 9 de 1979: Por la cual se dictan medidas sanitarias y se establecen normas para la protección de la salud de los trabajadores. *
# *Resolución 1843 de 2025 del Ministerio de Trabajo: Por la cual se regula la práctica de evaluaciones médicas ocupacionales y se dictan otras disposiciones. *
# *Resolución 0312 de 2019: Por la cual se definen los estándares mínimos del Sistema de Gestión de la Seguridad y Salud en el Trabajo (SG-SST). *
# *Ley 1562 de 2012: Reforma del Sistema General de Riesgos Laborales, que establece directrices para la promoción y prevención en salud ocupacional. *
# *Decreto 1072 de 2015 (Decreto Único Reglamentario del Sector Trabajo): Compila la normatividad vigente del SG-SST. *
# *Normas del Sistema General de Riesgos Laborales (SGRL) y demás disposiciones que regulan la vigilancia de la salud individual y colectiva en el entorno laboral. *
# *Estas normativas fundamentan la práctica de los exámenes médicos ocupacionales realizados y orientan la interpretación de los resultados aquí consignados, contribuyendo a la prevención de riesgos y a la promoción de ambientes de trabajo seguros y saludables. *

# [Evaluar el estado de salud de los trabajadores de la empresa {{nombre_cliente}}, a través de la realización de exámenes médicos ocupacionales, con el fin de determinar el perfil sociodemográfico, identificar posibles riesgos para la salud relacionados con sus funciones y generar recomendaciones que contribuyan a la promoción de ambientes de trabajo seguros y saludables.]
# *Identificar condiciones de salud que permitan orientar acciones de promoción de la salud en el entorno laboral. *
# *Detectar factores de riesgo individuales que puedan prevenirse o mitigarse mediante intervenciones grupales oportunas. *
# *Fomentar el autocuidado y los estilos de vida saludables a través del análisis de hallazgos médicos ocupacionales. *
# *Aportar información útil para el diseño e implementación de programas preventivos en el marco del Sistema de Gestión de Seguridad y Salud en el Trabajo (SG-SST). *
# *Apoyar la vigilancia epidemiológica en salud ocupacional mediante el seguimiento de indicadores de salud en la población trabajadora. *
# [Se realiza una caracterización de la empresa {{nombre_cliente}} identificada con NIT {{nit_cliente}}, cuya actividad económica corresponde a comercializar y distribuir productos avicolas. ]
# [Se lleva a cabo un estudio de tipo descriptivo, de corte transversal, basado en la valoración médica ocupacional realizada a trabajadores adscritos a la empresa {{nombre_cliente}} en el periodo comprendido entre el {{fecha_inicio}} y el {{fecha_fin}}. La muestra corresponde a {{numero_personas}} personas, el 66% de la población total de la empresa, la cual está conformada por un total de 150 empleados. ]
# [Las valoraciones médico ocupacionales realizadas en la empresa  {{nombre_cliente}} durante el periodo comprendido entre el {{fecha_inicio}} y el {{fecha_fin}}, fueron realizadas por médicos especialistas en salud ocupacional, seguridad y salud en el trabajo o medicina laboral todos con licencia vigente en seguridad y salud en el trabajo, en cumplimiento de la normatividad legal vigente. ]
# *El registro clínico de cada evaluación se efectuó a través del sistema SOFIA, plataforma especializada que dispone de un formato de historia clínica sistematizada con enfoque ocupacional. Esta herramienta garantiza la integridad, trazabilidad y custodia digital de la información, en cumplimiento de las disposiciones nacionales sobre el manejo de historias clínicas y la protección de datos personales. *
# *Durante el proceso de evaluación se recolectó información cualitativa mediante entrevista clínica estructurada, propia de la historia clínica ocupacional, e información cuantitativa derivada del examen físico. Para este último, se utilizaron equipos biomédicos estandarizados, garantizando la confiabilidad de las mediciones mediante la verificación de sus hojas de vida, en las cuales consta el cumplimiento de los cronogramas de calibración y mantenimiento preventivo. *
# *Posteriormente, se realizó una revisión sistemática de la totalidad de la información registrada, la cual fue organizada y analizada en una base de datos integrada. El procesamiento estadístico se llevó a cabo mediante la plataforma analítica de SOFIA. Para las variables cualitativas se emplearon distribuciones de frecuencias absolutas y relativas, mientras que para las variables cuantitativas se calcularon medidas de tendencia central (promedio, mediana) y de dispersión (rango, desviación estándar), según correspondiera. *
# [Se toma como fuente de información la matriz construida a partir de los datos extraídos de las historias clínicas ejecutadas. En total, fueron evaluados {{totales}} trabajadores, que asistieron para el examen médico ocupacional periodico durante el periodo comprendido entre el {{fecha_inicio}} y el {{fecha_fin}}. ]
# *Los resultados se desglosan en diferentes secciones: perfil sociodemográfico, perfil de hábitos saludables, perfil laboral y perfil de morbilidad. *
# *Es importante aclarar que, como anexo a este documento, se entrega la base de datos en formato Excel, que permitirá a la empresa filtrar la información según sus necesidades para una mejor gestión de los riesgos. Esta base de datos no contiene información personal que permita individualizar a los trabajadores, dando así cumplimiento a la normativa colombiana sobre confidencialidad de la información y el articulo 20 de la Resolución 1843 de 2025. *
# *El perfil sociodemográfico corresponde al conjunto de características básicas de la población trabajadora evaluada, tales como grupo etario, sexo, estado civil, nivel educativo y estrato socioeconómico. Esta información se obtiene durante la valoración médica ocupacional y permite contextualizar las condiciones de salud identificadas, así como orientar la toma de decisiones en materia de promoción, prevención y control de riesgos. *
# *Analizar estas variables sociodemográficas resulta clave para comprender las necesidades específicas de los trabajadores, segmentar intervenciones en salud y fortalecer la planeación de los programas de Medicina preventiva y del trabajo, enmarcados en el Sistema de gestión de Seguridad y Salud en el trabajo (SG-SST). *
# *La representación gráfica mediante una pirámide poblacional permite identificar tendencias demográficas de interés, como la concentración de trabajadores en determinados grupos etarios, la proporción por género y posibles diferencias en la composición de la población laboral. Esta información es valiosa para orientar estrategias de salud ocupacional acordes con las características de la fuerza laboral, tales como el diseño de programas específicos para trabajadores jóvenes, adultos o personas mayores, diferenciando además según el sexo. *
# *Los grupos etarios según la OMS (Organizacion mundial de la salud), clasifican a las personas según su edad y etapa del ciclo vital, permiten adaptar las medidas de prevención y promoción de la salud a las necesidades específicas de cada etapa de la vida. *
# *En el ámbito ocupacional esta clasificación más detallada determina los requerimientos de cada etapa de la vida, así se pueden determinar actividades de PyP dentro de las empresas, enfocadas a los riesgos y particularidades de cada grupo. *
# *En Colombia, conforme el Ministerio de salud y protección social, se clasifican los grupos etarios de la siguiente forma: *
# *Primera infancia 0 a 5 años*
# *Infancia 6 a 11 años*
# *Adolescencia 12 a 18 años*
# *Juventud 19 a 26 años*
# *Adultez: 27 a 59 años*
# *Adulto: 27 a 44 años*
# *Adulto maduro: 45 a 59 años*
# *Personas mayores: 60 años y más. *
# #GRAFICA#
# #grafico de barras horizontales llamada 'Personas por grupos etarios' que utilice la columna de edad en la columna X para la siguiente agrupación por rangos de edad: (total_0_y5_5, total_6_y_11, total_27_y_59,  total_12_y_18, total_19_y_26, total_mayores_60) y la columna de identificación para hacer un conteo de personas únicas#
# *Gráfica 1. Piramide poblacional, distribución de los trabajadores evaluados según sexo y grupo etario. *
# +Analisis de aproximadamente 3 parrafos con 60 palabras cada uno donde habla de información sobre la clasificación de grupos Etarios, especificamente de información asociada al análisis de las siguientes variables: ||DATOS|| es ||realiza el conteo del número de personas por rangos de edad: (total_0_y5_5, total_6_y_11, total_27_y_59,  total_12_y_18, total_19_y_26, total_mayores_60)|| +
# *El objetivo de la composición familiar es caracterizar el entorno familiar que puede influir en las condiciones de salud, bienestar psicosocial y necesidades individuales en el entorno laboral. *
# *Con esta información se puede identificar, por ejemplo, si la mayoría de los trabajadores tiene responsabilidades familiares directas, lo que puede estar asociado con mayores niveles de carga mental o estrés. A su vez, conocer el estado civil ayuda a dimensionar el nivel de apoyo social o red de contención del trabajador. *
# *El análisis de composición familiar permite a la empresa orientar acciones de promoción de la salud, conciliación vida-trabajo y beneficios dirigidos a subgrupos con cargas familiares específicas. *
# #GRAFICA#
# #grafico de barras llamada 'conteo de personas por rango de hijos' que utilice la columna de numero de hijos para clasificar entre Entre_0_y_1_hijos,  Entre_1_y 2_hijos,  3_mas_hijos en la columna X y cuente el número de registros por cada segmento utilizando la columna de identificación#
# *Gráfica 2. Distribución de la población trabajadora conforme estado civil y numero de hijos. *
# +Analisis de aproximadamente 3 parrafos con 60 palabras cada uno donde habla de información sobre la composición familiar de acuerdo a datos estadísticos especificos, especificamente de información asociada al análisis de las siguientes variables: ||DATOS|| es ||Realiza el conteo del número de personas de acuerdo con el rango del número de hijos: (Entre_0_y_1_hijos,  Entre_1_y 2_hijos,  3_mas_hijos)||+
# *La identificación del estrato socioeconómico de los trabajadores evaluados permite contextualizar las condiciones sociales en las que se desarrollan sus actividades diarias, facilitando una comprensión integral de los determinantes sociales de la salud. *
# *Esta información contribuye a orientar estrategias de promoción y prevención más pertinentes, alineadas con las necesidades reales de la población trabajadora, fortaleciendo así la toma de decisiones en salud ocupacional y el diseño de intervenciones más efectivas y equitativas. *
# #GRAFICA#
# #tabla llamada 'estrato socieconómico de la población evaluada' que utilice la columna de estrato socieconómico columna X y cuente el número de registros utilizando la columna identificación#
# [Tabla 1. Distribución absoluta y porcentual de la distribución del estrato de los candidatos evaluados en el periodo del {{fecha_inicio}} y el {{fecha_fin}}. ]
# +Dos parrafos de aproximadamente 60 palabras cada uno explicando información relevante sobre la distribución de la población de la base de datos por estrato socioeconómico, la información lahace analizando datos de las siguientes vriables: ||DATOS|| es ||Realiza el cálculo del conteo de personas por estrato socio económico: (personas_estrato_1 ,  personas_estrato_2 , personas_estrato_3,  personas_estrato_4,  personas_estrato_5,  personas_estrato_6}||+
# *El nivel de escolaridad permite establecer el grado de formación académica alcanzado por los trabajadores, lo cual influye en su capacidad de comprensión de las recomendaciones médicas, normas de seguridad y programas de capacitación en el entorno laboral. Esta variable es relevante en salud ocupacional, ya que se asocia con el autocuidado, la adherencia a tratamientos y la apropiación de conductas saludables en el ámbito laboral y personal. *
# #GRAFICA#
# #grafica de barras horizontal llamada 'escolaridad' que utilice la columna de nivel escolaridad en la columna descriptiva y realice un conteo del número de registros utilizando la columna de identificación#
# *Gráfica 3. Distribución porcentual del nivel de escolaridad de los candidatos evaluados en la valoración medico ocupacional.*
# +Análisis de dos parrafos con aproximadamente 30 palabras cada uno, sobre el número de personas clasificado por el nivel de escolaridad donde utilices los siguientes datos: ||DATOS|| || realiza un conteo del número de personas por nivel de escolaridad|| +
# *En este apartado se evaluaron tres componentes principales del estilo de vida de los trabajadores: la práctica regular de actividad física, el consumo de bebidas alcohólicas y el consumo de cigarrillo. Estos factores permiten identificar comportamientos que influyen directamente en la salud general y el riesgo de desarrollar enfermedades crónicas no transmisibles. *

# *El análisis de estos hábitos es fundamental, ya que la adopción de estilos de vida saludables como la actividad física frecuente y la reducción del consumo de sustancias nocivas contribuye significativamente al bienestar fisico y mental, mejora la productividad laboral y disminuye el ausentismo por causas médicas. Por tanto, conocer estos datos permite orientar acciones de promoción de la salud en el entorno laboral. *

# #GRAFICA#
# #Tabla llamada 'Porcentaje de habitos' donde se evalua en el eje x el tipo de hábito, y en columna y el porcentaje de si sí practica o si no practica#
# *Tabla 2. Distribución porcentual de la población evaluada de habitos de vida saludable. *

# *Se considera que una persona tiene hábitos de vida saludable cuando realiza actividad física al menos tres veces por semana, con una duración mínima de 30 minutos por sesión. El consumo habitual de licor se define como la ingesta regular, al menos una vez por semana. Por su parte, se considera fumador habitual a quien consume uno o más cigarrillos al día de forma constante. *

# *Se entiende por antecedentes de exposición laboral a factores de riesgo ocupacionales el historial de contacto, durante actividades laborales actuales o anteriores, con agentes o condiciones que pueden representar un riesgo para la salud física o mental del trabajador. Estos factores pueden ser de origen físico, químico, biológico, ergonómico, psicosocial o mecánico, y están asociados a las tareas, procesos, herramientas o ambientes en los que se ha desempeñado el trabajador. *

# *Durante las valoraciones médicas realizadas, esta información se recolecta mediante la entrevista clínica y el análisis de la historia laboral del trabajador, permitiendo identificar posibles efectos acumulativos o asociaciones entre dichas exposiciones y el estado actual de salud. Este análisis resulta clave para orientar acciones de vigilancia médica, prevenir enfermedades laborales y promover condiciones de trabajo más seguras. *

# #GRAFICA#
# #Grafica de barras llamada 'Tipo de riesgo' con un conteo de registros únicos de identificación que tengan el valor de si para las columnas de riesgo ergonómico, riesgo quimico, riesgo psicosocial, riesgo biomecanico#
# *Tabla 3. Distribución absoluta de tipos de riesgo ocupacional. *

# +Esquematiza un parrafo de aproximadamente 60 palabras con un análisis sobre antecedentes de exposición laboral a factores de reisgo ocupacionales, donde tomes información asociada a ||DATOS|| ||conteo del número de personas únicas con la identificación que hayan dado como valor si en cada columna de (ergonomico, quimico, psicosocia, biomecanico)|| +
# *La exposición laboral actual hace referencia a la presencia de factores de riesgo ocupacionales a los cuales se encuentran expuestos los trabajadores en el desarrollo de sus funciones, de manera continua o intermitente, en su entorno laboral actual. Estos factores pueden ser de origen físico, químico, biológico, ergonómico, psicosocial, mecánico, etc, y su identificación se realiza con base en la información recolectada durante las valoraciones médico-ocupacionales, los relatos de los trabajadores y las condiciones observadas. *

# *Conforme al artículo 20 de la Resolución 1843 de 2025, esta información se consigna diferenciando la exposición por área de trabajo, proceso productivo u oficio específico, lo cual permite una caracterización más precisa del riesgo y sirve para orientar adecuadamente las acciones de vigilancia en salud. *

# #GRAFICA#
# #Gráfica de barras horizontales llamada 'exposición laboral actual' con columna x agrupando el área u oficio y en el eje y el conteo del número de personas con identificación única con una leyenda de datos por tipo de riesgo (biológico, químico, ergonómico, psicosocial)#

# +Dos parrafos de aproximadamente 60 palabras en la cual se haga un análisis de la información asociada a riesgos psicosociales y ergonómicos, a partir de la información: ||DATOS|| ||Conteo del número de personas con identificación única clasificadas por tipos de riesgos de las columnas (biológico, químico, ergonómico, psicosocial) y por área u oficio|| +
# *En esta sección se registran los diagnósticos y condiciones de salud que han sido originados o agravados por la exposición a factores de riesgo presentes en el entorno laboral, ya sea en el cargo actual o en actividades desempeñadas con anterioridad. Estos antecedentes incluyen enfermedades laborales reconocidas y secuelas derivadas de accidentes de trabajo. *

# *La identificación de estos antecedentes se realiza a través de la revisión de la historia clínica ocupacional, la entrevista médica y el análisis de la trayectoria laboral del trabajador. Esta información permite establecer relaciones entre las condiciones de salud y las exposiciones ocupacionales, facilitando la implementación de medidas de prevención, control y vigilancia médica, así como el cumplimiento de lo establecido en la normativa nacional en materia de salud y seguridad en el trabajo. *

# +Realiza un análisis de 40 palabras donde analice con un porcentaje las personas que hayan manifestado desarrollar una patología asociada a su labor toma la información de ||DATOS|| ||conteo de número de personas que dicen haber tenido una patologia asociada a su labor y el número de personas que no||+

# *La sintomatología reportada corresponde a los signos y síntomas manifestados verbalmente por los trabajadores durante la valoración médica ocupacional, los cuales no siempre constituyen diagnósticos clínicos confirmados, pero si representan señales de alerta sobre posibles alteraciones en la salud física o mental. *

# #GRAFICA#
# #Grafica de tabla con el nombre de 'Reporte de sintomatología' que realice el conteo de personas de si reporta sintomatología o no#
# *Tabla 4. distribución absoluta y porcentual de presencia de sintomatología reportada por los trabajadores . *

# +Análisis de aproximadamente 60 palabras en la cual se haga un análisis del conteo de personas con sintomatología y las  que no con base a la siguiente información: ||DATOS|| ||conteo de personas con sintomatología y las  que no||+

# *La identificación y análisis de los sintomas referidos por los trabajadores durante la valoración medico-ocupacional constituye una herramienta clave para la vigilancia de la salud colectiva dentro de una organización, ya que permite: *
# •	*Detectar alertas tempranas de afectación en la salud laboral*
# •	*Identificar tendencias o patrones en la población trabajadora*
# •	*Relacionar condiciones de salud con el entorno laboral*
# •	*Cumplir con los requisitos de SG-SST y la normatividad vigente. *

# #GRAFICA# #Grafica de tabla llamada 'Clasificación trabajadores por sistema afectado' en la cual se tomará la columna x la columna de tipo de sistema afectado y el conteo de número de trabajadores diferentes de acuerdo con la identificación#

# *Tabla 4. Distribución porcentual y absoluta de trabajadores evaluados conforme sintomatologia reportada por sistema afectado. *

# +Análisis de aproximadamente 2 parrafos de 35 palabras cada uno donde se hable sobre tipo de sistema afectado sintomatologicamente de acuerdo con los examenes médicos acorde con el conteo de número de trabajadores diferentes de acuerdo con la identificación, toma los datos de: ||DATOS|| ||conteo de número de trabajadores por tipo de sistema efectado|| +

# *El índice de masa corporal (IMC) es un párametro antropométrico que relaciona el peso y la talla de una persona, y que se utiliza para estimar el estado nutricional de los trabajadores. Este indicador permite clasificar a las personas en rangos que orientan la detección temprana de riesgos asociados tanto al déficit como al exceso de peso.*

# *Clasificación de IMC según la organización mundial de la salud (OMS): *
# •	*Bajo peso menor que 18.5 kg/m²*
# •	*Peso normal 18.5 a 24.9 kg/m²*
# •	*Sobrepeso: 25 a 29.9 kg/m²*
# •	*Obesidad grado I: 30 a 34.9 kg/m²*
# •	*Obesidad grado II: 35 a 39.9 kg/m²*
# •	*Obesidad grado III:  mayor que 40 kg/m²*

# *En el ámbito de la salud ocupacional, el IMC es un indicador clave para detectar riesgos metabólicos y cardiovasculares que puedan afectar la capacidad laboral, identificar condiciones asociadas a sobrepeso u obesidad que incrementen la probabilidad de lesiones osteomusculares, reconocer casos de bajo peso que puedan relacionarse con desnutrición, menor resistencia física o disminución de la capacidad inmunológica. *

# #GRAFICA##Grafica de barras con el nombre de 'indice de masa corporal' con eje x para la clasificación de (Bajo peso: IMC menor que 18.5 kg/m², Peso normal: 18.5 menor o igual que IMC y este último menor o igual a 24.9 kg/m², Sobrepeso: 25.0 menor o igual que IMC y este último menor o igual a 29.9 kg/m², Obesidad grado I: 30.0 menor o igual que IMC y este último menor o igual a 34.9 kg/m², Obesidad grado II: 35.0 menor o igual que IMC y este último menor o igual que 39.9 kg/m², Obesidad grado III: IMC maoyr que 40.0 kg/m²) y eje x para el conteo del número de personas por número de identificación diferente#
# *Gráfica 5. Distribución de índice de masa corporal de la pobación evaluada. *

# +Analisis de dos parrados con 35 palabras cada uno aproximadamente donde se analice las cifras de la distribución del peso de las personas de acuerdo con las siguientes cifras: ||DATOS|| ||Conteo del número de personas diferentes es decir por número de identificación, distribuido por su indice de masa corporal agrupado por la siguiente tabla: (Bajo peso: IMC menor que 18.5 kg/m², Peso normal: 18.5 menor o igual que IMC y este último menor o igual a 24.9 kg/m², Sobrepeso: 25.0 menor o igual que IMC y este último menor o igual a 29.9 kg/m², Obesidad grado I: 30.0 menor o igual que IMC y este último menor o igual a 34.9 kg/m², Obesidad grado II: 35.0 menor o igual que IMC y este último menor o igual que 39.9 kg/m², Obesidad grado III: IMC maoyr que 40.0 kg/m²) || +

# *En el marco de las valoraciones médico-ocupacionales realizadas, se llevó a cabo un tamizaje para riesgo cardiovascular como parte de la vigilancia en salud colectiva de los trabajadores . Esta evaluación tiene como objetivo identificar de forma temprana la presencia de factores que puedan aumentar la probabilidad de desarrollar enfermedades cardiovasculares, las cuales presentan una de las principales causas de morbilidad y ausentismo laboral. *

# *Este tamizaje se realizó a partir del analisis de parámetros clínicos básicos y fácilmente accesibles como índice de masa corporal (IMC), tensión arterial y perímetro abdominal. Estos permiten estimar el riesgo cardiovascular individual y colectivo. Por ejemplo, un IMC igual o superior a 25, cifras tensionales elevadas ( mayor que 130/80mmHg) o perímetros abdominales por encima de los valores recomendados (mayor que 102cm en hombres y  mayor que 88cm en mujeres) son señales de alerta sobre posibles alteraciones metabólicas o cardiovasculares, especialmente si están presentes de manera combinada. Por lo que se clasifico asi: *
# *Verde, Bajo riesgo, Ningún factor presente (IMC menor que  25, PA normal, perímetro normal)*
# *Amaraillo, Riesgo moderado, 1 factor presente (ej. solo IMC elevado o solo PA elevado)*
# *Rojo, Alto riesgo, 2 o más factores presentes*
# *Tabla 5. Clasificación del nivel de riesgo cardiovascular para la población evaluada. *

# *La identificación de estos factores de riesgo resulta clave para orientar estrategias dentro del Programa de medicina preventiva y del trabajo, que promuevan estilos de vida saudables, fomenten el control médico oportuno y disminuyan el impacto de las enfermedades crónicas no transmisibles en el entorno laboral. Además, esta información constituye un insumo relevante para el diseño de intervenciones colectivas en promoción de la salud y prevención de enfermedades en el lugar de trabajo. *

# #GRAFICO##Grafico de tortas llamado 'Riesgo cardiovascular' donde se cuenta el número de personas registros únicos de identificación de acuerdo a su clasificación de acuerdo con la columna de riesgo cardiovascular#
# *Gráfica 6. Distribución de riesgo cardiovascular conforme tamizaje, en la poblacion evaluada. *

# +Análisis de aproximadamente un parrafo de 70 palabras acerca del número de personas clasificados por los diferentes riesgos cardiovasculares de los siguientes datos: ||DATOS|| ||Conteo del número de personas con registro único de identificación por cada clasificación de riesgo cardiovascular de la columna de riesgo cardiovascular|| +

# *En el marco del análisis de condiciones de salud de la población trabajadora, se utilizó la clasificación establecida por la Guía de atención integral de salud ocupacional basada en la evidencia para desórdenes musculo-esqueléticos (GATISO-DME, 2007), la cual permite identificar y clasificar a los trabajadores con posibles alteraciones musculoesqueléticas asociadas a factores de riesgo ocupacionales. *

# *Esta clasificación agrupa a los trabajadores en tres categorías: *

# •	*No caso: Trabajadores que no presentan síntomas ni hallazgos clínicos compatibles con desórdenes musculoesqueléticos en el momento de la evaluación. *
# •	*Sintomático: Trabajadores que refieren molestias musculoesqueléticas (como dolor, rigidez o fatiga muscular), pero sin evidencia clínica o funcional suficiente para confirmar un diagnostico ocupacional. *
# •	*Caso confirmado: Trabajadores que presentan síntomas persistentes acompañados de hallazgos físicos, antecedentes y pruebas clínicas. *

# *Esta clasificación constituye una herramienta fundamental para el seguimiento en salud ocupacional, ya que permite identificar necesidades de intervención, priorizar casos para vigilancia médica o ergonómica, y orientar estrategias de prevención y control en los diferentes procesos o áreas de trabajo. *

# #GRAFICA#
# #Grafica de barras llamada 'clasificación osteomuscular' donde se busca en la columna x tomar la clasificación y en la columna y un conteo del número único de personas por identificación#

# *Gráfica 7.  Clasificación osteomuscular de la población evaluada . *

# +Análisis de aproximadamente 2 parrafos de 65 palabras cada uno en la cual se pueda analizar en porcentajes la distribución de personas con sintomas osteomusculaes de acuerdo con: ||DATOS|| ||Conteo del número de personas con identificación unica de acuerdo con la columna de tipo de Clas_osteomuscular|| +

# *El proposito de este examen es realizar una evaluación tamiz de la capacidad visual del trabajador y con ello detectar alteraciones importantes, no solo para la vida cotidiana del trabajador, sino que ademas puedan genrar un riesgo para las labores que desempeña. *

# *Este examen permite evidenciar la presencia de ceguera temporal o permanente, asi como la existencia de alteraciones de la agudeza visual por defectos refractivos, alteracion en la percepción del color y de la profundidad. En terminos generales, se recomienda que no solo se tenga en cuenta el diagnostico de visiometría, ya que es un examen tamiz que detecta alteraciones pero no genera diagnosticos precisos. *

# *En las labores donde exista el riesgo de accidentalidad por un cuerpo extraño en los ojos, el optómetra puede recomendar si el trabajador requiere el uso permanente de gafas de seguridad, las cuales deben tener la formula refractiva requerida que corrija su defecto visual para mayor seguridad y confort en la tarea a realizar (Ministerio de salud y protección social, 2017) *

# #GRAFICA#
# #Gráfica de tabla con el nombre 'visiometría' con el conteo único de personas por identificación con la clasificación de xlabel de la columna de visiometría#

# *Tabla 6. Distribución absoluta y porcentual de conceptos de visiometrías realizadas. *


# #GRAFICA#
# #Gráfica de barras horizontales con el nombre 'visiometría' con el conteo único de personas por identificación con la clasificación de xlabel de la columna de visiometría#

# *Gráfica 8. Distribución de los conceptos de las visiometrias realizadas a la población evaluada .*

# +Análisis de 2 parrafos con aproximadamente 70 palabras donde se hable sobre: ||DATOS|| ||Conteo de registros únicos por identificación  con la clasificación de xlabel de la columna de visiometría||+

# *Este examen evalua la función del organo de la vision. Perrmite detectar alteraciones de la refracción del ojo (hipermetropia, miopia, astigmatismo y presbicia), alteracines de la acomodación, desbalances oculo-motores, alteraciones de la percepcion de colores y profundidad y patologías del segmento anterior y posterior del ojo, los cuales pueden ser producidos o no por factores de riesgo ocupacional (A. Arias, 2016).*

# #GRAFICA#
# #Gráfica de tabla con el nombre 'Optometría' con el conteo único de personas por identificación con la clasificación de xlabel de la columna de optometria#

# *Tabla 6. Distribución absoluta y porcentual de conceptos de optometria realizadas. *


# #GRAFICA##Gráfica de barras con el nombre 'Optometría' con el conteo único de personas por identificación con la clasificación de xlabel de la columna de optometria#
# *Gráfica 8. Distribución de los conceptos de las optometria realizadas a la población evaluada .*

# +Análisis de 2 parrafos con aproximadamente 70 palabras donde se hable sobre: ||DATOS|| ||Conteo de registros únicos por identificación  con la clasificación de xlabel de la columna de optometria||+

# *Este examen permite establecer un concepto preliminar sobre el estado de la audición, clasificandolo como normal o con indicios de alteración. Cabe aclarar que una alteración identificada en esta prueba no constituye un diagnostico clinico definitivo, por lo que, en caso de resultados anormales, se recomienda la realización de una audiometria clinica para determinar con precisión el tipo y grado de perdida auditiva del evaluado. *

# *Para asegurar la confiabilidad del resultado, se recomienda un reposo auditivo minimo de 12 horas antes dela prueba. *

# *La evaluación es realizada por profesionales en fonoaudiología, cuya interpretación permite desartar o detectar alteraciones auditivas de manera oportuna. Este examen es partivularmente importante en trabajadores expuestos a niveles de ruido a 85dB, y contituye un requisito obligatorio para la certificación en trabajo en alturas, debido a la importancia de una audición funcional adeciada para garantizar la seguridad en este tipo de labores. *

# #GRAFICA#
# #Gráfica de tabla con el nombre 'Audiometria' con el conteo único de personas por identificación con la clasificación de xlabel de la columna de Audiometria#

# *Tabla 6. Distribución absoluta y porcentual de conceptos de Audiometria realizadas. *


# #GRAFICA#
# #Gráfica de barras con el nombre 'Audiometria' con el conteo único de personas por identificación con la clasificación de xlabel de la columna de Audiometria#

# *Gráfica 8. Distribución de los conceptos de las Audiometria realizadas a la población evaluada .*

# +Análisis de 2 parrafos con aproximadamente 70 palabras donde se hable sobre: ||DATOS|| ||Conteo de registros únicos por identificación  con la clasificación de xlabel de la columna de Audiometria||+

# *La espirometria es un examen tamiz que evalua la función pulmonar, registra el maximo volumen de aire que puede movilizar una persona desde una inspiración maxima hasta una exhalación completa. Su objetivo es detectar alteraciones en la capacidad respiratoria  del trabajador con la finalidad de definir la exposición de un trabajador en ambientes con factores de riesgo (material particulado, vapores de sustancias quimicas, ambientes con alto nivel de polvo, etc.), sin causarle una enfermedad o agravar una ya existente. *

# #GRAFICA#
# #Gráfica de tabla con el nombre 'Espirometria' con el conteo único de personas por identificación con la clasificación de xlabel de la columna de Espirometria#
# *Tabla 6. Distribución absoluta y porcentual de conceptos de Espirometria realizadas. *


# #GRAFICA#
# #Gráfica de barras con el nombre 'Espirometria' con el conteo único de personas por identificación con la clasificación de xlabel de la columna de Espirometria#
# *Gráfica 8. Distribución de los conceptos de las Espirometria realizadas a la población evaluada .*

# +Análisis de 2 parrafos con aproximadamente 70 palabras donde se hable sobre: ||DATOS|| ||Conteo de registros únicos por identificación  con la clasificación de xlabel de la columna de Espirometria||+"""


# texto_plano = """[Evaluar el estado de salud de los trabajadores de la empresa {{nombre_cliente}}, a través de la realización de exámenes médicos ocupacionales, con el fin de determinar el perfil sociodemográfico, identificar posibles riesgos para la salud relacionados con sus funciones y generar recomendaciones que contribuyan a la promoción de ambientes de trabajo seguros y saludables.]
# *Identificar condiciones de salud que permitan orientar acciones de promoción de la salud en el entorno laboral. *
# *Detectar factores de riesgo individuales que puedan prevenirse o mitigarse mediante intervenciones grupales oportunas. *
# *Fomentar el autocuidado y los estilos de vida saludables a través del análisis de hallazgos médicos ocupacionales. *
# *Aportar información útil para el diseño e implementación de programas preventivos en el marco del Sistema de Gestión de Seguridad y Salud en el Trabajo (SG-SST). *
# *Apoyar la vigilancia epidemiológica en salud ocupacional mediante el seguimiento de indicadores de salud en la población trabajadora. *
# [Se realiza una caracterización de la empresa {{nombre_cliente}} identificada con NIT {{nit_cliente}}, cuya actividad económica corresponde a comercializar y distribuir productos avicolas. ]
# [Se lleva a cabo un estudio de tipo descriptivo, de corte transversal, basado en la valoración médica ocupacional realizada a trabajadores adscritos a la empresa {{nombre_cliente}} en el periodo comprendido entre el {{fecha_inicio}} y el {{fecha_fin}}. La muestra corresponde a {{numero_personas}} personas, el 66% de la población total de la empresa, la cual está conformada por un total de 150 empleados. ]
# [Las valoraciones médico ocupacionales realizadas en la empresa  {{nombre_cliente}} durante el periodo comprendido entre el {{fecha_inicio}} y el {{fecha_fin}}, fueron realizadas por médicos especialistas en salud ocupacional, seguridad y salud en el trabajo o medicina laboral todos con licencia vigente en seguridad y salud en el trabajo, en cumplimiento de la normatividad legal vigente. ]
# *El registro clínico de cada evaluación se efectuó a través del sistema SOFIA, plataforma especializada que dispone de un formato de historia clínica sistematizada con enfoque ocupacional. Esta herramienta garantiza la integridad, trazabilidad y custodia digital de la información, en cumplimiento de las disposiciones nacionales sobre el manejo de historias clínicas y la protección de datos personales. *
# *Durante el proceso de evaluación se recolectó información cualitativa mediante entrevista clínica estructurada, propia de la historia clínica ocupacional, e información cuantitativa derivada del examen físico. Para este último, se utilizaron equipos biomédicos estandarizados, garantizando la confiabilidad de las mediciones mediante la verificación de sus hojas de vida, en las cuales consta el cumplimiento de los cronogramas de calibración y mantenimiento preventivo. *
# *Posteriormente, se realizó una revisión sistemática de la totalidad de la información registrada, la cual fue organizada y analizada en una base de datos integrada. El procesamiento estadístico se llevó a cabo mediante la plataforma analítica de SOFIA. Para las variables cualitativas se emplearon distribuciones de frecuencias absolutas y relativas, mientras que para las variables cuantitativas se calcularon medidas de tendencia central (promedio, mediana) y de dispersión (rango, desviación estándar), según correspondiera. *
# [Se toma como fuente de información la matriz construida a partir de los datos extraídos de las historias clínicas ejecutadas. En total, fueron evaluados {{totales}} trabajadores, que asistieron para el examen médico ocupacional periodico durante el periodo comprendido entre el {{fecha_inicio}} y el {{fecha_fin}}. ]
# *Los resultados se desglosan en diferentes secciones: perfil sociodemográfico, perfil de hábitos saludables, perfil laboral y perfil de morbilidad. *"""

# texto_ia = """*Gráfica 8. Distribución de los conceptos de las optometria realizadas a la población evaluada .*

# #GRAFICA# #Tabla llamada 'Porcentaje de habitos' donde se evalua en el eje x el tipo de hábito, y en columna y el porcentaje de si sí practica o si no practica#

# +IA_ Análisis de 2 parrafos con aproximadamente 70 palabras donde se hable sobre: ||DATOS Conteo de registros únicos por identificación  con la clasificación de xlabel de la columna de optometria||+
# #GRAFICA# #Gráfica de barras con el nombre 'Espirometria' con el conteo único de personas por identificación con la clasificación de xlabel de la columna de Espirometria#

# *Este examen permite establecer un concepto preliminar sobre el estado de la audición, clasificandolo como normal o con indicios de alteración. Cabe aclarar que una alteración identificada en esta prueba no constituye un diagnostico clinico definitivo, por lo que, en caso de resultados anormales, se recomienda la realización de una audiometria clinica para determinar con precisión el tipo y grado de perdida auditiva del evaluado. *
# #GRAFICA# #Gráfica de tabla con el nombre 'Espirometria' con el conteo único de personas por identificación con la clasificación de xlabel de la columna de Espirometria#

# +IA_ Análisis de 2 parrafos con aproximadamente 70 palabras donde se hable sobre: ||DATOS Conteo de registros únicos por identificación  con la clasificación de xlabel de la columna de Audiometria||+

# *Durante el proceso de evaluación se recolectó información cualitativa mediante entrevista clínica estructurada, propia de la historia clínica ocupacional, e información cuantitativa derivada del examen físico. Para este último, se utilizaron equipos biomédicos estandarizados, garantizando la confiabilidad de las mediciones mediante la verificación de sus hojas de vida, en las cuales consta el cumplimiento de los cronogramas de calibración y mantenimiento preventivo. *
# *Posteriormente, se realizó una revisión sistemática de la totalidad de la información registrada, la cual fue organizada y analizada en una base de datos integrada. El procesamiento estadístico se llevó a cabo mediante la plataforma analítica de SOFIA. Para las variables cualitativas se emplearon distribuciones de frecuencias absolutas y relativas, mientras que para las variables cuantitativas se calcularon medidas de tendencia central (promedio, mediana) y de dispersión (rango, desviación estándar), según correspondiera. *

# +IA_ Análisis de 2 parrafos con aproximadamente 70 palabras donde se hable sobre: ||DATOS Conteo de registros únicos por identificación  con la clasificación de xlabel de la columna de visiometría||+

# """

# texto_plot = """
# *El análisis de estos hábitos es fundamental, ya que la adopción de estilos de vida saludables como la actividad física frecuente y la reducción del consumo de sustancias nocivas contribuye significativamente al bienestar fisico y mental, mejora la productividad laboral y disminuye el ausentismo por causas médicas. Por tanto, conocer estos datos permite orientar acciones de promoción de la salud en el entorno laboral. *

# #GRAFICA# #Gráfico de tortas llamado 'Riesgo cardiovascular' donde se cuenta el número de personas registros únicos de identificación de acuerdo a su clasificación de acuerdo con la columna de riesgo cardiovascular#
# *Tabla 2. Distribución porcentual de la población evaluada de habitos de vida saludable. *

# #GRAFICO# #Gráfico de barras con el nombre 'Espirometria' con el conteo único de personas por identificación con la clasificación de xlabel de la columna de Espirometria#
# *Se considera que una persona tiene hábitos de vida saludable cuando realiza actividad física al menos tres veces por semana, con una duración mínima de 30 minutos por sesión. El consumo habitual de licor se define como la ingesta regular, al menos una vez por semana. Por su parte, se considera fumador habitual a quien consume uno o más cigarrillos al día de forma constante. *

# #GRAFICA# #Gráfica de barras con el nombre de 'indice de masa corporal' con eje x para la clasificación de (Bajo peso: IMC menor que 18.5 kg/m², Peso normal: 18.5 menor o igual que IMC y este último menor o igual a 24.9 kg/m², Sobrepeso: 25.0 menor o igual que IMC y este último menor o igual a 29.9 kg/m², Obesidad grado I: 30.0 menor o igual que IMC y este último menor o igual a 34.9 kg/m², Obesidad grado II: 35.0 menor o igual que IMC y este último menor o igual que 39.9 kg/m², Obesidad grado III: IMC maoyr que 40.0 kg/m²) y eje x para el conteo del número de personas por número de identificación diferente#
# *Gráfica 5. Distribución de índice de masa corporal de la pobación evaluada. *

# #GRAFICA# #Gráfica de barras horizontales con el nombre 'visiometría' con el conteo único de personas por identificación con la clasificación de xlabel de la columna de visiometría#
# *Gráfica 8. Distribución de los conceptos de las visiometrias realizadas a la población evaluada .*

# #GRAFICA# #Gráfica de barras con el nombre 'Optometría' con el conteo único de personas por identificación con la clasificación de xlabel de la columna de optometria#
# *Gráfica 8. Distribución de los conceptos de las optometria realizadas a la población evaluada .*

# #GRAFICA# #Gráfica de tabla llamada 'Clasificación trabajadores por sistema afectado' en la cual se tomará la columna x la columna de tipo de sistema afectado y el conteo de número de trabajadores diferentes de acuerdo con la identificación#
# *Tabla 4. Distribución porcentual y absoluta de trabajadores evaluados conforme sintomatologia reportada por sistema afectado. *

# """

# txt_plot = """[{'idx': 1, 'prompt': "Gráfica de tabla con el nombre 'Espirometria' con el conteo único de personas por identificación con la clasificación de xlabel de la columna de Espirometria", 'span': (10, 30)}]"""



#informe_final_V2(df, df_datos, ctx)

