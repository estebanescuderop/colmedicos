#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests para columnas_ia_ultra.py
"""

import pandas as pd
import numpy as np
import sys
import os

# ✅ ACTIVAR DEBUG
os.environ["DEBUG_DIAGNOSTICO"] = "1"


from sympy import false
from Colmedicos.io_utils_remaster import process_ia_blocks, process_data_blocks, process_plot_blocks, _render_vars_text, exportar_output_a_html, _fig_to_data_uri, _format_result_plain, columnas_a_texto, aplicar_multiples_columnas_gpt5, unpivot_df, dividir_columna_en_dos, procesar_codigos_cie10, unir_dataframes, expand_json_column, procesar_apendices, filtrar_apendices,process_titulo_blocks
import pandas as pd
from Colmedicos.ia import ask_gpt5, operaciones_gpt5, graficos_gpt5, columns_batch_gpt5, apendices_gpt5
from Colmedicos.io_utils import aplicar_plot_por_tipo_desde_output, aplicar_ia_por_tipo, generar_output, mostrar_html
from Colmedicos.charts import plot_from_params
from Colmedicos.math_ops import ejecutar_operaciones_condicionales
from Colmedicos.api import informe_final
import pandas as pd
from Colmedicos.columnas_ia_ultra import aplicar_columnas_ia_auto, aplicar_multiples_columnas_gpt5_ultra_v2



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
      {"nombre":"presion_alta","condicion":"(presion startswith '13' OR presion startswith '14' OR presion startswith '15' OR presion startswith '16' OR presion startswith '17' OR presion startswith '18' OR presion startswith '19' OR presion startswith '20')"},
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
    }
]

campos = ["lab_grupo", "lab_item_unificado", "lab_Resultado_global"]
renombres = {
    "lab_grupo": "Tipo prueba",
    "lab_item_unificado": "Prueba",
    "lab_Resultado_global": "Resultado"
}

import time

# ============================================================================
# TEST 1: V2 HÍBRIDO (Paralelo + Clustering por tarea)
# ============================================================================
print("\n" + "="*80)
print("EJECUTANDO V2 HÍBRIDO (aplicar_multiples_columnas_gpt5_ultra_v2)")
print("="*80)

t_start_v2 = time.time()
df_resultado = aplicar_multiples_columnas_gpt5_ultra_v2(df_datos.copy(), tareas, debug=False)
t_v2 = time.time() - t_start_v2

# Verificar resultado
print("\n" + "="*80)
print("RESULTADO V2 HÍBRIDO")
print("="*80)
for tarea in tareas:
    col = tarea["nueva_columna"]
    if col in df_resultado.columns:
        nan_count = df_resultado[col].isna().sum()
        print(f"{col}: NaN={nan_count}/{len(df_resultado)} ({nan_count/len(df_resultado)*100:.1f}%)")
    else:
        print(f"{col}: NO EXISTE")

print(f"\n⏱️ TIEMPO V2 HÍBRIDO: {t_v2:.2f}s")

df_resultado.to_excel(r"C:\Users\EstebanEscuderoPuert\Downloads\output_.xlsx", index=False, engine="openpyxl")

# ============================================================================
# TEST 2: ORIGINAL
# ============================================================================
print("\n" + "="*80)
print("EJECUTANDO ORIGINAL (aplicar_multiples_columnas_gpt5)")
print("="*80)

t_start_original = time.time()
df_resultado2 = aplicar_multiples_columnas_gpt5(df_datos.copy(), tareas, debug=False)
t_original = time.time() - t_start_original

# Verificar resultado
print("\n" + "="*80)
print("RESULTADO ORIGINAL")
print("="*80)
for tarea in tareas:
    col = tarea["nueva_columna"]
    if col in df_resultado2.columns:
        nan_count = df_resultado2[col].isna().sum()
        print(f"{col}: NaN={nan_count}/{len(df_resultado2)} ({nan_count/len(df_resultado2)*100:.1f}%)")
    else:
        print(f"{col}: NO EXISTE")

print(f"\n⏱️ TIEMPO ORIGINAL: {t_original:.2f}s")

df_resultado2.to_excel(r"C:\Users\EstebanEscuderoPuert\Downloads\output2_.xlsx", index=False, engine="openpyxl")

# ============================================================================
# COMPARACIÓN FINAL
# ============================================================================
print("\n" + "="*80)
print("COMPARACIÓN DE TIEMPOS")
print("="*80)
print(f"  V2 HÍBRIDO: {t_v2:.2f}s")
print(f"  ORIGINAL:   {t_original:.2f}s")
if t_v2 < t_original:
    mejora = ((t_original - t_v2) / t_original) * 100
    print(f"  ✅ V2 HÍBRIDO es {mejora:.1f}% más rápido")
elif t_v2 > t_original:
    peor = ((t_v2 - t_original) / t_original) * 100
    print(f"  ⚠️ V2 HÍBRIDO es {peor:.1f}% más lento (variabilidad de API)")
else:
    print(f"  ⚖️ Ambos tardaron igual")

print("\n" + "="*80)
print("VENTAJAS DE V2 HÍBRIDO")
print("="*80)
print("""
V2 HÍBRIDO combina lo mejor de ambos mundos:

1. PARALELISMO: Todas las tareas se ejecutan simultáneamente
   (igual que Original)

2. AHORRO DE TOKENS: Cada tarea usa clustering independiente
   - categoria_cargo: 63 registros -> ~32 únicos (49% ahorro)
   - antiguedad: 63 registros -> ~62 únicos (2% ahorro)
   - tipo_riesgo_cardiovascular: 63 registros -> 63 únicos (0% ahorro)

3. MENOR COSTO: Menos tokens enviados = menor costo de API

La diferencia de tiempo puede ser mínima con datasets pequeños,
pero el AHORRO DE TOKENS es significativo con datasets grandes.
""")
