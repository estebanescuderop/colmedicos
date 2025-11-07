from Colmedicos.api import informe_final, informe_final_test
import pandas as pd
from Colmedicos.ia import ask_gpt5
from Colmedicos.io_utils import aplicar_plot_por_tipo_desde_output, aplicar_ia_por_tipo, generar_output, exportar_output_a_html, mostrar_html

# Ruta del archivo Excel
ruta_archivo = r"C:\Users\EstebanEscuderoPuert\Downloads\Ejemplo tabla.xlsx"
# Lee el archivo Excel (por defecto lee la primera hoja)
df = pd.read_excel(ruta_archivo)

ctx = {
    "nombre_cliente": "D1 S.A.S.",
    "nit_cliente": "901.234.567-8",
    "fecha_inicio": "2025-01-01",
    "fecha_fin": "2025-09-30",
    "numero_personas": 99,
    "totales": 150
}

# Ruta del archivo Excel
ruta_archivos = r"C:\Users\EstebanEscuderoPuert\Downloads\output_d1_colmedicos.xlsx"
df_datos = pd.read_excel(ruta_archivos)


informe_final_test(df,df_datos,ctx,valor_tipo_objetivo="Fijo con IA",reemplazar_en_html=True,token_reemplazo="#GRAFICA#")

#df_out = generar_output(df)


#df_ia = aplicar_ia_por_tipo(df_out,df_datos)

#df_final = aplicar_plot_por_tipo_desde_output(df_ia,df_datos)


#mostrar_html(exportar_output_a_html(df_final))
