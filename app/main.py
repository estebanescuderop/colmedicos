from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import pandas as pd
import io
import json
from typing import Optional

from Colmedicos.api import informe_final  # Ajusta si cambia la firma real


app = FastAPI(
    title="Colmedicos Analytics API",
    description=(
        "API para generar el informe final con análisis, gráficas embebidas y contexto del cliente. "
        "Puede recibir data en Excel/CSV o en JSON tabular."
    ),
    version="1.1.0",
)


class InformeResponse(BaseModel):
    status: str
    html_final: str
    meta: dict | None = None


@app.get("/health")
def healthcheck():
    return {"status": "ok"}


@app.post("/informe", response_model=InformeResponse)
async def generar_informe(
    # --- Entrada de DF principal ---
    archivo_principal: Optional[UploadFile] = File(
        default=None,
        description="Excel/CSV base (df). Ej: Ejemplo tabla.xlsx"
    ),
    json_principal: Optional[str] = Form(
        default=None,
        description='JSON tabular para df. Ej: [{"col1":1,"col2":"a"}, {"col1":2,"col2":"b"}]'
    ),

    # --- Entrada de DF detalle ---
    archivo_detalle: Optional[UploadFile] = File(
        default=None,
        description="Excel/CSV detalle (df_datos). Ej: Historias_D1_SAS_2025-07.xlsx"
    ),
    json_detalle: Optional[str] = Form(
        default=None,
        description='JSON tabular para df_datos. Mismo formato de lista de objetos.'
    ),

    # --- Contexto del informe (cliente, fechas, totales, etc.) ---
    ctx: str = Form(
        ...,
        description=(
            'JSON con contexto del cliente. '
            'Ej: {"nombre_cliente":"Avícola Andina S.A.S.","nit_cliente":"901.234.567-8",'
            '"fecha_inicio":"2025-01-01","fecha_fin":"2025-09-30","numero_personas":99,"totales":150}'
        )
    ),

    # --- Parámetros opcionales para tu lógica ---
    valor_tipo_objetivo: str = Form("Fijo con IA"),
    reemplazar_en_html: bool = Form(True),
    token_reemplazo: str = Form("#GRAFICA#"),
):

    # Utilidades internas ---------------------------------------------

    def extension_valida(nombre: str) -> bool:
        nombre = nombre.lower()
        return (
            nombre.endswith(".xlsx")
            or nombre.endswith(".xls")
            or nombre.endswith(".csv")
        )

    def cargar_excel_o_csv(upload: UploadFile) -> pd.DataFrame:
        """
        Lee un UploadFile (xlsx/xls/csv) a DataFrame.
        Lanza HTTPException si el formato no es soportado.
        """
        if upload is None:
            raise HTTPException(status_code=400, detail="Archivo no suministrado.")

        filename = upload.filename or ""
        if not extension_valida(filename):
            raise HTTPException(
                status_code=400,
                detail=f"Formato no soportado para {filename}. Usa .xlsx/.xls/.csv",
            )

        # leer contenido en memoria
        raw_bytes = file_bytes_map[upload.filename]
        # Nota: We'll fill file_bytes_map más abajo

        # decidir cómo leer
        if filename.lower().endswith(".csv"):
            return pd.read_csv(io.BytesIO(raw_bytes))
        else:
            return pd.read_excel(io.BytesIO(raw_bytes))

    def cargar_json_tabular(s: str) -> pd.DataFrame:
        """
        Convierte un string JSON en DataFrame.
        Espera lista de objetos o lista de listas con headers consistentes.
        """
        try:
            data = json.loads(s)
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"json_tabular inválido ({e}). Debe ser JSON válido."
            )

        try:
            df_local = pd.DataFrame(data)
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"No pude convertir el JSON en DataFrame ({e}). "
                       "Asegúrate de mandar una lista de objetos homogéneos."
            )
        if df_local.empty:
            # vacío puede ser válido dependiendo del caso, pero avisemos
            pass
        return df_local

    def resolver_dataframe(
        upload: Optional[UploadFile],
        json_str: Optional[str],
        nombre_logico: str,
    ) -> tuple[pd.DataFrame, dict]:
        """
        Devuelve (df_resuelto, meta_info)
        Reglas:
        - Si viene archivo -> lo usamos.
        - Si no viene archivo pero viene json -> usamos json.
        - Si no viene nada -> error 400.
        Además devolvemos metadata sobre qué se usó.
        """
        if upload is not None:
            # cargar desde archivo
            df_res = cargar_excel_o_csv(upload)
            return df_res, {
                "origen": "archivo",
                "filename": upload.filename,
                "rows": len(df_res),
                "cols": list(df_res.columns),
                "dataset": nombre_logico,
            }

        if json_str is not None:
            df_res = cargar_json_tabular(json_str)
            return df_res, {
                "origen": "json",
                "rows": len(df_res),
                "cols": list(df_res.columns),
                "dataset": nombre_logico,
            }

        raise HTTPException(
            status_code=400,
            detail=(
                f"Debes enviar {nombre_logico} como archivo (.xlsx/.xls/.csv) "
                f"o como json_{nombre_logico} (JSON tabular). Ninguno fue provisto."
            ),
        )

    # -----------------------------------------------------------------
    # 0. Leemos bytes de los archivos primero (porque .read() es await)
    # -----------------------------------------------------------------

    file_bytes_map: dict[str, bytes] = {}

    # leemos archivo_principal
    if archivo_principal is not None:
        raw_a = await archivo_principal.read()
        file_bytes_map[archivo_principal.filename] = raw_a

    # leemos archivo_detalle
    if archivo_detalle is not None:
        raw_b = await archivo_detalle.read()
        file_bytes_map[archivo_detalle.filename] = raw_b

    # -----------------------------------------------------------------
    # 1. Parsear ctx a dict
    # -----------------------------------------------------------------

    try:
        ctx_dict = json.loads(ctx)
        if not isinstance(ctx_dict, dict):
            raise ValueError("ctx debe ser un objeto JSON, no lista ni texto suelto")
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=(
                f"ctx no es JSON válido ({e}). Ejemplo ctx: "
                '{"nombre_cliente":"Avícola Andina S.A.S.","nit_cliente":"901.234.567-8",'
                '"fecha_inicio":"2025-01-01","fecha_fin":"2025-09-30","numero_personas":99,"totales":150}'
            )
        )

    # -----------------------------------------------------------------
    # 2. Resolver df y df_datos con prioridad archivo > json
    # -----------------------------------------------------------------

    df, meta_df = resolver_dataframe(
        upload=archivo_principal,
        json_str=json_principal,
        nombre_logico="df",
    )

    df_datos, meta_df_datos = resolver_dataframe(
        upload=archivo_detalle,
        json_str=json_detalle,
        nombre_logico="df_datos",
    )

    # -----------------------------------------------------------------
    # 3. Ejecutar tu lógica principal
    # -----------------------------------------------------------------

    try:
        html_final, meta_informe = informe_final(
            df,
            df_datos,
            ctx_dict,
            valor_tipo_objetivo=valor_tipo_objetivo,
            reemplazar_en_html=reemplazar_en_html,
            token_reemplazo=token_reemplazo,
        )
        # Ajusta si tu firma real de informe_final tiene nombres diferentes.
        # Lo importante es: informe_final(df, df_datos, ctx_dict, ...)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Ocurrió un error interno generando el informe: {e}",
        )

    # -----------------------------------------------------------------
    # 4. Construir metadata de respuesta
    # -----------------------------------------------------------------

    meta_global = {
        "ctx": ctx_dict,
        "df_info": meta_df,
        "df_datos_info": meta_df_datos,
        "pipeline_meta": meta_informe if isinstance(meta_informe, dict) else None,
        "parametros": {
            "valor_tipo_objetivo": valor_tipo_objetivo,
            "reemplazar_en_html": reemplazar_en_html,
            "token_reemplazo": token_reemplazo,
        },
    }

    respuesta = InformeResponse(
        status="ok",
        html_final=html_final,
        meta=meta_global,
    )

    return JSONResponse(content=respuesta.model_dump())
