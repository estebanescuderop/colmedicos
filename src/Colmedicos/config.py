# src/tu_proyecto/config.py
# src/Colmedicos/config.py  (idealmente SIN acento en el nombre del paquete)
import os
from dotenv import load_dotenv

# Carga el .env ubicado en la raíz del proyecto
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Validación mínima para fallar temprano si falta la clave
if not OPENAI_API_KEY:
    raise RuntimeError("Falta OPENAI_API_KEY en tu .env")
