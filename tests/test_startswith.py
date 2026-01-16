import sys
import pandas as pd

# Asegurar que el paquete Colmedicos está en sys.path
sys.path.insert(0, r"c:\Users\EstebanEscuderoPuert\colmedicos\src")
from Colmedicos import charts

print('charts module:', charts)

# Caso simple
df = pd.DataFrame({
    'Tipo prueba': ['Pruebas infecciosas', 'Pruebas infecXYZ', 'Otra cosa', None, 'pruebas infecciosas con minus']
})
conds = [['Tipo prueba', 'startswith', 'Pruebas infec']]
mask = charts._mask_from_conditions(df, conds)
print('Mask for startswith "Pruebas infec":', mask.tolist())

# Comparar con icontains
conds2 = [['Tipo prueba', 'icontains', 'pruebas infec']]
mask2 = charts._mask_from_conditions(df, conds2)
print('Mask for icontains "pruebas infec":', mask2.tolist())

# Probar directamente el operador
op = charts._OPS['startswith']
print('Direct op results:', op(df['Tipo prueba'], 'Pruebas infec').tolist())

# Probar normalización alternativa
op2 = charts._OPS['icontains']
print('Direct icontains op results:', op2(df['Tipo prueba'], 'pruebas infec').tolist())
