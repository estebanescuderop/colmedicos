import sys
sys.path.insert(0, r"c:\Users\EstebanEscuderoPuert\colmedicos\src")
import pandas as pd
from Colmedicos import charts

# Construir df de prueba
DF = pd.DataFrame({
    'Tipo prueba': ['Pruebas infecciosas', 'Hemograma', 'Pruebas infectivas extra', 'otro'],
    'documento': [1,2,3,4]
})
conds = [['Tipo prueba', 'startswith', 'Pruebas infec']]
mask = charts._mask_from_conditions(DF, conds)
print('Mask values:', mask.tolist())
print('Filtered rows:\n', DF.loc[mask])

# también probar icontains / istartswith
conds2 = [['Tipo prueba', 'icontains', 'infecc']]
print('icontains mask:', charts._mask_from_conditions(DF, conds2).tolist())

conds3 = [['Tipo prueba', 'istartswith', 'Pruebas infec']]
print('istartswith mask:', charts._mask_from_conditions(DF, conds3).tolist())
