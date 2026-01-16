import sys
sys.path.insert(0, r"c:\Users\EstebanEscuderoPuert\colmedicos\src")
from Colmedicos import charts
print([k for k in charts._OPS.keys() if 'start' in k or 'end' in k or 'icontains' in k])
