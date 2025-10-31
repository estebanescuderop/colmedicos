# api/views.py
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.http import HttpResponse
import pandas as pd, io, json
from Colmedicos.api import informe_final  # importa tu función real
from rest_framework import viewsets
from .models import Item
from .serializers import ItemSerializer
from rest_framework import renderers, status

class ItemViewSet(viewsets.ModelViewSet):
    queryset = Item.objects.all().order_by("-id")
    serializer_class = ItemSerializer


class InformeFinalView(APIView):
    def post(self, request):
        if request.content_type.startswith("application/json"):
            df = pd.DataFrame(request.data.get("df", []))
            df_datos = pd.DataFrame(request.data.get("df_datos", []))
            ctx = request.data.get("ctx", {})
            valor = request.data.get("valor_tipo_objetivo", "Fijo con IA")
            repl  = request.data.get("reemplazar_en_html", True)
            token = request.data.get("token_reemplazo", "#GRAFICA#")
        else:
            df_file = request.FILES.get("df_file")
            df_datos_file = request.FILES.get("df_datos_file")
            if not df_file or not df_datos_file:
                return Response("Faltan df_file y/o df_datos_file", status=status.HTTP_400_BAD_REQUEST)

            def load_df(f):
                name = (f.name or "").lower(); data = f.read()
                return pd.read_csv(io.BytesIO(data)) if name.endswith(".csv") else pd.read_excel(io.BytesIO(data))

            df = load_df(df_file)
            df_datos = load_df(df_datos_file)
            ctx = json.loads(request.POST.get("ctx_json","{}"))
            valor = request.POST.get("valor_tipo_objetivo","Fijo con IA")
            repl  = request.POST.get("reemplazar_en_html","true").lower()=="true"
            token = request.POST.get("token_reemplazo", "#GRAFICA#")

        from Colmedicos.api import informe_final
        html, meta = informe_final(df, df_datos, ctx, valor, repl, token)
        return Response(html, content_type="text/html; charset=utf-8")
