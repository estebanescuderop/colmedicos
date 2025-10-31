from django.db import models

class Item(models.Model):
    nombre = models.CharField(max_length=120)
    cantidad = models.IntegerField(default=0)
    creado = models.DateTimeField(auto_now_add=True)

# Create your models here.
