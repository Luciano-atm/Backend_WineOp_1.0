from django.db import models

class Maquina(models.Model):
    id_maquina = models.AutoField(primary_key=True)
    tipo = models.CharField(max_length=500)
    capacidad = models.IntegerField(default=0, blank=True,)
    tarea = models.CharField(max_length=500)

class Mantencion(models.Model):
    id_mantencion = models.AutoField(primary_key=True)
    tipo = models.CharField(max_length=500)
    inicio = models.CharField(max_length=500)
    final = models.CharField(max_length=500)
    dia = models.CharField(max_length=500)


class Schedule(models.Model):
    id_schedule = models.AutoField(primary_key=True)
    tarea = models.CharField(max_length=500)
    grupo = models.CharField(max_length=500)
    maquina = models.CharField(max_length=500)
    dia = models.CharField(max_length=500)
    horario = models.CharField(max_length=500)
    carga = models.FloatField(default=0, blank=True,)
