from rest_framework import serializers
from .models import Maquina, Mantencion, Schedule

class MaquinaSerializers(serializers.ModelSerializer):
    class Meta:
        model = Maquina
        fields = ('id_maquina','tipo','capacidad','tarea')

class MantencionSerializer(serializers.ModelSerializer):
    class Meta:
        model = Mantencion
        fields = ('id_mantencion','tipo','inicio','final','dia')

class ScheduleSerializer(serializers.ModelSerializer):
    class Meta:
        model = Schedule
        fields = ('id_schedule','tarea','grupo','maquina','dia','horario','carga')


