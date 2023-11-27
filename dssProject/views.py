from django.shortcuts import render, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from rest_framework.parsers import JSONParser
from django.http.response import JsonResponse
from rest_framework import status
from django.core.files.storage import FileSystemStorage
import os
from pathlib import Path
from django.http import FileResponse
from django.conf import settings
import pandas as pd
import json
import openpyxl
from openpyxl import Workbook
from openpyxl import load_workbook
from django.http import HttpResponse


import dssProject.Lectura.lectura
from dssProject.Lectura.lectura import *
import dssProject.Modelo_matemático.optback
from dssProject.Modelo_matemático.optback import *

from test1 import *

from dssProject.models import Maquina, Mantencion, Schedule
from dssProject.serializers import MaquinaSerializers, MantencionSerializer, ScheduleSerializer

BASE_DIR = Path(__file__).resolve().parent.parent


# Create your views here.
@csrf_exempt
def maquinaApi(request,id=0):
    if request.method == 'GET':
        maquinas = Maquina.objects.all()
        maquinas_serializer = MaquinaSerializers(maquinas,many=True)
        return JsonResponse(maquinas_serializer.data,safe=False)

@csrf_exempt
def getMaquinasId(request):
    if request.method == 'GET':
        maquinas = list(Maquina.objects.values('id_maquina'))
        return JsonResponse({"ID Maquinas":maquinas})

@csrf_exempt
def getMantenciones(request):
    if request.method == 'GET':
        mantenciones = Mantencion.objects.all()
        mantenciones_serializer = MantencionSerializer(mantenciones,many=True)
        return JsonResponse(mantenciones_serializer.data,safe=False)

@csrf_exempt
def createMantencion(request):
    if request.method == 'POST':
        mantencion_data = JSONParser().parse(request)
        mantencion_serializer = MantencionSerializer(data = mantencion_data)
        if mantencion_serializer.is_valid():
            mantencion_serializer.save()
            return JsonResponse(mantencion_serializer.data, status=status.HTTP_201_CREATED)
        return JsonResponse(mantencion_serializer.errors, status=status.HTTP_400_BAD_REQUEST)






@csrf_exempt
def uploadFile(request):
    myfile = request.FILES['myfile']
    mypdf = request.FILES['mypdf']
    print("pasa por aca")
    df=lectura_archivos(myfile,mypdf)
    
    return HttpResponse("Respuesta exitosa")

@csrf_exempt
def getFileInput(request):
    fs = FileSystemStorage(os.path.join(settings.MEDIA_ROOT))
    response = FileResponse(fs.open("Info-Día.xlsx",'rb'),content_type='application/force-download')
    response['Content-Disposition'] = 'attachment; filename="Info-Día.xlsx"'
    return response

@csrf_exempt
def setAgregarMaquina(request):
    if request.method == 'POST':
        data = json.loads(request.body)

        nombreMaquina = data.get('nombreMaquina')
        tipoMaquina = data.get('tipoMaquina')
        capacidadMaxima = data.get('capacidadMaxima')
        capacidadMinima = data.get('capacidadMinima')
        cantidad = data.get('cantidad')
        velocidadProcesado = data.get('velocidadProcesado')

        try:
            filepath = './Máquinas.xlsx' 
            workbook = load_workbook(filepath)
            sheet = workbook.active

            new_row = [
                nombreMaquina,
                tipoMaquina,
                capacidadMaxima,
                capacidadMinima,
                cantidad,
                velocidadProcesado
            ]
            sheet.append(new_row)  

            workbook.save(filepath)

            response = HttpResponse("Datos agregados correctamente al archivo Excel")
            return response

        except Exception as e:
            return HttpResponse(f"Error al agregar datos al archivo Excel: {str(e)}")

    else:
        return HttpResponse("No se recibieron datos o la solicitud no fue mediante POST")


@csrf_exempt
def iniciarPlanificacion(request):
    respuesta= modelo()
    return HttpResponse("Respuesta exitosa")








@csrf_exempt
def createOptimizacion(request):
    horario_count = Schedule.objects.count()
    if horario_count == 0:
        crearOptimizacion(semana)
    horario = Schedule.objects.all()
    horario_serializer = ScheduleSerializer(horario,many=True)
    return JsonResponse(horario_serializer.data,safe=False)

@csrf_exempt
def obtenerSemana(request):
    semana_data = JSONParser().parse(request)
    num = int(semana_data.get('semana'))
    global semana
    semana = num
    print(num)
    return JsonResponse(semana_data,status=status.HTTP_201_CREATED)

@csrf_exempt
def getSemana(request):
    dicc = {'semana':semana}
    return JsonResponse(semana,safe=False)

@csrf_exempt
def reiniciarSimulacion(request):
    Schedule.objects.all().delete()
    Mantencion.objects.all().delete()
    fs = FileSystemStorage()
    fs.delete("pln_v1.xls")
    fs.delete('planificacion.pdf')
    horario = Schedule.objects.all()
    horario_serializer = ScheduleSerializer(horario,many=True)
    return JsonResponse(horario_serializer.data,safe=False)

@csrf_exempt
def getFile(request):
    fs = FileSystemStorage(os.path.join(settings.MEDIA_ROOT))
    response = FileResponse(fs.open("planificacion.pdf",'rb'),content_type='application/force-download')
    response['Content-Disposition'] = 'attachment; filename="planificacion.pdf"'
    return response


