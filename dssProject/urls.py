from django.urls import re_path
from dssProject import views

from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    re_path(r'^maquinas$',views.maquinaApi),
    re_path(r'^maquina/([0-9]+)$',views.maquinaApi),
    re_path(r'^maquinasID$',views.getMaquinasId),
    re_path(r'^mantenciones$',views.getMantenciones),
    re_path(r'^createMantencion$',views.createMantencion),
    re_path(r'^optimizar$',views.createOptimizacion),
    re_path(r'^obtenerSemana$',views.obtenerSemana),
    re_path(r'^getSemana$',views.getSemana),
    re_path(r'^reinciarSimulacion$',views.reiniciarSimulacion),
    re_path(r'^getFile$',views.getFile),


    re_path(r'^iniciarPlanificacion$',views.iniciarPlanificacion),
    re_path(r'^postFile$',views.uploadFile),
    re_path(r'^getFileInput$',views.getFileInput),
    re_path(r'^setAgregarMaquina$',views.setAgregarMaquina),
    re_path(r'^getFileOutputModelo$',views.getFileOutputModelo),
    re_path(r'^getFileOutputResumen$',views.getFileOutputResumen),
    re_path(r'^getUrlPlanificacion$',views.getUrlPlanificacion),
    re_path(r'^getListas_habilitar$',views.getListas_habilitar),
    re_path(r'^cambiar_estado_maquina$',views.cambiar_estado_maquina)
    
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)