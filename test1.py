import time as TTT
from mip import *
import pandas as pd
from itertools import product
import random
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime

from dssProject.serializers import ScheduleSerializer
from django.http.response import JsonResponse
from rest_framework import status
from dssProject.models import Mantencion
from dssProject.serializers import MantencionSerializer
from dssProject.models import Maquina




def optimizar(semana):
    pp = PdfPages('planificacion.pdf')
    df=pd.read_excel("pln_v1.xls")
    dic=pd.read_excel("dicc.xls")
    aux1=dic[dic['t']<=96]
    df.dropna(inplace=True)
    df.drop(columns=['ESTADO COSECHA'], inplace=True)
    #df.drop(columns=['PRODUCTOR'], inplace=True)
    w1=df.loc[df['SEMANA'] ==semana, ['D','BLOQUE','kg','h','PRODUCTOR']]
    w1.reset_index(inplace=True)
    w1['t']=w1['D']*0
    w1.drop(columns=['index'], inplace=True)
    w2=pd.merge(w1,aux1, on='h', how='left')
    w2['t']=w2['t_x']+w2['t_y']
    w2.drop(columns=['t_y'], inplace=True)
    w2.drop(columns=['t_x'], inplace=True)
    w2.reset_index(inplace=True)
    w2.D=w2.D.replace({2:'Lunes',3:'Martes',4:'Miercoles', 
                   5:'Jueves', 6:'Viernes', 1:'Domingo', 7:"Sabado"})
    w2.drop(columns=['index'], inplace=True)
    w2.kg = w2.kg/1000

    #Conjuntos
    dias=[]
    for i in range(len(w2.D)):
        if w2.D[i] not in dias:
            dias.append(w2.D[i])
    time=[]
    for i in range(48):
        time.append(i)
    tasks1=['Despalillado','Prensado']
    tasks2=['Pflot','flot']
    material1=['uva','uvad','jugo','raquis','orujo']
    material2=['jugo', 'jugopf','jugof']
    grupo=[]
    for i in range(len(w2.BLOQUE)):
        if w2.BLOQUE[i] not in grupo:
            grupo.append(w2.BLOQUE[i])
    
    parte=[0,1]
    material=[material1,material2]
    tasks=[tasks1,tasks2]
    nombre=["Despalillado y prensado","Preflotación y flotación"]
    hora=[]
    for i in range(len(dic)):
        hora.append(str(dic.h[i]))
    
    
    color_g=["#F08080","#FFA500","#FFFF00","#E6E6FA","#ADFF2F","#90EE90","#00FFFF",
         "#7FFFD4","#FFF8DC","#F0FFF0","#DCDCDC","#FF69B4","#FFA500","#87CEEB"]
    
    procedencias=[]
    
    for d in range(len(dias)):
        procedencias.append([])
        for g in range(len(grupo)):
            procedencias[d].append([])
            for pr in range(len(w2.t)):
                if w2.BLOQUE[pr]==grupo[g] and w2.PRODUCTOR[pr] not in procedencias[d][g] and dias[d]==w2.D[pr]:
                    procedencias[d][g].append(w2.PRODUCTOR[pr])
    

    
    rsgt=[]
    cont=0
    for d in range(len(dias)):
        rsgt.append([])
        for pt in range(len(parte)):
            rsgt[d].append([])
            for s in range(len(material[pt])):
                rsgt[d][pt].append([])
                for g in range(len(grupo)):
                    rsgt[d][pt][s].append([])
                    for t1 in range(len(time)):
                        rsgt[d][pt][s][g].append(0)
                        for t2 in range(len(w2.t)):
                            if grupo[g]==w2.BLOQUE[t2] and material[pt][s]=='uva' and w2.D[t2]==dias[d] and time[t1]==w2.t[t2]:
                                #print(f"día: {dias[d]} match en {time[t1]} y {w2.t[t2]}, los kg son {w2.kg[t2]} y el grupo es {w2.BLOQUE[t2]}, grupo a revisar: {grupo[g]}")
                                rsgt[d][pt][s][g][t1]+=w2.kg[t2]
                                cont+=1

    #Parámetros asociados a la maquinaria. 
    df=pd.read_excel("settings.xls")
    machinesTipo = list(Maquina.objects.values("tipo"))
    machines1=[]
    machines2=[]
    for m in machinesTipo:
        if m.get("tipo").startswith('Pozo') or m.get("tipo").startswith('Prens'):
            machines1.append(m.get("tipo"))
        else:
            machines2.append(m.get("tipo"))
    Vmax1=[]
    Vmax2=[]
    Ij1=[]
    Ij2=[]
    L3=[]
    L4=[]
    P1=[]
    P2=[]
    for j in df.Maquinas:
        for n in range(len(df.Cantidad)):
            if j==df.Maquinas[n]:
                    L3.append(j)
                    L4.append(df.Cantidad[n])
    for i in range(len(L4)):
        for j in range(L4[i]):
            if L3[i].startswith('Pozo') or L3[i].startswith('Prens'):
                Vmax1.append(df.Capacidad[i])
                Ij1.append(df.Tarea[i])
                P1.append(df.Procesamiento[i])
            else: 
                Vmax2.append(df.Capacidad[i])
                Ij2.append(df.Tarea[i])
                P2.append(df.Procesamiento[i])
            #print(f"{L3[i]}{j+1}")
    Vmin1=[]
    Vmin2=[]
    for i in range(len(Vmax1)):
        Vmin1.append(Vmax1[i]*0.5)
    for i in range(len(Vmax2)):
        Vmin2.append(Vmax2[i]*0.5)
    machines=[machines1,machines2]
    Ij=[Ij1,Ij2]
    P=[P1,P2]
    Vmin=[Vmin1,Vmin2]
    Vmax=[Vmax1,Vmax2]

    Pi1=[]
    Pi2=[]
    auxde=[]
    auxpr=[]
    auxcpf=[]
    auxflt=[]
    for i in range(len(L4)):
        for j in range(L4[i]):    
            if L3[i].startswith('Pozo'):
                    auxde.append(df.Procesamiento[i])
            elif L3[i].startswith('Pren'):
                    auxpr.append(df.Procesamiento[i])
            elif L3[i].startswith('CPF'):
                    auxcpf.append(df.Procesamiento[i])
            else:
                    auxflt.append(df.Procesamiento[i])
    Pi1.append(auxde)
    Pi1.append(auxpr)
    Pi2.append(auxcpf)
    Pi2.append(auxflt)
    Pi=[Pi1,Pi2]

    mantenciones = Mantencion.objects.all()
    mantenciones_serializer = MantencionSerializer(mantenciones,many=True)
    mantenciones_json = mantenciones_serializer.data
    mantencionesdf = pd.DataFrame(columns=["tipo","inicio","final","dia"])
    for mant in range(0,len(mantenciones_json)):
        currentData = mantenciones_json[mant]
        mantencionesdf.loc[mant] = [mantenciones_json[mant]["tipo"],mantenciones_json[mant]["inicio"],mantenciones_json[mant]["final"],mantenciones_json[mant]["dia"]]
    
    for i in range(len(mantencionesdf)):
        mantencionesdf.inicio[i] = datetime.strptime(mantencionesdf.inicio[i],'%H:%M:%S').time()
        mantencionesdf.final[i] = datetime.strptime(mantencionesdf.final[i],'%H:%M:%S').time()

    #mantencionesdf=pd.read_excel("mant.xls")

    mantI=[]
    stat = False
    for d in range(len(dias)):
        mantI.append([])
        for pt in range(len(parte)):
            mantI[d].append([])
            for j in range(len(machines[pt])):
                for m in range(len(mantencionesdf)):
                    for z in range(len(aux1.h)):
                        if machines[pt][j]==mantencionesdf.tipo[m] and mantencionesdf.inicio[m]==aux1.h[z] and mantencionesdf.dia[m]== dias[d]:
                            mantI[d][pt].append(aux1.t[z])
                            stat=True
                if stat==True:
                    print('')
                else:    
                    mantI[d][pt].append(0)
                stat=False
    mantF=[]
    for d in range(len(dias)):
        mantF.append([])
        for pt in range(len(parte)):
            mantF[d].append([])
            for j in range(len(machines[pt])):
                for m in range(len(mantencionesdf)):
                    for z in range(len(aux1.h)):
                        if machines[pt][j]==mantencionesdf.tipo[m] and mantencionesdf.final[m]==aux1.h[z] and mantencionesdf.dia[m]== dias[d]:
                            mantF[d][pt].append(aux1.t[z])
                            stat=True
                if stat==True:
                    print('')
                else:    
                    mantF[d][pt].append(0)
                stat=False
    #un M lo suficientemente grande
    #M1 = sum(Pi1[i][j] for i in range(len(Pi1)) for j in range(len(Pi1[i]))) 
    #M2=sum(Pi2[i][j] for i in range(len(Pi2)) for j in range(len(Pi2[i]))) 
    M1=300
    M2=300
    M=[M1,M2]


    #Lista de tareas que pueden ser realizadas por una maquinaria
    Ki1=[] 
    Ki2=[]
    auxde=[]
    auxpr=[]
    auxcpf=[]
    auxflt=[]
    for i in range(len(machines1)):
        if machines1[i].startswith('Poz'):
            auxde.append(i)
        elif machines1[i].startswith('Pren'):
            auxpr.append(i)
    for i in range(len(machines2)):
        if machines2[i].startswith('CPF'):
            auxcpf.append(i)
        else:
            auxflt.append(i)
    Ki1.append(auxde)
    Ki1.append(auxpr)
    Ki2.append(auxcpf)
    Ki2.append(auxflt)
    Ki=[Ki1,Ki2]


    #### Si={'Despalillado':'uva', 'Prensado':'uvad','Preflotacion':'jugo','Flotacion':'pref'}
    ####So={'Despalillado':('uvad','raquis'),'Prensado':('jugo','orujo'),'Preflotacion':'pref','Flotacion':'jugof'}
    Si1=[0,1]
    So1=[[1,5],[2,6]]
    Si2=[0,1]
    So2=[1,2]
    Si=[Si1,Si2]
    So=[So1,So2]


    #Rhos de entrada y salida en lista.
    rhoi1={
        ('Despalillado','uva'):1,
        ('Despalillado','uvad'):0,
        ('Despalillado','jugo'):0,
        ('Despalillado','raquis'):0,
        ('Despalillado','orujo'):0,
        ('Prensado','uva'):0,
        ('Prensado','uvad'):1,
        ('Prensado','jugo'):0,
        ('Prensado','raquis'):0,
        ('Prensado','orujo'):0,
    }
    rhoi2={
        ('Preflotacion','jugo'):1,
        ('Preflotacion','pref'):0,
        ('Preflotacion','jugof'):0,
        ('Flotacion','jugo'):0,
        ('Flotacion','pref'):1,
        ('Flotacion','jugof'):0,
    }
    rhoo1={
        ('Despalillado','uva'):0,
        ('Despalillado','uvad'):0.96,
        ('Despalillado','jugo'):0,
        ('Despalillado','raquis'):0.04,
        ('Despalillado','orujo'):0,
        ('Prensado','uva'):0,
        ('Prensado','uvad'):0,
        ('Prensado','jugo'):0.867,
        ('Prensado','raquis'):0,
        ('Prensado','orujo'):0.133,
    }
    rhoo2={
        ('Preflotacion','jugo'):0,
        ('Preflotacion','pref'):1,
        ('Preflotacion','jugof'):0,
        ('Flotacion','jugo'):0,
        ('Flotacion','pref'):0,
        ('Flotacion','jugof'):1,
    }
    rhoil1=[]
    rhool1=[]
    rhoil2=[]
    rhool2=[]
    auxde=[]
    auxpr=[]
    auxcpf=[]
    auxflt=[]
    for i,s in rhoi1.items():
        if i[0].startswith('Desp'):
            auxde.append(s)
        elif i[0].startswith('Pren'):
            auxpr.append(s)
    for i,s in rhoi2.items():
        if i[0].startswith('Preflo'):
            auxcpf.append(s)
        else:
            auxflt.append(s)
    rhoil1.append(auxde)
    rhoil1.append(auxpr)
    rhoil2.append(auxcpf)
    rhoil2.append(auxflt)
    auxde=[]
    auxpr=[]
    auxcpf=[]
    auxflt=[]
    for i,s in rhoo1.items():
        if i[0].startswith('Desp'):
            auxde.append(s)
        elif i[0].startswith('Pren'):
            auxpr.append(s)
    for i,s in rhoo2.items():
        if i[0].startswith('Preflo'):
            auxcpf.append(s)
        else:
            auxflt.append(s)
    rhool1.append(auxde)
    rhool1.append(auxpr)
    rhool2.append(auxcpf)
    rhool2.append(auxflt)
    rhoil=[rhoil1,rhoil2]
    rhool=[rhool1,rhool2]

    Til1=[[0],[1],[],[],[]]
    Tol1=[[],[0],[1],[0],[1]]
    CCC1=[10,3,0.1,0.1,0.1]
    Til2=[[0],[1],[]]
    Tol2=[[],[0],[1]]
    CCC2=[10,3,0.1]
    Til=[Til1,Til2]
    Tol=[Tol1,Tol2]



    Cs1=[]
    for s in range(len(material[0])):
        Cs1.append([])
        for t in range(len(time)):
            Cs1[s].append(CCC1[s]*(1+t))

    Cs2=[]        
    for s in range(len(material[1])):
        Cs2.append([])
        for t in time:
            Cs2[s].append(CCC2[s]*(1+t))
    Cs=[Cs1,Cs2]

    def crear_gantt(maquinas, ht):
        # Parámetros:
        hbar = 4
        tticks = 4
        nmaq = len(maquinas)

        # Creación de los objetos del plot:
        fig, gantt = plt.subplots(figsize=(25,20))
        gantt.set_title(f"Planificación día {dias[d]}, tareas {nombre[pt]}",size=20)
        # Diccionario con parámetros:
        diagrama = {
            "fig": fig,
            "ax": gantt,
            "hbar": hbar,
            "tticks": tticks,
            "maquinas": maquinas,
            "ht": ht
        }

        # Etiquetas de los ejes:
        gantt.set_xlabel("Tiempo",size=20)
        gantt.set_ylabel("Máquinas",size=20)
        

        # Límites de los ejes:
        gantt.set_xlim(0, ht)
        gantt.set_ylim(0, nmaq*hbar)

        # Divisiones del eje de tiempo:
        gantt.set_xticks(time)
        gantt.grid(True, axis='x', which='both')
        gantt.set_xticklabels(hora, size="15",rotation='vertical')

        
        # Divisiones del eje de máquinas:
        gantt.set_yticks(range(hbar, nmaq*hbar, hbar), minor=True)
        gantt.grid(True, axis='y', which='minor')

        # Etiquetas de máquinas:
        gantt.set_yticks(np.arange(hbar/2, hbar*nmaq - hbar/2 + hbar,
                                hbar))
        gantt.set_yticklabels(maquinas, size="15")

        return diagrama

    # Función para armar tareas:
    def agregar_tarea(diagrama, t0, d, maq, nombre, color,carga):
        maquinas = diagrama["maquinas"]
        hbar = diagrama["hbar"]
        gantt = diagrama["ax"]
        ht = diagrama["ht"]

        # Chequeos:
        assert t0 + d <= ht, "La tarea debe ser menor al horizonte temporal."
        assert t0 >= 0, "El t0 no puede ser negativo."
        assert d > 0, "La duración d debe ser positiva."


        # Índice de la máquina:
        imaq = maquinas.index(maq)
        # Posición de la barra:
        gantt.broken_barh([(t0, d)], (hbar*imaq, hbar),
                        facecolors=(color))
        # Posición del texto:
        gantt.text(x=(t0 + d/2), y=(hbar*imaq + hbar/2),s=f"({round(carga,2)})", va='center', ha='center', color='black',size=15)

    def gruposd(Dias):    
        df = pd.DataFrame(
        procedencias[Dias],index=grupo)
        df.reset_index(inplace=True)
        df.rename(columns={'index':'Grupo'}, inplace=True)
        color=[]
        col_count = df.shape[1]
        for i in range(len(df)):
            color.append([])
            for r in range(col_count):
                color[i].append(color_g[i])

        fig, ax = plt.subplots(figsize=(25,4))

        # hide axes
        fig.patch.set_visible(False)
        ax.axis('off')
        ax.axis('tight')
        ax.set_title(f"organización de los bloques de mezcla del día: {dias[Dias]}")
        the_table = ax.table(cellText=df.values, colLabels=df.columns,cellLoc='center' ,loc='center',cellColours=color )
        the_table.auto_set_font_size(False)
        the_table.set_fontsize(12.5)

        fig.tight_layout()


    def opt(D,PT,dia):
        m=Model(sense=MINIMIZE,solver_name=CBC)
        W1=[[[[m.add_var(var_type=BINARY, name='WA({},{},{},{})'.format(i,j,g,t))for t in range(len(time))] for g in range(len(grupo))]for j in range(len(machines[PT]))] for i in range(len(tasks[PT]))]
        B1=[[[[m.add_var(var_type=CONTINUOUS, name='BA({},{},{},{})'.format(i,j,g,t))for t in range(len(time))] for g in range(len(grupo))]for j in range(len(machines[PT]))] for i in range(len(tasks[PT]))]
        S1=[[[m.add_var(var_type=CONTINUOUS, name='SA({},{},{})'.format(s,g,t))for t in range(len(time))] for g in range(len(grupo))]for s in range(len(material[PT]))]
        
        m.objective=minimize(sum(Cs[PT][s][t] * S1[s][g][t] for t in range(len(time)) for g in range(len(grupo)) for s in range(len(material[PT])) ))
        
        for (I,G,T,J) in product(range(len(tasks[PT])), range(len(grupo)), range(len(time)), range(len(machines[PT]))):
            if J in Ki[PT][I]:
                m+=(B1[I][J][G][T]>= Vmin[PT][J]*W1[I][J][G][T], (f"C1({I},{J},{G},{T})"))
                m+=(B1[I][J][G][T]<= Vmax[PT][J]*W1[I][J][G][T], (f"C2({I},{J},{G},{T})"))
                
            
        for (J, T, I, G) in product(range(len(machines[PT])), range(len(time)), range(len(tasks[PT])), range(len(grupo))):
            if I == Ij[PT][J]:
                if T<= time[-1]-P[PT][J]:
                    m+=(xsum(W1[i][J][g][t] for t in range(T,T+P[PT][J]) for g in range(len(grupo)) for i in range(len(tasks[PT])) if i == Ij[PT][J])-1<= M[PT]*(1-W1[I][J][G][T]),(f"C3({I},{J},{G},{T})") )
                else:
                    m+=W1[I][J][G][T]==0

        for (s,t,g) in product(range(len(material[PT])),range(len(time)),range(len(grupo))):
            if t>= 1:
                m+= (S1[s][g][t] == S1[s][g][t-1] + rsgt[D][PT][s][g][t] + sum(rhool[PT][i][s]*sum(B1[i][j][g][t-(P[PT][j])] for j in Ki[PT][i] if t>= P[PT][j])for i in Tol[PT][s])-(sum(rhoil[PT][i][s]*sum(B1[i][j][g][t] for j in Ki[PT][i])for i in Til[PT][s])),(f"C4({s},{g},{t})") )

            else:
                m+= (S1[s][g][t] == rsgt[D][PT][s][g][t] - (sum(rhoil[PT][i][s]*sum(B1[i][j][g][t] for j in Ki[PT][i])for i in Til[PT][s])))
        for J in range(len(machines[PT])):
            if mantI[D][PT][J]<mantF[D][PT][J]:
                for t in range(mantI[D][PT][J],mantF[D][PT][J]+1):
                    m+= (xsum(W1[I][J][G][t] for I in range(len(tasks[PT])) for G in range(len(grupo)))==0)
        for (G,T,J) in product(range(len(grupo)), range(len(time)),range(len(machines[PT]))):
            if grupo[G]=='SPK' and machines[PT][J].endswith('SPK')==False and machines[PT][J].startswith('Pozo'):
                m+=( W1[0][J][G][T] == 0 )
            if grupo[G]!='SPK' and machines[PT][J].endswith('SPK')==True and machines[PT][J].startswith('Pozo'):
                m+=( W1[0][J][G][T] == 0 ) 
        m.max_gap = 0.05
        status = m.optimize(max_seconds=100)


        Wi=[]
        Wr=[]
        Br=[]
        si=[]
        sr=[]
        for i in m.vars:
            if str(i).startswith('W'):
                if i.x ==1:
                    Wi.append(i.name)
                    Wr.append(i.x)
            if str(i).startswith('B'):
                if i.x !=0:
                    Br.append(round(i.x,2))
            if str(i).startswith('S'):          
                si.append(i.name)   
                sr.append(i.x)
        WI=[]
        WJ=[]
        WG=[]
        WT=[]
        SS=[]
        SG=[]
        ST=[]
        fuera="WA'()"
        for i in Wi:
            for x in range(len(fuera)):
                i= i.replace(fuera[x],"")
            for z in range(len(i.split(","))):
                if z == 0:
                    WI.append(i.split(",")[int(z)])
                if z == 1:
                    WJ.append(i.split(",")[int(z)])
                if z== 2:
                    WG.append(i.split(",")[int(z)])
                if z==3:
                    WT.append(i.split(",")[int(z)])

        fuera="SA('')"
        for i in si:
            for x in range(len(fuera)):
                i= i.replace(fuera[x],"")
            for z in range(len(i.split(","))):
                if z == 0:
                    SS.append(i.split(",")[int(z)])
                if z == 1:
                    SG.append(i.split(",")[int(z)])
                if z== 2:
                    ST.append(i.split(",")[int(z)])
        for i in range(len(WI)):
            id = 1
            tarea2 = tasks[PT][int(WI[i])]
            grupo2 = grupo[int(WG[i])]
            maquina2 = machines[PT][int(WJ[i])]
            horario2 = str(dic.h[int(WT[i])])
            carga2 = Br[i]
            dia2 = dia
            dicc = {'id_schedule':i,'tarea':tarea2,'grupo':grupo2,'maquina':maquina2,'dia':dia2,'horario':horario2,'carga':carga2}
            #print(dicc)
            schedule_serializer = ScheduleSerializer(data = dicc)
            if schedule_serializer.is_valid():
                schedule_serializer.save()
            else:
                print("no entra")
                print(dicc)
        
        if PT==0:
            for i in range(len(si)):
                if ST[i]=='47' :
                    #print(f"el material {material[PT][int(SS[i])]} del grupo {SG[i]} contiene {sr[i]} al final del periodo")
                    if SS[i]!='2' and D<len(dias)-1:   
                        rsgt[D+1][PT][int(SS[i])][int(SG[i])][0]+=sr[i]
                if i>1 and sr[i]!= sr[i-1] and SS[i]=="2" and SG[i]==SG[i-1]:
                    rsgt[D][PT+1][0][int(SG[i])][int(ST[i])]+= sr[i]-sr[i-1]
                    #print(f"{material[PT][int(SS[i])]} del grupo {SG[i]} adquiere {sr[i]-sr[i-1]} en el periodo {ST[i]}")
        if PT==1:
            for i in range(len(si)):
                if ST[i]=='47' :
                    if SS[i]!='2' and D<len(dias)-1:
                        #print(f"el material {material[PT][int(SS[i])]} del grupo {SG[i]} contiene {sr[i]} al final del periodo")
                        rsgt[D+1][PT][int(SS[i])][int(SG[i])][0]+=sr[i]
        if PT==0:
            gruposd(D)
            pp.savefig()

        gant=crear_gantt(machines[PT],len(time))
        for i in range(len(Wi)):
            agregar_tarea(gant,int(WT[i]),P[PT][int(WJ[i])],machines[PT][int(WJ[i])],grupo[int(WG[i])],color_g[int(WG[i])],Br[i])
            if mantI[D][PT][int(WJ[i])]!=mantF[D][PT][int(WJ[i])]:
                agregar_tarea(gant,mantI[D][PT][int(WJ[i])],mantF[D][PT][int(WJ[i])]-mantI[D][PT][int(WJ[i])],machines[PT][int(WJ[i])],'Mantencion','r',1)
        pp.savefig()
        dat=[]
        for t in range(len(time)):
            dat.append([])
            for s in range(len(material[PT])):
                aux=0
                for S in range(len(si)):
                    if time[t]==time[int(ST[S])] and material[PT][s]==material[pt][int(SS[S])]:
                        aux+=sr[S]
                dat[t].append(round(aux,2))
        aux2=pd.DataFrame(data=dat,index=time,columns=material[PT])
        aux4=pd.DataFrame(dat[-1],index=material[pt],columns=['Ton'])
        aux4.reset_index(inplace=True)
        aux4.rename(columns={'index':'Material'}, inplace=True)
        sns.set_style('whitegrid')
        fig=aux2.plot.line( y=material[PT],xticks=time,fontsize=15,figsize=(25,20), title=f'Flujo de materiales día {dias[D]}')
        fig.set_xticks(time)
        fig.set_xticklabels(hora, size="15",rotation='vertical')
        pp.savefig()
        
        fig, ax = plt.subplots(figsize=(25,3))

        fig.patch.set_visible(False)
        ax.axis('off')
        ax.axis('tight')
        ax.set_title(f"Materiales disponibles al final de las tareas {nombre[PT]} del día: {dias[D]}")
        the_table = ax.table(cellText=aux4.values,cellLoc='center' ,loc='center',colLabels=aux4.columns)
        the_table.auto_set_font_size(False)
        the_table.set_fontsize(12.5)
        fig.tight_layout()
        
        pp.savefig()




    for d in range(len(dias)):
        #print(F"El dia {dias[d]} se sigue la siguiente planificacion:")
        for pt in range(len(parte)):
            #print(F"La parte{parte[pt]} se sigue la siguiente planificacion:")
            opt(d,pt,dias[d])
    pp.close()
    
    

def crearOptimizacion(semana):
    try:
        gt=TTT.time()
        optimizar(semana)
        print("TODO ESTO DEMORO--- %s segundos ---" % (TTT.time() - gt))

    except Exception as err:
        print(err)









































