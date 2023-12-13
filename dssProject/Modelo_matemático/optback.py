import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from pyomo.environ import *
from time import time
import os
import numpy as np
import tabula
from tabula.io import read_pdf
from matplotlib.patheffects import withStroke
from datetime import datetime
import glob
import PyPDF2

def modelo():
    def read_input_file(df):
        input = df
        GROUPS = set(input['Bloque'].unique())
        #intervalos = [[i, i + 1] for i in range(int(round(min(input['Hr Bodega']),0)), 24, 1)]
        intervalos = [[i, i + 1] for i in range(0, 24, 1)]
        miniTIME = [i for i in range(len(intervalos))]
        input['t'] = pd.cut(input['Hr Bodega'],
                                bins=[intervalo[0] for intervalo in intervalos] + [intervalos[-1][1]],
                                labels=miniTIME, right=False)
        Rgrup = input.groupby(['t','Bloque'])['Kilos'].sum()/1000
        Recieve = Rgrup.reset_index()
        
        return GROUPS, Recieve

    def preprocess_machines_data(file_path):
        df = pd.read_excel(file_path)
        df['Bmin'] = df['Bmax']*0.4
        CubasDisp = df[df['Máquina'].str.startswith('CubaF')]
        Bminprom = CubasDisp['Bmin'].mean()
        ReqCfer = round(ReqTon*0.96*0.84*0.94*760/Bminprom,0)
        df['Estado'] = ['Deshabilitado' if 'CubaF' in maquina and int(maquina.split('_')[1]) > ReqCfer else 'Habilitado' for maquina in df['Máquina']]
        df=df[df['Estado'] == 'Habilitado'].reset_index()
        rho_convert = {'Despalillado': 1.0,
                'Prensado': 1.0,
                'Pre-flotación': 1/760,
                'Flotación': 1/760,
                'Fermentación': 1/760}
        UNITS_TASKS = {(df['Máquina'][i],df['Tarea'][i]): {'Bmin': round(df['Bmin'][i]*rho_convert[df['Tarea'][i]],0), 
                                                        'Bmax': round(df['Bmax'][i]*rho_convert[df['Tarea'][i]],0),
                                                        'Proc': df['Proc'][i], 'Tclean':0} for i in range(len(df))}
        return UNITS_TASKS

    def create_VCT():
        # planning horizon
        H = 24
        
        VCT = {
            # time grid
            'TIME':  range(0, H+1),
        
            # states
            'STATES': {
                'Racimo_uva'        : {'capacity': float("inf"), 'initial': 0, 'price':  100},
                'Uva_despalillada'  : {'capacity': float("inf"), 'initial': 0,   'price':  80},
                'Escobajo'          : {'capacity': float("inf"), 'initial': 0,    'price':  1},
                'Jugo_uva_I'        : {'capacity': float("inf"), 'initial': 0,    'price': 60},
                'Orujo'             : {'capacity': float("inf"), 'initial': 0,    'price': 1},
                'Jugo_uva_II'       : {'capacity': float("inf"), 'initial': 0,    'price': 40},
                'Borra'             : {'capacity': float("inf"), 'initial': 0,    'price': 1},
                'Mosto'             : {'capacity': float("inf"), 'initial': 0,    'price': 20},
                'Vino'              : {'capacity': float("inf"), 'initial': 0,     'price': 1},
            },
        
            # state-to-task arcs indexed by (state, task)
            'ST_ARCS': {
                ('Racimo_uva',        'Despalillado')   : {'rho': 1.0},
                ('Uva_despalillada',  'Prensado')       : {'rho': 1.0},
                ('Jugo_uva_I',        'Pre-flotación')  : {'rho': 1.0},
                ('Jugo_uva_II',       'Flotación')      : {'rho': 1.0},
                ('Mosto',             'Fermentación')   : {'rho': 1.0}
            },
        
            # task-to-state arcs indexed by (task, state)
            'TS_ARCS': {
                ('Despalillado',  'Uva_despalillada') : {'dur': 1, 'rho': 0.96},
                ('Despalillado',  'Escobajo')         : {'dur': 1, 'rho': 0.04},
                ('Prensado',      'Jugo_uva_I')       : {'dur': 4, 'rho': 0.86},
                ('Prensado',      'Orujo')            : {'dur': 4, 'rho': 0.14},
                ('Pre-flotación', 'Jugo_uva_II')      : {'dur': 1, 'rho': 1.0},
                ('Flotación',     'Mosto')            : {'dur': 2, 'rho': 0.94},
                ('Flotación',     'Borra')            : {'dur': 2, 'rho': 0.06},
                ('Fermentación',  'Vino')             : {'dur': 8, 'rho': 1.0}
            }
        }
        
        return VCT

    def initialize_R(Recieve, STATES, GROUPS, TIME):
        R = {(s, g, t): 0 for s in STATES for g in GROUPS for t in TIME}
        for i in range(len(Recieve)):
            R['Racimo_uva', Recieve['Bloque'][i], Recieve['t'][i]] = Recieve['Kilos'][i]
        return R

    def characterize_tasks(UNIT_TASKS, ST_ARCS, TS_ARCS):
        # set of tasks
        TASKS = set([i for (j,i) in UNIT_TASKS])
        
        # S[i] input set of states which feed task i
        S = {i: set() for i in TASKS}
        for (s,i) in ST_ARCS:
            S[i].add(s)
        
        # S_[i] output set of states fed by task i
        S_ = {i: set() for i in TASKS}
        for (i,s) in TS_ARCS:
            S_[i].add(s)
        
        # rho[(i,s)] input fraction of task i from state s
        rho = {(i,s): ST_ARCS[(s,i)]['rho'] for (s,i) in ST_ARCS}
        
        # rho_[(i,s)] output fraction of task i to state s
        rho_ = {(i,s): TS_ARCS[(i,s)]['rho'] for (i,s) in TS_ARCS}
        
        # K[i] set of units capable of task i
        K = {i: set() for i in TASKS}
        for (j,i) in UNIT_TASKS:
            K[i].add(j)

        return TASKS, S, S_, rho, rho_, K

    def characterize_states(STATES, ST_ARCS, TS_ARCS):
        # T[s] set of tasks receiving material from state s
        T = {s: set() for s in STATES}
        for (s,i) in ST_ARCS:
            T[s].add(i)
        
        # set of tasks producing material for state s
        T_ = {s: set() for s in STATES}
        for (i,s) in TS_ARCS:
            T_[s].add(i)
        
        # C[s] storage capacity for state s
        C = {s: STATES[s]['capacity'] for s in STATES}

        return T, T_, C

    def characterize_units(UNIT_TASKS):
        UNITS = set([j for (j,i) in UNIT_TASKS])
        
        # I[j] set of tasks performed with unit j
        I = {j: set() for j in UNITS}
        for (j,i) in UNIT_TASKS:
            I[j].add(i)
        
        # Bmax[(i,j)] maximum capacity of unit j for task i
        Bmax = {(i,j):UNIT_TASKS[(j,i)]['Bmax'] for (j,i) in UNIT_TASKS}
        
        # Bmin[(i,j)] minimum capacity of unit j for task i
        Bmin = {(i,j):UNIT_TASKS[(j,i)]['Bmin'] for (j,i) in UNIT_TASKS}
        
        pr = {j:UNIT_TASKS[(j,i)]['Proc'] for (j,i) in UNIT_TASKS}

        return UNITS, I, Bmax, Bmin, pr

    def create_maravelias_model(GROUPS,UNITS_TASKS,dfRec):
        STN = create_VCT()
        STATES = STN['STATES']
        ST_ARCS = STN['ST_ARCS']
        TS_ARCS = STN['TS_ARCS']
        UNIT_TASKS = UNITS_TASKS
        TIME = STN['TIME']
        TIME = np.array(TIME)
        R = initialize_R(dfRec, STATES, GROUPS, TIME)
    
        TASKS, S, S_, rho, rho_, K = characterize_tasks(UNIT_TASKS, ST_ARCS, TS_ARCS)
        T, T_, C = characterize_states(STATES, ST_ARCS, TS_ARCS)
        UNITS, I, Bmax, Bmin, pr = characterize_units(UNIT_TASKS)
        prensas = sorted([item for item in UNITS if item.startswith('PR')])
        indice_medio = len(prensas) // 2
        pr_left = prensas[:indice_medio]
        pr_right = prensas[indice_medio:]
        model = ConcreteModel()
        
        # W[i,j,t] 1 if task i starts in unit j at time t
        model.W = Var(TASKS, UNITS, GROUPS, TIME, domain=Boolean)
        
        # B[i,j,t,] size of batch assigned to task i in unit j at time t
        model.B = Var(TASKS, UNITS, GROUPS,TIME, domain=NonNegativeReals)
        
        # S[s,t] inventory of state s at time t
        model.S = Var(STATES.keys(), GROUPS, TIME, domain=NonNegativeReals)
        
        # Q[j,t] inventory of unit j at time t
        model.Q = Var(UNITS, TIME, domain=NonNegativeReals)
        
        model.O = Var(UNITS, TIME, domain=NonNegativeReals)
        
        # Objective function
        Cost = {(s,t): STATES[s]['price']*(1+t) for s in STATES for t in TIME}
        # project value
        model.Value = Var(domain=NonNegativeReals)
        model.valuec = Constraint(expr = model.Value == sum([Cost[s,t]*model.S[s,g,t] for s in STATES.keys() for g in GROUPS for t in TIME]))
        model.obj = Objective(expr=model.Value, sense = minimize)
        
        # Constraints
        model.cons = ConstraintList()
        
        # a unit can only be allocated to one task
        for j in UNITS:
            for t in TIME:
                lhs = 0
                for i in I[j]:
                    if i == 'Fermentación':
                        for g in GROUPS:
                            for tprime in TIME:
                                if tprime <= t:
                                    lhs += model.W[i,j,g,tprime]
                    else:
                        for g in GROUPS:
                            for tprime in TIME:
                                if tprime >= (t-pr[j]+1-UNIT_TASKS[(j,i)]['Tclean']) and tprime <= t:
                                    lhs += model.W[i,j,g,tprime]
                model.cons.add(lhs <= 1)
        
        # state capacity constraint
        model.sc = Constraint(STATES.keys(), GROUPS, TIME, rule = lambda model, s, g, t: model.S[s,g,t] <= C[s])
        
        # state mass balances
        for s in STATES.keys():
            for g in GROUPS:
                rhs = STATES[s]['initial']
                for t in TIME:
                    for i in T_[s]:
                        for j in K[i]:
                            if t >= pr[j]:
                                rhs += rho_[(i,s)]*model.B[i,j,g,max(TIME[TIME <= t-pr[j]])]
                    for i in T[s]:
                        rhs -= rho[(i,s)]*sum([model.B[i,j,g,t] for j in K[i]])
                    model.cons.add(model.S[s,g,t] == rhs+R[s,g,t])
                    rhs = model.S[s,g,t]
        
        # unit capacity constraints
        for t in TIME:
            for g in GROUPS:
                for j in UNITS:
                    for i in I[j]:
                        model.cons.add(model.W[i,j,g,t]*Bmin[i,j] <= model.B[i,j,g,t])
                        model.cons.add(model.B[i,j,g,t] <= model.W[i,j,g,t]*Bmax[i,j])
        
        # unit mass balances
        for j in UNITS:
            rhs = 0
            for t in TIME:
                out = 0
                rhs += sum([model.B[i,j,g,t] for i in I[j] for g in GROUPS])
                out += sum([model.B[i,j,g,t] for i in I[j] for g in GROUPS])
                for i in I[j]:
                    for g in GROUPS:
                        for s in S_[i]:
                            if t >= pr[j]:
                                rhs -= rho_[(i,s)]*model.B[i,j,g,max(TIME[TIME <= t-pr[j]])]
                                #out += rho_[(i,s)]*model.B[i,j,g,max(TIME[TIME <= t-pr[j]])]
                model.cons.add(model.Q[j,t] == rhs)
                model.cons.add(model.O[j,t] == out)
                rhs = model.Q[j,t]
        # unit terminal condition
        #model.tc = Constraint(prensas, rule = lambda model, j: model.Q[j,H] == 0)
        for t in TIME:
            model.cons.add(sum([model.O[j,t] for j in pr_left]) <= sum([model.O[j,t] for j in pr_right]))
            

        return model, UNITS, I, TIME, pr, STATES

    def solve_maravelias_model(model):
        #solver = SolverFactory('cbc')
        solver = SolverFactory('cbc', executable="C:/Users/lucia/OneDrive/Escritorio/WineOP 1.0/Backend-Capstone-main/dssProject/Pyomo/CBC/bin/cbc")
        solver.options['sec'] = 60 * 60  # Time limit in seconds
        solver.options['ratioGap'] = 5  # Maximum gap of 5%
        solver.options['findInitial'] = True

        # Solve the model
        results = solver.solve(model)
        Condition = results.solver.termination_condition
        return Condition

    def unites(ORDERTASK):
        UnidadMedida = []
        for i in ORDERTASK:
            if i == 'Fermentación' or i == 'Pre-flotación' or i=='Flotación':
                UnidadMedida.append('\n[kLt]')
            else:
                UnidadMedida.append('\n[Ton]')
        UnidadMedida.append('\n[Ton]')
        return UnidadMedida

    def results_model(model,I,UNITS,GROUPS,TIME,pr):    
        RESULTS = []
        ORDERTASK = ['Despalillado','Prensado','Pre-flotación','Flotación','Fermentación']
        for m in ORDERTASK:
            results=[]
            if m == 'Fermentación' or m == 'Pre-flotación' or m=='Flotación':
                rho = 760/1000
            else:
                rho = 1
            result =[{'Machine': j,
                    'Block': int(g),
                    'Start': t,
                        'Duration': pr[j],
                        'Finish': t+pr[j],
                    'Batch': round(model.B[i,j,g,t]()*rho,2)}
                    for j in UNITS for i in I[j] for g in GROUPS for t in TIME if model.W[i,j,g,t]()>0 and i == m]
            RESULTS.append(result)
            
        return RESULTS,ORDERTASK

    def visualize(results,task,um,iteracion):
        schedule = pd.DataFrame(results)
        JOBS = sorted(list(schedule['Block'].unique()))
        MACHINES = sorted(list(schedule['Machine'].unique()))
        if task == 'fermentación':
            ferm = 24 * 10
        else:
            ferm = 1
        title='Bloque de\nmezcla'
        schedule['Finish'] = schedule['Start'] + schedule['Duration'] * ferm
        makespan = schedule['Finish'].max()
        minespan = schedule['Start'].min()
        bar_style = {'alpha': 1.0, 'lw': 25, 'solid_capstyle': 'butt'}
        text_style = {'color': 'white', 'weight': 'bold', 'ha': 'center', 'va': 'center', 'fontsize': 8}
        colors = ['#05851F', '#009FAD', '#8A3500', '#F9752D', '#D6C400', '#4356FA',
            '#B27BFA','#FA6F95','#FA3C2C','#44AD8A']
        title_style = {
        'fontsize': 16,
        'color': 'black',
        'weight': 'bold',
        'ha': 'center',
        'va': 'center',
        'path_effects': [withStroke(linewidth=3, foreground='white')]}
        fig, ax = plt.subplots(1, 1, figsize=(12, 5 + len(MACHINES) / 4))
        # Agregar una leyenda fuera del gráfico
        if task == 'recolección':
            title='Tipo de\ncamiones'
            colors = ['#E6170E','#5FB324','#B3A927','#1C91E6']
        for i in range(len(schedule)):
            xs = schedule['Start'][i]
            xf = schedule['Finish'][i]
            textstart = ""
            if task == 'fermentación':
                hstart = f"{(xs % 24):02d}:00"
                textstart = "Hora de inicio->"+hstart+" | "
                um = um.replace("\n", "")
            mdx = MACHINES.index(schedule['Machine'][i]) + 1
            ax.plot([xs, xf], [mdx] * 2, c=colors[schedule['Block'][i]-1], **bar_style)
            ax.text((xs + xf) / 2, mdx, textstart+str(schedule['Batch'][i]) + um, **text_style)
            ax.vlines(x=xf, ymin=mdx - 0.5, ymax=mdx + 0.5, color='white', linewidth=0.25)
        fecha_actual=datetime.now()
        fecha_actual=fecha_actual.strftime("%d/%m/%Y")
        ax.set_title('Planificación del proceso de '+task+'\n('+fecha_actual+')',**title_style)
        ax.set_ylabel('Máquinas')
        ax.set_ylim(0.5, len(MACHINES) + 0.5)
        ax.set_yticks(range(1, 1 + len(MACHINES)))
        ax.set_yticklabels(MACHINES)

        # Agregar horas del día en el eje x
        ax.set_xlabel('Tiempo (horas)')  # Etiqueta del eje x
        ax.set_xlim(minespan, makespan)  # Rango del eje x

        # Configurar las horas en el eje x
        # Puedes ajustar esto según sea necesario

        if task == 'fermentación':
            ax.set_xlabel('Tiempo (Días)')
            hours = range(int(minespan), int(makespan) + 1, 24)
            ax.set_xticks(hours)
            ax.set_xticklabels([f"Día {hour // 24 + 1}" for hour in hours], rotation=45, ha='right')
        else:
            hours = range(int(minespan), int(makespan) + 1, 2)
            ax.set_xticks(hours)
            ax.set_xticklabels([f"{(hour % 24):02d}:00" for hour in hours], rotation=45, ha='right')

        #ax.text(makespan, ax.get_ylim()[0] - 0.2, "{0:0.1f}".format(makespan), ha='center', va='top')
        ax.plot([makespan] * 2, ax.get_ylim(), 'r--')
        ax.grid(True)

        
        job_labels = [mpl.patches.Patch(color=colors[JOBS[i]-1], label=JOBS[i]) for i in range(len(JOBS))]
        ax.legend(handles=job_labels, bbox_to_anchor=(1, 1), loc='upper left', title=title)

        fig.tight_layout()
        ruta_deseada = os.path.join(os.path.dirname(__file__),'Files','{:02}-{}.pdf'.format(iteracion,task))
        plt.savefig(ruta_deseada)
        #plt.show()
        
    def collection_truck(df):
        # Lista de camiones
        camiones = {"C1": None, "C2": None, "C3": None, "C4": None}

        # Inicializar variables
        asignaciones = []  # Lista para almacenar las asignaciones
        toneladas_acumuladas = 0  # Toneladas acumuladas

        # Iterar sobre el DataFrame
        for i, toneladas in enumerate(df):
            toneladas_acumuladas += toneladas

            if toneladas_acumuladas >= 30 and any(disponibilidad is None for disponibilidad in camiones.values()):
                # Si hay al menos 30 toneladas acumuladas y hay camiones disponibles, asignar uno nuevo
                camion_disponible = next((camion for camion, disponibilidad in camiones.items() if disponibilidad is None), None)
                if camion_disponible:
                    indice = next((indice for indice, (camion, disponibilidad) in enumerate(camiones.items()) if camion == camion_disponible), None)
                    # Asignar el camión disponible
                    asignaciones.append({'Machine': camion_disponible, 'Block': indice+1, 'Start': i, 'Duration':8,
                                        'Finish': i + 8, 'Batch': round(toneladas_acumuladas,1)})
                    toneladas_acumuladas = 0  # Reiniciar las toneladas acumuladas
                    camiones[camion_disponible] = {'inicio': i, 'fin': i + 8}

            # Verificar si algún camión ha regresado y está disponible para una nueva asignación
            for camion, disponibilidad in camiones.items():
                if disponibilidad and i >= disponibilidad['fin']:
                    camiones[camion] = None  # Marcar el camión como disponible

        df_asignaciones = pd.DataFrame(asignaciones)
        return df_asignaciones, toneladas_acumuladas

    def merge_pdfs(input_folder, output_file):
        pdf_merger = PyPDF2.PdfMerger()

        # Obtener la lista de archivos PDF en el directorio de entrada
        pdf_files = [f for f in os.listdir(input_folder) if f.endswith('.pdf')]

        # Ordenar los archivos PDF por nombre
        pdf_files.sort()

        # Agregar cada archivo PDF al objeto PdfMerger
        for pdf_file in pdf_files:
            pdf_path = os.path.join(input_folder, pdf_file)
            pdf_merger.append(pdf_path)

        # Guardar el archivo combinado
        with open(output_file, 'wb') as output_pdf:
            pdf_merger.write(output_pdf)
        

    def deletepdf():
        # Ruta de la carpeta que contiene los archivos PDF
        carpeta = os.path.join(os.path.dirname(__file__),'Files')
        
        # Patrón para buscar archivos PDF
        patron_pdf = '*.pdf'
        
        # Crear el camino completo al patrón de archivos PDF
        ruta_completa = os.path.join(carpeta, patron_pdf)
        
        # Obtener la lista de archivos PDF en la carpeta
        archivos_pdf = glob.glob(ruta_completa)
        
        # Eliminar cada archivo PDF
        for archivo_pdf in archivos_pdf:
            try:
                os.remove(archivo_pdf)
                #print(f"Archivo eliminado: {archivo_pdf}")
            except Exception as e:
                print(f"No se pudo eliminar {archivo_pdf}: {e}")



    rho_convert = {'Despalillado': 1.0,
                    'Prensado': 1.0,
                    'Pre-flotación': 1/760,
                    'Flotación': 1/760,
                    'Fermentación': 1/760}

    machines_file_path = './dssProject/Modelo_matemático/settings.xlsx' 
    TotalProc = [[],[],[]]#desp,pren,prefl,flt,ferm
    # Lista para almacenar las rutas completas de los archivos encontrados
    # Buscar recursivamente el archivo en la ruta inicial y sus subdirectorios
    input_info=pd.read_excel('input.xlsx')
    GROUPS, Recieve = read_input_file(input_info)
    ReqTon = Recieve['Kilos'].sum()

    UNITS_TASKS = preprocess_machines_data(machines_file_path)
    model, UNITS, I, TIME, tau, STATES = create_maravelias_model(GROUPS,UNITS_TASKS,Recieve)
    start_time = time()
    condition = solve_maravelias_model(model)
    end_time = time()
    execute_time = end_time-start_time
    dicc_results, TASKS = results_model(model,I,UNITS,GROUPS,TIME, tau)
    um = unites(TASKS)
    Inv = {(s,t): sum([model.S[s,g,t]() for g in GROUPS]) for s in STATES for t in TIME}
    states_time=pd.DataFrame([[Inv[s,t] for s in STATES.keys()] for t in TIME], columns = STATES.keys(), index = TIME)
    states_time['Desechos'] = states_time['Escobajo']+states_time['Orujo']
    states_time['Desechos'] = states_time['Desechos'].diff()
    states_time.loc[0,'Desechos']=0
    schedl_truck, pending = collection_truck(states_time['Desechos'])
    dicc_results.append(schedl_truck)
    TASKS.append('Recolección')
    for idx in range(len(TASKS)):
        total=0
        if len(dicc_results[idx]) > 0 and condition == 'optimal':
            total=pd.DataFrame(dicc_results[idx])['Batch'].sum()
            visualize(dicc_results[idx], TASKS[idx].lower(), um[idx], idx+1)
        TotalProc[0].append(TASKS[idx])
        TotalProc[1].append(total)
        TotalProc[2].append(um[idx])

    #generar pdf
    fecha_actual=datetime.now().strftime("%d-%m-%Y")
    # Ejemplo de uso
    input_folder = os.path.join(os.path.dirname(__file__),'Files')#Directorio que contiene los archivos PDF a combinar
    output_file =  os.path.join(os.path.dirname(__file__),'Files','output',"Planificacion_"+fecha_actual+".pdf" )
    merge_pdfs(input_folder, output_file)
    deletepdf()

    result =[{'Proceso':  TotalProc[0][i],
                'Cantidad procesada':  TotalProc[1][i],
                'Unidad de medida':  TotalProc[2][i]}
                for i in range(len(TotalProc[0]))]
    result = pd.DataFrame(result)
    ruta_deseada=os.path.join(os.path.dirname(__file__), 'Files',"Resumen.xlsx")
    result_testeo = result.to_excel(ruta_deseada, index=False)
    
    print(model.Value())
    return ("hola")