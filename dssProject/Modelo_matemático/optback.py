import pandas as pd
from pyomo.environ import *
from time import time
import os
import numpy as np
from dssProject.Pyomo.CBC.bin import *

def modelo():
    def search_rute(file_path):
        # Ruta del archivo actual
        nombre_archivo = file_path
        ruta_actual = os.getcwd()

        # Buscar recursivamente el archivo en la ruta actual y sus subdirectorios
        for carpeta_actual, carpetas, archivos in os.walk(ruta_actual):
            if nombre_archivo in archivos:
                ruta_completa = os.path.join(carpeta_actual, nombre_archivo)
                return ruta_completa
                break
        else:
            return ""
        
        
    def read_input_file(file_path):
        input = pd.read_excel(file_path)
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

    def preprocess_machines_data(file_path, ReqCfer):
        df = pd.read_excel(file_path)
        df.loc[len(df)-1, 'Cantidad'] = ReqCfer
        UNITS = []
        TASKS = []
        BMIN = []
        BMAX = []
        TAU = []
        PROC = []
        
        pozo = 0
        pr = 0
        cpf = 0
        flt = 0
        cf = 0
        for i in range (len(df)):
            for j in range(df['Cantidad'][i]):
                TASKS.append(df['Tarea'][i])
                task = df['Tarea'][i]
                BMIN.append(round(df['Bmin'][i]*rho_convert[task],0))
                BMAX.append(round(df['Bmax'][i]*rho_convert[task],0))
                TAU.append(df['Tau'][i])
                PROC.append(df['Proc'][i])
                if df['Maquinas'][i].startswith('Pozo'):
                    pozo+=1
                    UNITS.append(df['Maquinas'][i])
                elif df['Maquinas'][i].startswith('Prens'):
                    pr+=1
                    UNITS.append(df['Maquinas'][i])
                elif df['Maquinas'][i].startswith('CPF'):
                    cpf+=1
                    UNITS.append('CubaF_{:02}'.format(cpf))
                elif df['Maquinas'][i].startswith('FLT'):
                    flt+=1
                    UNITS.append('Flotador_{:02}'.format(flt))
                elif df['Maquinas'][i].startswith('CF'):
                    cf+=1
                    UNITS.append('CubaF_{:02}'.format(cf))
        
        UNITS_TASKS = {(UNIT,TASK): {'Bmin': BMIN, 'Bmax': BMAX, 'Proc': PROC, 'Tau': TAU, 'Cost':1, 'Tclean':0} for (UNIT,TASK,BMIN,BMAX,TAU,PROC) in zip(UNITS,TASKS,BMIN,BMAX,TAU,PROC)}

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
        
        model = ConcreteModel()

        model.W = Var(TASKS, UNITS, GROUPS, TIME, domain=Boolean)
        model.B = Var(TASKS, UNITS, GROUPS,TIME, domain=NonNegativeReals)
        model.S = Var(STATES.keys(), GROUPS, TIME, domain=NonNegativeReals)
        
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
            

        return model, TASKS, UNITS, I

    def solve_maravelias_model(model):
        solver = SolverFactory('cbc',executable='./dssProject/Pyomo/CBC/bin/cbc')
        solver.options['sec'] = 60 * 30  # Time limit in seconds
        solver.options['ratioGap'] = 5  # Maximum gap of 5%
        solver.options['findInitial'] = True

        # Solve the model
        results = solver.solve(model)
        Condition = results.solver.termination_condition

    # Main Code
    input_file_path = './input.xlsx'
    machines_file_path = './dssProject/Modelo_matemático/Machines.xlsx' 
    GROUPS, Recieve = read_input_file(input_file_path)

    rho_convert = {'Despalillado': 1.0,
                    'Prensado': 1.0,
                    'Pre-flotación': 1/760,
                    'Flotación': 1/760,
                    'Fermentación': 1/760}

    ReqTon = Recieve['Kilos'].sum()
    ReqCfer = round(ReqTon * 0.96 * 0.84 * 0.94 * 760 / 25000, 0)

    UNITS_TASKS = preprocess_machines_data(machines_file_path, ReqCfer)
    #
    model, TASKS, UNITS, I = create_maravelias_model(GROUPS,UNITS_TASKS,Recieve)
    start_time = time()
    solve_maravelias_model(model)
    end_time = time()

    print(model.Value())

    return ("hola")