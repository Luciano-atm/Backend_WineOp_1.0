import tabula
from tabula.io import read_pdf
import pandas as pd
from datetime import datetime
from django.http import HttpResponse


def lectura_archivos(myfile, mypdf):
    # Cargar el archivo Excel en un DataFrame
    dataframe_xlsx = pd.read_excel(myfile)

    # Guardar la fecha entregada en la primera hoja
    df = tabula.read_pdf(mypdf, pages='all')[0]
    fecha= df.iloc[0]['Dia/Mes']

    # Guardar el ultimo valor de productor
    ultimo_valor_productor = ""




    def buscar_fecha_en_fila3(dataframe_xlsx, fecha_buscada):
        # Obtener la fila 3 de dataframe_xlsx y convertir todos los valores a strings
        fila_3 = dataframe_xlsx.iloc[3].astype(str)

        # Función para formatear la fecha
        def formatear_fecha(valor):
            try:
                fecha = datetime.strptime(valor, "%Y-%m-%d %H:%M:%S")
                return fecha.strftime("%m-%d")
            except ValueError:
                return valor

        # Modificar los valores de la fila 3 para tomar solo la parte "05-01"
        fila_3 = fila_3.apply(formatear_fecha)

        # Buscar la fecha en la fila 3 y determinar la columna
        columna_encontrada = None
        for columna, valor in fila_3.items():
            if valor == fecha_buscada:
                columna_encontrada = columna
                break

        return columna_encontrada

    # Función para limpiar una página del PDF (páginas impares)

    def limpiar_pagina_columna(df,ultimo_valor_productor):
        # Cambiar el nombre de la columna 'Fundo Contrato' a 'Fundo'
        df.rename(columns={'Fundo Contrato': 'Fundo'}, inplace=True)



        
        # Función para eliminar filas donde la columna "Fundo" contiene la palabra "total"
        def eliminar_filas_con_total(df):
            df = df[~df['Fundo'].str.contains('TOTAL', case=False)]
            return df

        # Llamar a la función para eliminar las filas
        df = eliminar_filas_con_total(df)

        # Cambiar el nombre de la columna 'Unnamed: 0' a 'Contrato'
        df.rename(columns={'Unnamed: 0': 'Contrato'}, inplace=True)

        # Eliminar la columna " N Viaje"
        df.drop("N Viaje", axis=1, inplace=True)

        if not df.empty:
            # Función para repetir la primera fecha en todas las filas
            df['Dia/Mes'] = fecha

            # Convertir la columna "Kilos" a cadena de texto y mantener solo la primera palabra
            df["Kilos"] = df["Kilos"].astype(str).str.split().str[0]

            primer_valor_productor = df["Productor"].iloc[0]
            # Verificar si el primer valor es None o NaN
            if primer_valor_productor is None or pd.isna(primer_valor_productor):
                # Si el primer valor de "Productor" es None o NaN, reemplazarlo con "ultimo_valor_productor"
                df["Productor"].iloc[0] = ultimo_valor_productor

            # Rellenar NaN en la columna 'Productor' con los valores de la fila superior
            df['Productor'].fillna(method='ffill', inplace=True)

            # Verificar si la última palabra de cada valor en 'Fundo' es un entero
            ultima_palabra_entero = df['Fundo'].str.split().str[-1].str.isdigit()

            # Usar la máscara booleana para condicionar la modificación de las columnas
            df.loc[ultima_palabra_entero, 'Contrato'] = df.loc[ultima_palabra_entero, 'Fundo'].str.split().str[-1]
            df.loc[ultima_palabra_entero, 'Fundo'] = df.loc[ultima_palabra_entero, 'Fundo'].str.rsplit(' ', n=1).str[0]


            # Convertir la columna 'Contrato' a cadenas (strings) y eliminar ".0"
            df['Contrato'] = df['Contrato'].astype(str).str.replace('\.0', '', regex=True)




            def convertir_dia_mes(valor):
                if valor is not None and '/' in valor:
                    partes = valor.split('/')
                    if len(partes) == 2:
                        dia, mes = partes
                        dia = dia.zfill(2)
                        mes = mes.zfill(2)
                        return f"{mes}-{dia}"
                return valor

            df["Dia/Mes"] = df["Dia/Mes"].apply(convertir_dia_mes)
            
            # Función para obtener el bloque
            def obtener_bloque(row, dataframe_xlsx):
                contrato = str(row["Contrato"])  # Convertir a cadena (string)
                filtro = dataframe_xlsx[dataframe_xlsx["Unnamed: 0"].astype(str) == contrato]
                bloque = None  # Inicializamos el bloque como None por defecto
                if not filtro.empty:
                    # Obtener la primera palabra de la columna "Variedad" de df
                    primera_palabra_variedad = row["Variedad"].split()[0]
                    
                    for idx, fila in filtro.iterrows():
                        # Obtener la primera palabra de la columna "Unnamed: 2" para cada fila
                        primera_palabra_unnamed2 = fila["Unnamed: 2"].split()[0]

                        # Verificar si las primeras palabras son iguales
                        if primera_palabra_variedad == primera_palabra_unnamed2:
                            fecha_buscada = row["Dia/Mes"]
                            columna_fecha = buscar_fecha_en_fila3(dataframe_xlsx, fecha_buscada)
                            if columna_fecha is not None:
                                valor_fecha = fila[columna_fecha]
                                if not pd.isna(valor_fecha):
                                    bloque = fila["Unnamed: 4"]
                                    return bloque

                return bloque

            # Llamar a la función con los parámetros apropiados
            df["Bloque"] = df.apply(obtener_bloque, args=(dataframe_xlsx,), axis=1)
            df['Bloque'] = df['Bloque'].astype(str).str.replace('\.0', '', regex=True)

            ultimo_valor_productor = df["Productor"].iloc[-1]

            
            # Llamar a la función para eliminar las filas
            df = eliminar_filas_con_total(df) 

        return df,ultimo_valor_productor

    # Función para limpiar una página del PDF (páginas pares)
    def limpiar_pagina_corridas(df,ultimo_valor_productor):
        # Eliminar las filas donde la columna 'Dia mes' comience con la palabra 'Total' o no sea de tipo string
        df = df[~(df['Dia/Mes'].str.startswith('TOTAL') | ~df['Productor'].apply(lambda x: isinstance(x, str)))]

        # Mover todos los valores una columna a la derecha
        df = df.shift(axis=1)

        # Eliminar la columna " N Viaje"
        df.drop("N Viaje", axis=1, inplace=True)

        if not df.empty:
            
            # Identificar filas donde 'Fundo' es un número o una cadena
            is_numeric_fundo = df['Fundo'].apply(lambda x: isinstance(x, int) or (isinstance(x, str) and x.isdigit()))
            
            # Mover toda la fila una columna a la derecha si 'Fundo' contiene un número o una cadena
            df.loc[is_numeric_fundo] = df.loc[is_numeric_fundo].shift(axis=1)
            
            # Convertir la columna "Kilos" a cadena de texto y mantener solo la primera palabra
            df["Kilos"] = df["Kilos"].astype(str).str.split().str[0]

            # Verificar si el primer valor es None o NaN
            primer_valor_productor = df["Productor"].iloc[0]
            if primer_valor_productor is None or pd.isna(primer_valor_productor):
                # Si el primer valor de "Productor" es None o NaN, reemplazarlo con "ultimo_valor_productor"
                df["Productor"].iloc[0] = ultimo_valor_productor
                
            # Copiar el valor que se encuentra arriba en la columna 'Productor' cuando tenga NaN
            df['Productor'].fillna(method='ffill', inplace=True)

            # Convertir la columna 'Contrato' a cadenas (strings) y eliminar ".0"
            df['Contrato'] = df['Contrato'].astype(str).str.replace('\.0', '', regex=True)
            
            df['Dia/Mes'] = fecha





            def convertir_dia_mes(valor):
                if valor is not None and '/' in valor:
                    partes = valor.split('/')
                    if len(partes) == 2:
                        dia, mes = partes
                        dia = dia.zfill(2)
                        mes = mes.zfill(2)
                        return f"{mes}-{dia}"
                return valor

            df["Dia/Mes"] = df["Dia/Mes"].apply(convertir_dia_mes)
            
            # Función para obtener el bloque
            def obtener_bloque(row, dataframe_xlsx):
                contrato = str(row["Contrato"])  # Convertir a cadena (string)
                filtro = dataframe_xlsx[dataframe_xlsx["Unnamed: 0"].astype(str) == contrato]
                bloque = None  # Inicializamos el bloque como None por defecto
                if not filtro.empty:
                    # Obtener la primera palabra de la columna "Variedad" de df
                    primera_palabra_variedad = row["Variedad"].split()[0]
                    
                    for idx, fila in filtro.iterrows():
                        # Obtener la primera palabra de la columna "Unnamed: 2" para cada fila
                        primera_palabra_unnamed2 = fila["Unnamed: 2"].split()[0]

                        # Verificar si las primeras palabras son iguales
                        if primera_palabra_variedad == primera_palabra_unnamed2:
                            fecha_buscada = row["Dia/Mes"]
                            columna_fecha = buscar_fecha_en_fila3(dataframe_xlsx, fecha_buscada)
                            if columna_fecha is not None:
                                valor_fecha = fila[columna_fecha]
                                if not pd.isna(valor_fecha):
                                    bloque = fila["Unnamed: 4"]
                                    return bloque

                return bloque

            # Llamar a la función con los parámetros apropiados
            df["Bloque"] = df.apply(obtener_bloque, args=(dataframe_xlsx,), axis=1)
            df['Bloque'] = df['Bloque'].astype(str).str.replace('\.0', '', regex=True)



            ultimo_valor_productor = df["Productor"].iloc[-1]

          
           
            df = df[~df['Fundo'].str.contains('TOTAL', case=False)]
           
        
        
        return df,ultimo_valor_productor



    # Leer el archivo PDF y procesar todas las páginas
    pages = tabula.read_pdf(mypdf, pages='all')

    # Inicializar una lista para almacenar los DataFrames procesados de cada página
    dfs = []


    for i, df in enumerate(pages):
        
        print("Página antes de la limpieza:  ")
        print(df)  # Imprimir la página antes de la limpieza
        print("-" * 40)


        if "Unnamed: 1" in df.columns:
            df = df.drop('Unnamed: 1', axis=1)
            df = df.rename(columns={'Unnamed: 0': 'aux1'})
            df = df.rename(columns={'Fundo Contrato': 'aux2'})
            df = df.rename(columns={'aux1': 'Fundo Contrato'})
            df = df.rename(columns={'aux2': 'Unnamed: 0'})
            resultado = limpiar_pagina_columna(df,ultimo_valor_productor)
            df_cleaned = resultado[0]
            ultimo_valor_productor = resultado[1]

        elif "Unnamed: 0" in df.columns:
            resultado = limpiar_pagina_columna(df,ultimo_valor_productor)
            df_cleaned = resultado[0]
            ultimo_valor_productor = resultado[1]

        else: 
            primer_valor_fundo = df["Fundo"].iloc[0]
            primer_valor_productor = df["Productor"].iloc[0]
            ultimo_valor_productor = resultado[1]


            if str(primer_valor_fundo).isnumeric():
                resultado = limpiar_pagina_corridas(df,ultimo_valor_productor)
                df_cleaned = resultado[0]
                ultimo_valor_productor = resultado[1]

            elif str(primer_valor_productor).isnumeric():
                df = df.shift(axis=1)
                resultado = limpiar_pagina_corridas(df,ultimo_valor_productor)
                df_cleaned = resultado[0]
                ultimo_valor_productor = resultado[1]


        

        dfs.append(df_cleaned)
        print("Página después de la limpieza:  ")
        print(df_cleaned)  # Imprimir la página después de la limpieza
        print("=" * 40)

    # Concatenar todos los DataFrames en uno solo
    result_df = pd.concat(dfs, ignore_index=True)

    # Guardar el DataFrame modificado en un archivo CSV
    archivo_resultante= result_df.to_excel("Info-Día.xlsx", index=False)

    # Imprimir el DataFrame modificado
    #print(result_df)

    #Se agrupan por hora y bloque


    result_df['Kilos'] = result_df['Kilos'].astype(float)
    sum_df = result_df.groupby(['Hr Bodega', 'Bloque'])['Kilos'].sum().reset_index()
    sum_df['Kilos'] = sum_df['Kilos'].astype(int)

    def convertir_hora_a_decimal(hora_str):
        horas, minutos = map(int, hora_str.split(":"))
        return horas + minutos / 60.0

    sum_df['Hr Bodega'] = sum_df['Hr Bodega'].apply(convertir_hora_a_decimal)

    # Filtrar los datos eliminando filas con valores NaN en la columna 'Bloque'
    sum_df = sum_df.dropna(subset=['Bloque'])
    sum_df = sum_df[sum_df['Bloque'] != 'nan']

    archivo_suma = sum_df.to_excel("input.xlsx", index=False)

    

    #Los nan 
    #Se obtiene los bloque agrupago con nan en bloque
    #print("----------------------------------------")
    is_not = sum_df.loc[:, 'Bloque'] != 'nan'
    #print(sum_df.loc[is_nan])
    #print("----------------------------------------")
    #Se obtienen los datos con nan en bloque
    is_nan =result_df.loc[:, 'Bloque'] == 'nan'
    
    #print(result_df.loc[is_nan])
    #print("++++++++++++++++++++++++++++++++++")
    #print(sum_df)


    return archivo_resultante

