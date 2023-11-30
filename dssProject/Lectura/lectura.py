import pdfplumber
import pandas as pd
from django.http import HttpResponse


def lectura_archivos(myfile, mypdf):
    def cleaning_pdf(df):
        df['Productor'] = df['Productor'].astype(str)
        df['Dia/Mes'] = df['Dia/Mes'].astype(str)
        df['Productor'] = df['Productor'].replace('None', method='ffill')
        df['Dia/Mes'] = df['Dia/Mes'].replace('None', method='ffill')
        df = df.dropna(subset=['Link GPS'])
        df['Contrato'] = df['Contrato'].astype(str)
        df['Variedad'] = df['Variedad'].astype(str)
        df['Kilos'] = df['Kilos'].str.replace('.', '').astype(int)
        df['Dia/Mes'] = pd.to_datetime(df['Dia/Mes'], format='%d/%m').copy()
        # Formatear la columna 'Dia/Mes' para mostrar solo 'dd/mm'
        df['Dia/Mes'] = df['Dia/Mes'].dt.strftime('%d/%m').copy()
        df = df.reset_index(drop=True)
        return df
    def table_to_df(table):
        return pd.DataFrame(table[1:], columns=table[0])
    def read_pdf(pdf_file):
    # Lista para almacenar los DataFrames de cada página
        dfs = []
        # Iterar a través de las páginas
        for page in pdf_file.pages:
            # Convertir la tabla de la página a DataFrame
            df_page = table_to_df(page.extract_table())

            # Agregar el DataFrame al final de la lista
            dfs.append(df_page)

        # Cerrar el archivo PDF después de haber extraído todas las tablas
        pdf_file.close()

        # Concatenar todos los DataFrames en uno solo
        result_df = pd.concat(dfs, ignore_index=True)
        result_df.columns = ['Dia/Mes','Productor','Fundo','Contrato','Variedad',
                            'Kilos','N Viaje','Patente','Hr Fundo','Hr Bodega',
                            'Link GPS']
        # Hacer limpieza de datos

        return cleaning_pdf(result_df)

    def cleaning_vendimia():
        # Encontrar el índice de la columna 'Programado'
        df = pd.read_excel(myfile,skiprows=4)
        programado_index = df.columns.get_loc('PROGRAMADO')

        # Obtener las columnas de fechas que están después de 'Programado'
        date_columns = df.columns[programado_index + 1:]

        # Crear un diccionario para mapear nombres de columnas
        column_mapping = {}
        for col in date_columns:
            new_col = pd.to_datetime(col).strftime('%d/%m')  # Nuevo nombre de columna formateado
            column_mapping[col] = str(new_col)
        df['CTTO'] = df['CTTO'].astype(str)
        # Renombrar las columnas usando el diccionario
        df = df.rename(columns=column_mapping)
        return df

    def bridge_inputs(df_enologo,df_logistica):
        df_logistica['Bloque']=0
        for i in range(len(df_logistica)):
            numero_ctto = df_logistica['Contrato'][i]
            #fecha_dia_mes = str(df_logistica['Dia/Mes'][i])
            variedad = df_logistica['Variedad'][i].split()[0]
            #filtro = (df_enologo['CTTO'] == numero_ctto) & (df_enologo['VARIEDAD'].str.startswith(variedad)) & (df_enologo[fecha_dia_mes].notna())
            filtro = (df_enologo['CTTO'] == numero_ctto) & (df_enologo['VARIEDAD'].str.startswith(variedad))
            resultado = df_enologo[filtro]
            # Obtener el valor de la columna 'BLOQUE' para la fila resultante
            if not resultado.empty:
                bloque = resultado['BLOQUE'].values[0]
                #print(f"El valor de 'BLOQUE' para CTTO {numero_ctto}, fecha {fecha_dia_mes}, y variedad {variedad} es: {bloque}")
            else:
                bloque = 0
            df_logistica['Bloque'][i] = bloque
        return df_logistica

    def input_model(df):
        df = df[['Hr Bodega', 'Bloque','Kilos']]
        df = df.groupby(['Hr Bodega', 'Bloque'])['Kilos'].sum().reset_index()
        df['Hr Bodega'] = pd.to_datetime(df['Hr Bodega'], format='%H:%M').dt.time
        df['Hr Bodega'] = df['Hr Bodega'].apply(lambda x: round(x.hour + x.minute / 60.0,1))
        df = df.sort_values(by='Hr Bodega').reset_index(drop=True)
        df = df.dropna().reset_index(drop=True)
        df['Bloque'] = df['Bloque'].astype(int)
        df = df[df['Bloque'] != 0]
        return df

    #import os
    vendimia_clean = cleaning_vendimia()
    # initial_path="C:/Users/Usuario/Documents/Capstone 2023/Testeo-prototipo/Prototipo 01 - WineOptimize/inputs_pdf"
    # # Lista para almacenar las rutas completas de los archivos encontrados
    # # Buscar recursivamente el archivo en la ruta inicial y sus subdirectorios
    # for carpeta_actual, _, archivos in os.walk(initial_path):
    #     for i in archivos:
    #         try: 
    #             ruta_completa = os.path.join(carpeta_actual, i)
    #             pdf = pdfplumber.open(ruta_completa)
    #             input_bodega = read_pdf(pdf)
    #             print(i[:-4])
    #             merge_inputs = bridge_inputs(vendimia_clean,input_bodega)
    #             input_final = input_model(merge_inputs)
    #         except Exception as e:
    #             print(i+"-> ERROR",e)
    #             break

    # vendimia_clean = cleaning_vendimia()
    pdf = pdfplumber.open(mypdf)
    input_bodega = read_pdf(pdf)
    merge_inputs = bridge_inputs(vendimia_clean,input_bodega)
    merge_inputs.to_excel("Info-Día.xlsx", index=False)
    input_final = input_model(merge_inputs)
    # # Especifica el nombre del archivo Excel
    nombre_archivo_excel = 'input.xlsx'

    # # Exporta el DataFrame a Excel
    input=input_final.to_excel(nombre_archivo_excel, index=False)
    # merge_inputs.to_excel('Planificacion_bloque.xlsx', index=False)

    return input

