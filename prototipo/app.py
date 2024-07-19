from flask import Flask, render_template
import pandas as pd
import re
import xgboost as xgb
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os

app = Flask(__name__)

# Funciones de cálculo
def calcular_porcentaje(total_kWhD, total_kVarhD):
    """Calcula el porcentaje de energía reactiva respecto a la energía activa."""
    if total_kWhD == 0:
        raise ValueError("El total de energía activa es 0, no se puede calcular el porcentaje.")
    return (total_kVarhD / total_kWhD) * 100

def calcular_totales(df):
    """Calcula los totales de las columnas 'kWhD' y 'kVarhD'."""
    total_kWhD = df[df['VARIABLE'] == 'kWhD']['USAGE_DATA'].sum()
    total_kVarhD = df[df['VARIABLE'] == 'kVarhD']['USAGE_DATA'].sum()
    return total_kWhD, total_kVarhD

def verificar_supera_umbral(porcentaje, umbral=50):
    """Verifica si el porcentaje de energía reactiva supera un umbral dado."""
    return porcentaje > umbral

def visualizar_datos(df, cliente, max_bars=100):
    """Genera un gráfico de barras comparando las energías activa y reactiva para un cliente específico, limitado a un máximo de barras."""
    client_data = df[df['CLIENTE'] == cliente]
    df_pivot = client_data.pivot_table(index='DATETIME', columns='VARIABLE', values='USAGE_DATA', aggfunc='sum').fillna(0)
    if len(df_pivot) > max_bars:
        df_pivot = df_pivot.head(max_bars)
    ax = df_pivot.plot(kind='bar', figsize=(10, 6))
    ax.set_xlabel('Periodo')
    ax.set_ylabel('Energía')
    ax.set_title(f'Comparación de Energía Activa y Reactiva para Cliente {cliente}')
    
    # Crear directorio para almacenar los gráficos si no existe
    plot_dir = os.path.join('static', 'plots')
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    
    plot_path = os.path.join(plot_dir, f'bar_plot_{cliente}.png')
    plt.savefig(plot_path)
    plt.close()
    
    return f'plots/bar_plot_{cliente}.png'

@app.route('/')
def index():
    # Cargar datos
    fileHistoricOne = pd.read_csv('load/Report Historic - Ene-Feb-Mar-2023.csv')
    fileHistoricTwo = pd.read_csv('load/Report Historic - Abr-May-Jun 2023.csv')
    fileHistoricThre = pd.read_csv('load/Report Historic - Jul-Ago-Sep 2023.csv')
    fileHistoricFour = pd.read_csv('load/Report Historic - Oct-Nov-Dic 2023.csv')
    fileHistoricFive = pd.read_csv('load/Report Historic - Ene-Feb-2024.csv')

    # Concatenar los dos DataFrames
    fileHistoric = pd.concat([fileHistoricOne, fileHistoricTwo, fileHistoricThre, fileHistoricFour, fileHistoricFive], ignore_index=True)

    # Eliminar columnas innecesarias
    columns_to_drop = ['CODIGO SIC', 'MEDIDOR', 'YEARX', 'MONX', 'DAYX', 'HOURX']
    fileHistoric = fileHistoric.drop(columns=columns_to_drop)

    # Convertir la columna DATEX1 a tipo fecha (date)
    fileHistoric['DATEX1'] = pd.to_datetime(fileHistoric['DATEX1'], format='%d/%m/%Y').dt.date

    # Convertir la columna CLIENTE a tipo categoría
    fileHistoric['CLIENTE'] = fileHistoric['CLIENTE'].astype('category')
    fileHistoric['CLIENTE_CODE'] = fileHistoric['CLIENTE'].cat.codes + 1

    # Convertir la columna DATETIME a tipo datetime
    fileHistoric['DATETIME'] = pd.to_datetime(fileHistoric['DATETIME'], format='%m/%d/%Y %H:%M', errors='coerce')

    # Filtrar filas para mantener solo las variables 'kVarhD' y 'kWhD'
    fileHistoric = fileHistoric[fileHistoric['VARIABLE'].isin(['kVarhD', 'kWhD'])]

    # Función para limpiar y convertir a float
    def clean_and_convert(value):
        cleaned_value = re.sub(r'[^\d.]', '', str(value))
        try:
            return float(cleaned_value)
        except ValueError:
            return None

    fileHistoric['USAGE_DATA'] = fileHistoric['USAGE_DATA'].apply(clean_and_convert)

    # Convertir la columna VARIABLE a tipo categoría
    fileHistoric['VARIABLE'] = fileHistoric['VARIABLE'].astype('category')
    fileHistoric['VARIABLE_CODE'] = fileHistoric['VARIABLE'].cat.codes + 1

    # Entrenamiento de modelos
    models = {}
    for client in fileHistoric['CLIENTE'].unique():
        client_data = fileHistoric[fileHistoric['CLIENTE'] == client]
        X = client_data.drop(columns=['USAGE_DATA', 'DATEX1', 'DATETIME', 'CLIENTE', 'CLIENTE_CODE'])
        y = client_data['USAGE_DATA']

        # Convertir la columna 'VARIABLE' a tipo numérico (por ejemplo, categoría)
        X['VARIABLE'] = X['VARIABLE'].astype('category').cat.codes

        # Dividir el conjunto de datos en entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Crear DMatrix de XGBoost, habilitando el manejo de variables categóricas
        dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
        dtest = xgb.DMatrix(X_test, label=y_test, enable_categorical=True)

        # Definir los parámetros del modelo
        params = {
            'objective': 'reg:squarederror',
            'max_depth': 50,
            'eta': 0.5,
            'eval_metric': 'rmse'
        }

        # Entrenar el modelo
        bst = xgb.train(params, dtrain, num_boost_round=100)

        # Guardar el modelo en el diccionario
        models[client] = bst

    # Predicciones y gráficos
    plots = []
    for client in fileHistoric['CLIENTE'].unique():
        client_data = fileHistoric[fileHistoric['CLIENTE'] == client].copy()  # Utilizar una copia para evitar modificaciones en el original
        X = client_data.drop(columns=['USAGE_DATA', 'DATEX1', 'DATETIME', 'CLIENTE', 'CLIENTE_CODE', 'PREDICTION'], errors='ignore')

        # Convertir la columna 'VARIABLE' a tipo numérico
        X['VARIABLE'] = X['VARIABLE'].astype('category').cat.codes

        # Crear DMatrix de XGBoost para predicción
        dpredict = xgb.DMatrix(X, enable_categorical=True)

        # Realizar la predicción con el modelo entrenado
        y_pred = models[client].predict(dpredict)

        # Agregar las predicciones al DataFrame original
        fileHistoric.loc[fileHistoric['CLIENTE'] == client, 'PREDICTION'] = y_pred

        # Gráficos para cada cliente
        for variable in ['kVarhD', 'kWhD']:
            data = fileHistoric[(fileHistoric['CLIENTE'] == client) & (fileHistoric['VARIABLE'] == variable)]

            # Crear directorio para almacenar los gráficos si no existe
            plot_dir = os.path.join('static', 'plots')
            if not os.path.exists(plot_dir):
                os.makedirs(plot_dir)

            # Gráfica de línea
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(data['DATETIME'], data['USAGE_DATA'], label='Actual')
            ax.plot(data['DATETIME'], data['PREDICTION'], linestyle='--', label='Predicción')
            ax.set_xlabel('Fecha')
            ax.set_ylabel(f'Uso de Energía {variable}')
            ax.set_title(f'Cliente {client} - Uso de Energía {variable}')
            ax.legend()
            plot_path = os.path.join(plot_dir, f'client_{client}_{variable}.png')
            plt.savefig(plot_path)
            plt.close()

            # Guardar solo el nombre del archivo en la ruta de la gráfica generada
            plots.append({'client': client, 'variable': variable, 'plot_path': f'plots/client_{client}_{variable}.png'})

    
    # Inicializar diccionario para almacenar resultados por cliente
    clientes_info = []

    # Verificar para cada cliente si supera el umbral
    for cliente in fileHistoric['CLIENTE'].unique():
        client_data = fileHistoric[fileHistoric['CLIENTE'] == cliente]
        total_kWhD_cliente = client_data[client_data['VARIABLE'] == 'kWhD']['USAGE_DATA'].sum()
        total_kVarhD_cliente = client_data[client_data['VARIABLE'] == 'kVarhD']['USAGE_DATA'].sum()
        porcentaje_cliente = calcular_porcentaje(total_kWhD_cliente, total_kVarhD_cliente)
        supera_umbral_cliente = verificar_supera_umbral(porcentaje_cliente)
        clientes_info.append({
            'cliente': cliente,
            'total_kWhD': total_kWhD_cliente,
            'total_kVarhD': total_kVarhD_cliente,
            'porcentaje': porcentaje_cliente,
            'supera_umbral': supera_umbral_cliente
        })

    # Generar gráficos de torta y guardarlos
    plot_paths = []

    # Calcular el consumo total por cliente para cada variable
    total_kVarhD = fileHistoric[fileHistoric['VARIABLE'] == 'kVarhD'].groupby('CLIENTE')['USAGE_DATA'].sum()
    total_kWhD = fileHistoric[fileHistoric['VARIABLE'] == 'kWhD'].groupby('CLIENTE')['USAGE_DATA'].sum()

    # Crear la gráfica de torta para kVarhD
    plt.figure(figsize=(8, 6))
    plt.pie(total_kVarhD, explode=[0.1] * len(total_kVarhD), labels=total_kVarhD.index, autopct='%1.1f%%', shadow=True, startangle=140)
    plt.title('Consumo de kVarhD por cliente')
    plt.axis('equal')
    plot_path_kVarhD = os.path.join('static', 'plots', 'pie_kVarhD.png')
    plt.savefig(plot_path_kVarhD)
    plt.close()
    plot_paths.append('plots/pie_kVarhD.png')

    # Crear la gráfica de torta para kWhD
    plt.figure(figsize=(8, 6))
    plt.pie(total_kWhD, labels=total_kWhD.index, autopct='%1.1f%%', shadow=True, startangle=140)
    plt.title('Consumo de kWhD por cliente')
    plt.axis('equal')
    plot_path_kWhD = os.path.join('static', 'plots', 'pie_kWhD.png')
    plt.savefig(plot_path_kWhD)
    plt.close()
    plot_paths.append('plots/pie_kWhD.png')

    # Generar gráficos de predicción y guardarlos

    # Filtrar datos para kVarhD y kWhD
    kVarhD_data = fileHistoric[fileHistoric['VARIABLE'] == 'kVarhD']
    kWhD_data = fileHistoric[fileHistoric['VARIABLE'] == 'kWhD']

    # Agrupar por cliente y sumar los valores de predicción
    kVarhD_summary = kVarhD_data.groupby('CLIENTE')['PREDICTION'].sum()
    kWhD_summary = kWhD_data.groupby('CLIENTE')['PREDICTION'].sum()

    # Crear la gráfica de torta para predicción de kVarhD
    plt.figure(figsize=(10, 7))
    plt.pie(kVarhD_summary, labels=kVarhD_summary.index, autopct='%1.1f%%', startangle=140)
    plt.title('Distribución de Predicción de Energía kVarhD por Cliente')
    plt.axis('equal')
    plot_path_pred_kVarhD = os.path.join('static', 'plots', 'pie_pred_kVarhD.png')
    plt.savefig(plot_path_pred_kVarhD)
    plt.close()
    plot_paths.append('plots/pie_pred_kVarhD.png')

    # Crear la gráfica de torta para predicción de kWhD
    plt.figure(figsize=(10, 7))
    plt.pie(kWhD_summary, labels=kWhD_summary.index, autopct='%1.1f%%', startangle=140)
    plt.title('Distribución de Predicción de Energía kWhD por Cliente')
    plt.axis('equal')
    plot_path_pred_kWhD = os.path.join('static', 'plots', 'pie_pred_kWhD.png')
    plt.savefig(plot_path_pred_kWhD)
    plt.close()
    plot_paths.append('plots/pie_pred_kWhD.png')

    # Generar gráficos de barras comparando energías activa y reactiva para cada cliente
    bar_plot_paths = {}
    for cliente in fileHistoric['CLIENTE'].unique():
        try:
            bar_plot_paths[cliente] = visualizar_datos(fileHistoric, cliente)
        except (TypeError, ValueError) as e:
            print(f"Error generando gráfico para cliente {cliente}: {str(e)}")
            # Puedes manejar el error aquí, por ejemplo, saltarlo o guardar un gráfico predeterminado
            bar_plot_paths[cliente] = 'static/plots/default_plot.png'  # Ruta a una imagen predeterminada

    return render_template('index.html', plot_paths=plot_paths, bar_plot_paths=bar_plot_paths,
                           total_kWhD=total_kWhD, total_kVarhD=total_kVarhD, plots=plots,
                           clientes_info=clientes_info)

if __name__ == '__main__':
    app.run(debug=True)
