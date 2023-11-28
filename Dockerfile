# Usa la imagen oficial de Python con la versión que necesitas
FROM python:3.10.0

# Configura el entorno
ENV PYTHONUNBUFFERED 1
ENV PYTHONDONTWRITEBYTECODE 1
ENV LANG C.UTF-8

# Instala las dependencias de Django y otras bibliotecas de Python
RUN pip install --upgrade pip
COPY requirements.txt /app/
RUN pip install -r /app/requirements.txt

# Instala OpenJDK 8
RUN apt-get update && apt-get install -y openjdk-11-jdk coinor-cbc
# RUN chmod +x /dssProject/Pyomo/CBC/bin/cbc
RUN apt-get install -y coinor-cbc coinor-libcbc-dev

# Define el directorio de trabajo
WORKDIR /..

# Copia el código de la aplicación a la imagen
COPY . .

# Comando para ejecutar la aplicación (ajústalo según la configuración de tu aplicación)

# Ejecutable local
#CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]

# Anterior ejecutable de render
#ENTRYPOINT [ "gunicorn", "backAPI.wsgi"]  

# Actual ejecutable para render *limita
ENTRYPOINT ["gunicorn", "backAPI.wsgi", "--workers", "3", "--timeout", "1000", "--bind", "0.0.0.0:8000"] 
