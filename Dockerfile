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
RUN apt-get update && apt-get install -y openjdk-11-jdk

# Define el directorio de trabajo
WORKDIR /app

# Copia el código de la aplicación a la imagen
COPY . /app/

# Comando para ejecutar la aplicación (ajústalo según la configuración de tu aplicación)
#CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
ENTRYPOINT [ "gunicorn", "core.wsgi"]
