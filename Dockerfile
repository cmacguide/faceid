#base image
# FROM continuumio/anaconda3
FROM python:3.8
# Adiciona o conda-forge aos seus canais
# RUN conda config --add channels conda-forge
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6 v4l-utils libx11-xcb-dev libglu1-mesa-dev libxrender-dev libxi-dev -y 
RUN apt-get install libxcb-xinerama0
RUN pip install --upgrade pip

# Define o diretório de trabalho
WORKDIR /app
COPY requirements.txt /app/requirements.txt
COPY requirements.txt /app/requirements_additional.txt
RUN pip install -r requirements.txt
RUN pip install -r requirements_additional.txt
# Copia o diretório atual para dentro do container
COPY ./deepface /app/deepface
COPY ./api/app.py /app/
COPY ./api/routes.py /app/
COPY ./api/service.py /app/
COPY ./setup.py /app/
# COPY . /app

EXPOSE 5000
CMD ["gunicorn", "--workers=1", "--timeout=3600", "--bind=0.0.0.0:5000", "app:create_app()"]
# CMD ["python", "stream.py"]
