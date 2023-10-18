#base image
# FROM continuumio/anaconda3
FROM python:3.8
# Adiciona o conda-forge aos seus canais
# RUN conda config --add channels conda-forge
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6 v4l-utils libx11-xcb-dev libglu1-mesa-dev libxrender-dev libxi-dev -y 

RUN apt-get install libxcb-xinerama0

# Define a prioridade do canal como estrita
# RUN conda config --set channel_priority strict
RUN pip install --upgrade pip
# Instala a biblioteca DeepFace
# RUN conda install -c conda-forge deepface
# RUN conda install -c conda-forge gunicorn
# RUN pip install gunicorn

# Define o diretório de trabalho
WORKDIR /app
COPY requirements.txt /app/requirements.txt
COPY requirements.txt /app/requirements_additional.txt
RUN pip install -r requirements.txt
RUN pip install -r requirements_additional.txt
# Copia o diretório atual para dentro do container
COPY . /app

EXPOSE 5000
# CMD ["gunicorn", "--workers=1", "--timeout=3600", "--bind=0.0.0.0:5000", "app:create_app()"]
CMD ["python", "stream.py"]
