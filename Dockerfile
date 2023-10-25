FROM python:3.8
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6 v4l-utils libx11-xcb-dev libglu1-mesa-dev libxrender-dev libxi-dev -y 
RUN apt-get install libxcb-xinerama0
RUN pip install --upgrade pip

# Define o diretório de trabalho
WORKDIR /opt
COPY requirements.txt ./requirements.txt
COPY requirements.txt ./requirements_additional.txt
RUN pip install -r requirements.txt
RUN pip install -r requirements_additional.txt

# Copia o diretório atual para dentro do container
WORKDIR /opt/app
COPY ./api/main.py ./api/main.py
COPY ./api/routes.py ./api/routes.py
COPY ./api/service.py ./api/service.py
COPY ./deepface ./deepface
COPY ./setup.py ./

WORKDIR /opt/app/api
EXPOSE 7000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7000","--reload"]