FROM python:3.11-slim
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6 v4l-utils -y 
RUN apt-get install libxcb-xinerama0
RUN python3 -m pip install --no-cache-dir --upgrade pip

# Define o diretório de trabalho
WORKDIR /opt
COPY requirements.txt ./requirements.txt
COPY requirements.txt ./requirements_additional.txt
# RUN pip install --ignore-installed -r requirements.txt
RUN python3 -m pip install --ignore-installed --no-cache-dir -r requirements.txt -U
RUN python3 -m pip install --ignore-installed --no-cache-dir -r requirements_additional.txt  -U 

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