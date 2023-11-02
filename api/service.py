from deepface import DeepFace
from dotenv import load_dotenv
import os

load_dotenv()

db_path = os.getenv("DB_PATH")
img_path = os.getenv("IMG_PATH")
models = [
    "ArcFace",
    "VGG-Face",
    "Facenet",
    "Facenet512",
    "DeepFace",
    "OpenFace",
]

backends = [
    'yolov8',
    'retinaface',
    'mtcnn',
    'mediapipe',
    'opencv',
    'ssd',
]

normalization = [
    'base',
    'VGGFace',
    'VGGFace2',
]

distance_metric = [
    'cosine',
    'euclidean',
    'euclidean_l2',
]


async def stream():
    print("Stream function called")
    obj = DeepFace.stream(
        db_path=db_path, model_name=models[2], detector_backend=backends[2], distance_metric=[0], enforce_detection=True, align=True)
    print("Stream object:", obj)
    return obj


def represent(img_path, model_name, detector_backend, enforce_detection, align):
    result = {}
    embedding_objs = DeepFace.represent(
        img_path=img_path,
        model_name=model_name,
        detector_backend=detector_backend,
        enforce_detection=enforce_detection,
        align=align,
    )
    result["results"] = embedding_objs
    return result


async def verify(
    img1_path, img2_path, model_name, detector_backend, distance_metric, enforce_detection, align, normalization
):
    obj = DeepFace.verify(
        img1_path=img1_path,
        img2_path=img2_path,
        model_name=model_name,
        detector_backend=detector_backend,
        distance_metric=distance_metric,
        normalization=normalization,
        align=align,
        enforce_detection=enforce_detection,
    )
    return obj


async def analyze(img_path, actions, detector_backend, enforce_detection, align, normalization, distance_metric):
    result = {}
    demographies = DeepFace.analyze(
        img_path=img_path,
        actions=actions,
        detector_backend=detector_backend,
        enforce_detection=enforce_detection,
        distance_metric=distance_metric,
        normalization=normalization,
        align=align,
    )
    result["results"] = demographies
    return result


async def find(img_path, db_path, model_name, detector_backend, enforce_detection, align, normalization, distance_metric):
    print("find function called")
    resultId = {}
    resultMt = {}
    prediction = DeepFace.find(
        img_path=img_path,
        db_path=db_path,
        model_name=model_name,
        detector_backend=detector_backend,
        enforce_detection=enforce_detection,
        normalization=normalization,
        distance_metric=distance_metric,
        align=align,
    )
    resultId["identity"] = prediction[0]["identity"][:3]
    resultMt["VGG-Face"] = prediction[0]["VGG-Face_cosine"][:3]

    # gerar o retorno em formato JSON
    return resultId, resultMt


async def find_face():
    print("find_face function called")
    resultId = {}
    resultMt = {}

    prediction = DeepFace.find(
        img_path=img_path,
        db_path=db_path,
        model_name=models[1],
        detector_backend=backends[1],
        distance_metric=distance_metric[0],
        normalization=normalization[2],
        enforce_detection=False,
        align=True,
    )
    resultId["identity"] = prediction[0]["identity"][:1]
    resultMt["VGG-Face"] = prediction[0]["VGG-Face_cosine"][:1]

    # apagar o arquivo da pasta employees
    if img_path and os.path.exists(img_path):
        os.remove(img_path)

    # gerar o retorno em formato JSON
    return resultId, resultMt
