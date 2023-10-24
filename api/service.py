from deepface import DeepFace
import json

FACE_DB_PATH = './employee/image.jpg'
FACE_DB = './representations'
db_path = "/home/cmac/desenvolvimento/trc/trcPlatform/faceRecognition/faceIDPi/representations"
img_path = "/home/cmac/desenvolvimento/trc/trcPlatform/faceRecognition/faceIDPi/employee/image.jpg"
models = [
  "Facenet512",   
  "DeepFace",  
  "VGG-Face", 
  "OpenFace", 
  "Facenet",   
]

backends = [
  'opencv', 
  'ssd',
  'retinaface',    
  'mtcnn', 
]
async def stream():
   print("Stream function called")
   obj = DeepFace.stream(db_path ="./representations")
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
    img1_path, img2_path, model_name, detector_backend, distance_metric, enforce_detection, align
):
    obj = DeepFace.verify(
        img1_path=img1_path,
        img2_path=img2_path,
        model_name=model_name,
        detector_backend=detector_backend,
        distance_metric=distance_metric,
        align=align,
        enforce_detection=enforce_detection,
    )
    return obj


async def analyze(img_path, actions, detector_backend, enforce_detection, align):
    result = {}
    demographies = DeepFace.analyze(
        img_path=img_path,
        actions=actions,
        detector_backend=detector_backend,
        enforce_detection=enforce_detection,
        align=align,
    )
    result["results"] = demographies
    return result

async def find(img_path, db_path, model_name, detector_backend, enforce_detection, align):
    resultId = {}
    resultMt = {}
    prediction = DeepFace.find(
        img_path=img_path,
        db_path=db_path,
        model_name=model_name,
        detector_backend=detector_backend,
        enforce_detection=enforce_detection,
        align=align,
    )   
    resultId["identity"] = prediction[0]["identity"][:3]
    resultMt["VGG-Face"] = prediction[0]["VGG-Face_cosine"][:3]
      
    # gerar o retorno em formato JSON
    return resultId, resultMt
 

async def find_face():
    resultId = {}
    resultMt = {}
    prediction = DeepFace.find(
        img_path=img_path,
        db_path=db_path,
        model_name=models[2],
        detector_backend=backends[2],
        enforce_detection=True,
        align=True,
    )
    resultId["identity"] = prediction[0]["identity"][:1]
    resultMt["VGG-Face"] = prediction[0]["VGG-Face_cosine"][:1]
    
    # gerar o retorno em formato JSON
    return resultId, resultMt