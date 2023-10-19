from deepface import DeepFace

FACE_DB_PATH = './face-db01/00155196510.jpg'
FACE_DB_PATH2 = './face-db/11677541741.jpg'
# FACE_DB = './face-db'
FACE_DB = './representations'

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
FACE_DB_PATH = './face-db01/00155196510.jpg'
FACE_DB_PATH2 = './face-db/11677541741.jpg'
# FACE_DB = './face-db'
FACE_DB = './representations'

prediction = DeepFace.find(img_path=FACE_DB_PATH, db_path=FACE_DB,
                        model_name=models[2],
                        detector_backend=backends[2],
                        enforce_detection=False,
                        
                        )
result = prediction[0]["identity"][0]
print(prediction)
print("result",result)