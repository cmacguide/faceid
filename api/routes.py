from fastapi import APIRouter, Request, Response, status, UploadFile, File  
import aiofiles
import service
import os

router = APIRouter()

os_path = "/home/cmac/desenvolvimento/trc/trcPlatform/faceRecognition/faceIDPi/employee"
db_path = "/home/cmac/desenvolvimento/trc/trcPlatform/faceRecognition/faceIDPi/representations"

# escrever uma rota para o serviço de upload
from fastapi import Request

@router.get("/")
async def root():
    return "<h1>Welcome to DeepFace API!</h1>"

@router.post("/find")
async def find(request: Request):   
    body = await request.json()
    
    img_path = body.get("img_path")
    if img_path is None:
        return {"message": "you must pass img_path input"}

    db_path = body.get("db_path")
    if db_path is None:
        return {"message": "you must pass db_path input"}

    model_name = body.get("model_name", "VGG-Face")
    detector_backend = body.get("detector_backend", "retinaface")
    enforce_detection = body.get("enforce_detection", False)
    align = body.get("align", True)

    prediction = await service.find(
        img_path=img_path,
        db_path=db_path,
        model_name=model_name,
        detector_backend=detector_backend,
        enforce_detection=enforce_detection,
        align=align,
    )

    return prediction

@router.get("/stream")
async def stream():
  return await service.stream()

@router.post("/represent")
async def represent(request: Request):
    body = request.json()

    if body is None:
        return {"message": "empty input set passed"}

    img_path = body.get("img")
    if img_path is None:
        return {"message": "you must pass img_path input"}

    model_name = body.get("model_name", "VGG-Face")
    detector_backend = body.get("detector_backend", "opencv")
    enforce_detection = body.get("enforce_detection", True)
    align = body.get("align", True)

    obj = await service.represent(
        img_path=img_path,
        model_name=model_name,
        detector_backend=detector_backend,
        enforce_detection=enforce_detection,
        align=align,
    )

    return obj

@router.post("/verify")
async def verify(request: Request):
    body = request.json()    

    if body is None:
        return {"message": "empty input set passed"}

    img1_path = body.get("img1_path")
    img2_path = body.get("img2_path")

    if img1_path is None:
        return {"message": "you must pass img1_path input"}

    if img2_path is None:
        return {"message": "you must pass img2_path input"}

    model_name = body.get("model_name", "VGG-Face")
    detector_backend = body.get("detector_backend", "opencv")
    enforce_detection = body.get("enforce_detection", True)
    distance_metric = body.get("distance_metric", "cosine")
    align = body.get("align", True)

    verification = await service.verify(
        img1_path=img1_path,
        img2_path=img2_path,
        model_name=model_name,
        detector_backend=detector_backend,
        distance_metric=distance_metric,
        align=align,
        enforce_detection=enforce_detection,
    )

    verification["verified"] = str(verification["verified"])

    return verification

@router.post("/analyze")
async def analyze(request: Request):
    body = request.json()

    if body is None:
        return {"message": "empty input set passed"}

    img_path = body.get("img_path")
    if img_path is None:
        return {"message": "you must pass img_path input"}

    detector_backend = body.get("detector_backend", "opencv")
    enforce_detection = body.get("enforce_detection", True)
    align = body.get("align", True)
    actions = body.get("actions", ["age", "gender", "emotion", "race"])

    demographies = await service.analyze(
        img_path=img_path,
        actions=actions,
        detector_backend=detector_backend,
        enforce_detection=enforce_detection,
        align=align,
    )

    return demographies

@router.post('/upload')
async def create_upload_file(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        img_path = os.path.join(os_path, file.filename)
        async with aiofiles.open(img_path, 'wb') as f:
            await f.write(contents)
    except Exception: 
        pass
    finally: 
        await file.close()      
        return {"message": f"Successfully uploaded {file.filename}"}
   