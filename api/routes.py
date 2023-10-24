from fastapi import APIRouter, Request, Response, status, UploadFile, File  
import aiofiles
import service
import os

router = APIRouter()

os_path = "/home/cmac/desenvolvimento/trc/trcPlatform/faceRecognition/faceIDPi/employee"
db_path = "/home/cmac/desenvolvimento/trc/trcPlatform/faceRecognition/faceIDPi/representations"

# escrever uma rota para o serviço de upload
from fastapi import Request

@router.post('/upload')
# escrever a função para o upload da imagem
# async def upload_file(file : UploadFile):
#     # escrever um código para salvar o arquivo no diretório
#     print(file.content_type)          
#     # escrever um if para verificar se o arquivo é uma imagem
#     if file.content_type.startswith('image/'):
#         file_contents = await file.read()
#     # verificar o nome do arquivo 
#         # print(file_contents.filename)
#         # img_path = os.path.join(os_path, file.filename)
#         # file.save(img_path)
#         return {"message": "File successfully uploaded", "img_path": img_path}
#         # return {"message": "File is an image"}
#     else:
#         return {"message": "File is not an image"}
async def create_upload_file(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        async with aiofiles.open(file.name, 'wb') as f:
            await f.write(contents)
    except Exception: 
        pass
    finally: 
        await file.close()      
        return {"message": f"Successfully uploaded {file.filename}"}
   
    # file_contents = await file.read()
    # print("Request",file.name)
    # if 'file' not in request.file:
    #     return {"message": "No file part in the request"}

    # file = request.file['file']  
    # if file.filename == '':
    #     return {"message": "No file selected for uploading"}

    # if file:
    #     print(file)
    #     img_path = os.path.join(os_path, file.filename)
    #     file.save(img_path)
    #     return {"message": "File successfully uploaded", "img_path": img_path}

# escrever uma rota para o serviço de find
@router.post("/find")
# escrever a função para o find fazendo o upload da imagem
async def find(request: Request):
    input_args = request.get_json()
    if input_args is None:
        return {"message": "empty input set passed"}

    img_path = input_args.get("img_path")
    if img_path is None:
        return {"message": "you must pass img_path input"}

    db_path = input_args.get("db_path")
    if db_path is None:
        return {"message": "you must pass db_path input"}

    model_name = input_args.get("model_name", "VGG-Face")
    detector_backend = input_args.get("detector_backend", "opencv")
    enforce_detection = input_args.get("enforce_detection", False)
    align = input_args.get("align", True)

    prediction = await service.find(
        img_path=img_path,
        db_path=db_path,
        model_name=model_name,
        detector_backend=detector_backend,
        enforce_detection=enforce_detection,
        align=align,
    )

    return prediction

@router.get("/")
async def root():
    return "<h1>Welcome to DeepFace API!</h1>"

@router.get("/stream")
async def stream():
  return await service.stream()

@router.post("/represent")
async def represent(request: Request):
    input_args = request.get_json()

    if input_args is None:
        return {"message": "empty input set passed"}

    img_path = input_args.get("img")
    if img_path is None:
        return {"message": "you must pass img_path input"}

    model_name = input_args.get("model_name", "VGG-Face")
    detector_backend = input_args.get("detector_backend", "opencv")
    enforce_detection = input_args.get("enforce_detection", True)
    align = input_args.get("align", True)

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
    input_args = request.get_json()
    

    if input_args is None:
        return {"message": "empty input set passed"}

    img1_path = input_args.get("img1_path")
    img2_path = input_args.get("img2_path")

    if img1_path is None:
        return {"message": "you must pass img1_path input"}

    if img2_path is None:
        return {"message": "you must pass img2_path input"}

    model_name = input_args.get("model_name", "VGG-Face")
    detector_backend = input_args.get("detector_backend", "opencv")
    enforce_detection = input_args.get("enforce_detection", True)
    distance_metric = input_args.get("distance_metric", "cosine")
    align = input_args.get("align", True)

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
    input_args = request.get_json()

    if input_args is None:
        return {"message": "empty input set passed"}

    img_path = input_args.get("img_path")
    if img_path is None:
        return {"message": "you must pass img_path input"}

    detector_backend = input_args.get("detector_backend", "opencv")
    enforce_detection = input_args.get("enforce_detection", True)
    align = input_args.get("align", True)
    actions = input_args.get("actions", ["age", "gender", "emotion", "race"])

    demographies = await service.analyze(
        img_path=img_path,
        actions=actions,
        detector_backend=detector_backend,
        enforce_detection=enforce_detection,
        align=align,
    )

    return demographies
