from flask import Blueprint, request
import service
import os

blueprint = Blueprint("routes", __name__)

os_path = "/home/cmac/desenvolvimento/trc/trcPlatform/faceRecognition/faceIDPi/employee"
db_path = "/home/cmac/desenvolvimento/trc/trcPlatform/faceRecognition/faceIDPi/representations"

# escrever uma rota para o serviço de upload
@blueprint.route('/upload', methods=['POST'])
# escrever a função para o upload da imagem
def upload_file():
    if 'file' not in request.files:
        return {"message": "No file part in the request"}

    file = request.files['file']  
    if file.filename == '':
        return {"message": "No file selected for uploading"}

    if file:
        print(file)
        img_path = os.path.join(os_path, file.filename)
        file.save(img_path)
        return {"message": "File successfully uploaded", "img_path": img_path}

# escrever uma rota para o serviço de find
@blueprint.route("/find", methods=["POST"])
# escrever a função para o find fazendo o upload da imagem
def find():
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

    prediction = service.find(
        img_path=img_path,
        db_path=db_path,
        model_name=model_name,
        detector_backend=detector_backend,
        enforce_detection=enforce_detection,
        align=align,
    )

    return prediction



@blueprint.route("/")
def home():
    return "<h1>Welcome to DeepFace API!</h1>"

@blueprint.route("/stream")
def stream():
    service.stream()

@blueprint.route("/represent", methods=["POST"])
def represent():
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

    obj = service.represent(
        img_path=img_path,
        model_name=model_name,
        detector_backend=detector_backend,
        enforce_detection=enforce_detection,
        align=align,
    )

    return obj


@blueprint.route("/verify", methods=["POST"])
def verify():
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

    verification = service.verify(
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


@blueprint.route("/analyze", methods=["POST"])
def analyze():
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

    demographies = service.analyze(
        img_path=img_path,
        actions=actions,
        detector_backend=detector_backend,
        enforce_detection=enforce_detection,
        align=align,
    )

    return demographies
