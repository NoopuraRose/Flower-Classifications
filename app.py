from pathlib import Path
from uuid import uuid4

from flask import Flask, render_template, request, url_for
from PIL import Image, UnidentifiedImageError
from ultralytics import YOLO
from werkzeug.utils import secure_filename


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "runs" / "classify" / "train" / "weights" / "best.pt"
UPLOAD_DIR = BASE_DIR / "static" / "uploads"
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}
MAX_UPLOAD_SIZE = 15 * 1024 * 1024

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_SIZE
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Trained model was not found at {MODEL_PATH}")

model = YOLO(str(MODEL_PATH))


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def remove_old_uploads() -> None:
    """Keep the upload folder from growing forever during local use."""
    files = sorted(
        (path for path in UPLOAD_DIR.iterdir() if path.is_file()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for path in files[20:]:
        path.unlink(missing_ok=True)


def classify_image(image_path: Path) -> tuple[str, float, list[dict[str, float | str]]]:
    result = model(str(image_path), verbose=False)[0]
    probabilities = result.probs
    top_indices = probabilities.top5
    top_confidences = probabilities.top5conf.tolist()
    predictions = [
        {
            "label": result.names[index],
            "confidence": float(confidence),
        }
        for index, confidence in zip(top_indices, top_confidences)
    ]
    return (
        predictions[0]["label"],
        predictions[0]["confidence"],
        predictions,
    )


@app.errorhandler(413)
def request_too_large(_error):
    return render_template("index.html", error="That image is larger than 15 MB."), 413


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        return render_template("index.html")

    uploaded_file = request.files.get("image")
    if uploaded_file is None or not uploaded_file.filename:
        return render_template("index.html", error="Please choose a flower image first."), 400
    if not allowed_file(uploaded_file.filename):
        return render_template(
            "index.html",
            error="Unsupported format. Please use JPG, JPEG, PNG, or WEBP.",
        ), 400

    filename = f"{uuid4().hex}_{secure_filename(uploaded_file.filename.rsplit('.', 1)[0])}.jpg"
    image_path = UPLOAD_DIR / filename
    uploaded_file.save(image_path)

    try:
        with Image.open(image_path) as image:
            image.verify()
        with Image.open(image_path) as image:
            image.convert("RGB").save(image_path, format="JPEG", quality=95)
        label, confidence, predictions = classify_image(image_path)
    except (UnidentifiedImageError, OSError):
        image_path.unlink(missing_ok=True)
        return render_template("index.html", error="The uploaded file is not a valid image."), 400
    except Exception:
        image_path.unlink(missing_ok=True)
        app.logger.exception("Flower prediction failed")
        return render_template(
            "index.html",
            error="The model could not process this image. Please try another one.",
        ), 500

    remove_old_uploads()
    return render_template(
        "index.html",
        image_url=url_for("static", filename=f"uploads/{filename}"),
        label=label,
        confidence=confidence,
        predictions=predictions,
    )


if __name__ == "__main__":
    app.run(debug=True)
