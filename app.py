from flask import Flask
app = Flask(__name__, template_folder="templates")
from flask import Flask, render_template, request, send_file
import os
from PIL import Image, ImageChops, ImageEnhance
from PIL.ExifTags import TAGS
import numpy as np
import cv2
import uuid

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
REPORT_FOLDER = "static/reports"

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(REPORT_FOLDER, exist_ok=True)


def unique_filename(filename):
    ext = filename.split(".")[-1]
    return str(uuid.uuid4()) + "." + ext


# ---------- ELA ----------
def perform_ela(image_path):
    original = Image.open(image_path)

    if original.mode == "RGBA":
        original = original.convert("RGB")

    temp_path = "temp.jpg"
    original.save(temp_path, "JPEG", quality=90)

    compressed = Image.open(temp_path)
    diff = ImageChops.difference(original, compressed)

    ela_image = ImageEnhance.Brightness(diff).enhance(15)

    filename = "ela_" + os.path.basename(image_path)
    path = os.path.join(UPLOAD_FOLDER, filename)

    ela_image.save(path)

    score = np.array(ela_image).mean()
    return filename, score


# ---------- Heatmap ----------
def generate_heatmap(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (21,21), 0)

    heatmap = cv2.absdiff(gray, blur)
    heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
    heatmap = cv2.applyColorMap(heatmap.astype(np.uint8), cv2.COLORMAP_JET)

    filename = "heatmap_" + os.path.basename(image_path)
    path = os.path.join(UPLOAD_FOLDER, filename)

    cv2.imwrite(path, heatmap)
    return filename


# ---------- Metadata ----------
def extract_metadata(image_path):
    metadata = {}
    image = Image.open(image_path)
    exif_data = image._getexif()

    if exif_data:
        for tag_id, value in exif_data.items():
            tag = TAGS.get(tag_id, tag_id)
            metadata[tag] = value

    return metadata


# ---------- Classification ----------
def classify_image(score):
    if score < 15:
        return "Likely Authentic"
    elif score < 30:
        return "Possibly Manipulated"
    else:
        return "Highly Suspicious"


# ---------- PDF Report ----------
def generate_pdf(results, authentic):

    file_path = os.path.join(REPORT_FOLDER, "report.pdf")

    doc = SimpleDocTemplate(file_path)
    styles = getSampleStyleSheet()

    content = []

    content.append(Paragraph("Digital Image Forensics Report", styles["Title"]))
    content.append(Spacer(1, 20))

    for img in results:
        text = f"""
        Image: {img['name']}<br/>
        Score: {img['score']}<br/>
        Status: {img['class']}<br/>
        Camera: {img['camera']}<br/>
        Date: {img['date']}<br/>
        Software: {img['software']}<br/><br/>
        """
        content.append(Paragraph(text, styles["Normal"]))
        content.append(Spacer(1, 15))

    content.append(Paragraph(f"Most Authentic Image: {authentic}", styles["Heading2"]))

    doc.build(content)

    return file_path


@app.route("/", methods=["GET", "POST"])
def index():

    results = []

    if request.method == "POST":

        images = request.files.getlist("images")

        for image in images:

            filename = unique_filename(image.filename)
            path = os.path.join(UPLOAD_FOLDER, filename)
            image.save(path)

            ela, score = perform_ela(path)
            heatmap = generate_heatmap(path)

            metadata = extract_metadata(path)

            results.append({
                "name": filename,
                "ela": ela,
                "heatmap": heatmap,
                "score": round(score, 2),
                "class": classify_image(score),
                "camera": metadata.get("Model", "Unknown"),
                "date": metadata.get("DateTime", "Unknown"),
                "software": metadata.get("Software", "Not Available")
            })

        most_authentic = min(results, key=lambda x: x["score"])

        pdf_path = generate_pdf(results, most_authentic["name"])

        return render_template(
            "index.html",
            results=results,
            authentic=most_authentic["name"],
            report=pdf_path
        )

    return render_template("index.html", results=None)


@app.route("/download")
def download():
    return send_file("static/reports/report.pdf", as_attachment=True)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
    