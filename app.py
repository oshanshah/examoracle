from flask import Flask, request, render_template
import os
from agent import run_exam_oracle

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    subject = request.form.get("subject")

    files = request.files.getlist("files")

    for file in files:
        if file.filename:
            file.save(os.path.join(UPLOAD_FOLDER, file.filename))

    result = run_exam_oracle(subject, UPLOAD_FOLDER)

    return render_template("index.html", result=result)


if __name__ == "__main__":
    app.run(debug=True, port=8000)