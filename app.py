import os
import uuid
import subprocess
from flask import Flask, render_template, request, send_file
from moviepy.editor import *
import cv2
import whisper

app = Flask(__name__)

UPLOAD = "uploads"
OUTPUT = "outputs"

os.makedirs(UPLOAD, exist_ok=True)
os.makedirs(OUTPUT, exist_ok=True)

model = whisper.load_model("base")


# ---------- Face Crop ----------
def crop_vertical(input_path, output_path):
    cap = cv2.VideoCapture(input_path)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    target_w = int(height * 9 / 16)
    x1 = (width - target_w) // 2

    cmd = f'ffmpeg -y -i "{input_path}" -vf "crop={target_w}:{height}:{x1}:0" "{output_path}"'
    os.system(cmd)


# ---------- Subtitle ----------
def add_subtitle(video_path):
    result = model.transcribe(video_path)
    srt = video_path.replace(".mp4", ".srt")

    with open(srt, "w", encoding="utf-8") as f:
        for i, seg in enumerate(result["segments"]):
            f.write(f"{i+1}\n")
            f.write(f"00:00:{int(seg['start']):02d},000 --> 00:00:{int(seg['end']):02d},000\n")
            f.write(seg["text"] + "\n\n")

    return srt


# ---------- Generate Shorts ----------
def generate_shorts(video_path, duration, count):
    clip = VideoFileClip(video_path)
    total = int(clip.duration)

    shorts = []

    for i in range(count):
        start = (total // count) * i
        end = start + duration

        sub = clip.subclip(start, end)

        name = f"{uuid.uuid4()}.mp4"
        temp = os.path.join(OUTPUT, "temp_" + name)
        final = os.path.join(OUTPUT, name)

        sub.write_videofile(temp, codec="libx264")

        crop_vertical(temp, final)

        shorts.append(final)

    return shorts


# ---------- ROUTES ----------
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/generate", methods=["POST"])
def generate():
    file = request.files["video"]
    duration = int(request.form["duration"])
    count = int(request.form["count"])

    path = os.path.join(UPLOAD, file.filename)
    file.save(path)

    videos = generate_shorts(path, duration, count)

    return render_template("index.html", videos=videos)


@app.route("/download/<path:path>")
def download(path):
    return send_file(path, as_attachment=True)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
