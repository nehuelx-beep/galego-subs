import os
import uuid
import json
import subprocess
import threading
from pathlib import Path
from flask import Flask, request, jsonify, render_template, send_file
import whisper

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500 MB

BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"

UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Estado de los trabajos en memoria
jobs = {}

# Carga el modelo Whisper (cambia a "medium" o "large" para más precisión)
print("Cargando modelo Whisper... (puede tardar un momento la primera vez)")
model = whisper.load_model("medium")
print("Modelo listo.")


def update_job(job_id, **kwargs):
    jobs[job_id].update(kwargs)


def process_video(job_id, video_path: Path):
    try:
        update_job(job_id, status="extracting", progress=10, message="Extraendo audio do vídeo...")

        # 1. Extraer audio con FFmpeg
        audio_path = UPLOAD_DIR / f"{job_id}_audio.wav"
        result = subprocess.run([
            "ffmpeg", "-i", str(video_path),
            "-vn", "-acodec", "pcm_s16le",
            "-ar", "16000", "-ac", "1",
            str(audio_path), "-y"
        ], capture_output=True, text=True)

        if result.returncode != 0:
            raise Exception(f"FFmpeg error: {result.stderr}")

        update_job(job_id, status="transcribing", progress=30, message="Transcribindo en galego con Whisper...")

        # 2. Transcribir con Whisper en gallego
        transcription = model.transcribe(
            str(audio_path),
            language="gl",
            task="transcribe",
            word_timestamps=True,
            verbose=False
        )

        update_job(job_id, status="generating_srt", progress=65, message="Xerando ficheiro de subtítulos...")

        # 3. Generar archivo SRT
        srt_path = UPLOAD_DIR / f"{job_id}.srt"
        srt_content = generate_srt(transcription["segments"])

        with open(srt_path, "w", encoding="utf-8") as f:
            f.write(srt_content)

        # Guardar también el texto para mostrarlo en la UI
        full_text = transcription["text"].strip()
        update_job(job_id, transcript=full_text, srt=srt_content)

        update_job(job_id, status="burning", progress=80, message="Engadindo subtítulos ao vídeo...")

             # 4. Quemar subtítulos con FFmpeg
        output_path = OUTPUT_DIR / f"{job_id}_subtitulado.mp4"

        # Estilo pensado para redes sociales: texto grande, fondo semitransparente
        srt_fixed = str(srt_path).replace("\\", "/").replace(":", "\\:")

        subtitles_filter = (
            f"subtitles='{srt_fixed}':"
            "force_style='FontName=Arial,FontSize=22,PrimaryColour=&HFFFFFF,"
            "OutlineColour=&H40000000,BorderStyle=4,BackColour=&H80000000,"
            "Outline=0,Shadow=0,Alignment=2,MarginV=40'"
        )

        result = subprocess.run([
            "ffmpeg", "-i", str(video_path),
            "-vf", subtitles_filter,
            "-c:v", "libx264", "-crf", "23",
            "-c:a", "aac", "-b:a", "192k",
            str(output_path), "-y"
        ], capture_output=True, text=True)
        if result.returncode != 0:
            raise Exception(f"FFmpeg burn error: {result.stderr}")

        # 5. Limpiar archivos temporales
        audio_path.unlink(missing_ok=True)

        update_job(job_id,
            status="done",
            progress=100,
            message="Vídeo listo!",
            output_file=output_path.name
        )

    except Exception as e:
        update_job(job_id, status="error", progress=0, message=f"Erro: {str(e)}")


def generate_srt(segments):
    """Convierte los segmentos de Whisper a formato SRT."""
    lines = []
    for i, seg in enumerate(segments, 1):
        start = format_timestamp(seg["start"])
        end = format_timestamp(seg["end"])
        text = seg["text"].strip()
        lines.append(f"{i}\n{start} --> {end}\n{text}\n")
    return "\n".join(lines)


def format_timestamp(seconds: float) -> str:
    """Formatea segundos a HH:MM:SS,mmm para SRT."""
    ms = int((seconds % 1) * 1000)
    s = int(seconds) % 60
    m = int(seconds // 60) % 60
    h = int(seconds // 3600)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    if "video" not in request.files:
        return jsonify({"error": "Non se recibiu ningún ficheiro"}), 400

    file = request.files["video"]
    if not file.filename:
        return jsonify({"error": "Nome de ficheiro baleiro"}), 400

    job_id = str(uuid.uuid4())[:8]
    ext = Path(file.filename).suffix.lower()
    video_path = UPLOAD_DIR / f"{job_id}{ext}"
    file.save(str(video_path))

    jobs[job_id] = {
        "status": "queued",
        "progress": 0,
        "message": "En cola...",
        "filename": file.filename,
        "transcript": "",
        "srt": "",
        "output_file": ""
    }

    thread = threading.Thread(target=process_video, args=(job_id, video_path))
    thread.daemon = True
    thread.start()

    return jsonify({"job_id": job_id})


@app.route("/status/<job_id>")
def status(job_id):
    if job_id not in jobs:
        return jsonify({"error": "Traballo non atopado"}), 404
    return jsonify(jobs[job_id])


@app.route("/download/<job_id>")
def download(job_id):
    if job_id not in jobs or not jobs[job_id].get("output_file"):
        return jsonify({"error": "Ficheiro non dispoñible"}), 404
    output_path = OUTPUT_DIR / jobs[job_id]["output_file"]
    return send_file(str(output_path), as_attachment=True,
                     download_name=f"subtitulado_{jobs[job_id]['filename']}")


@app.route("/download_srt/<job_id>")
def download_srt(job_id):
    if job_id not in jobs or not jobs[job_id].get("srt"):
        return jsonify({"error": "SRT non dispoñible"}), 404
    srt_path = UPLOAD_DIR / f"{job_id}.srt"
    return send_file(str(srt_path), as_attachment=True,
                     download_name=f"subtitulos_{jobs[job_id]['filename']}.srt")


if __name__ == "__main__":
    import os
    print("\n✓ Abre o navegador en: http://localhost:5000\n")

    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000))
    )
