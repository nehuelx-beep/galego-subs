import os
import sys
import uuid
import subprocess
import threading
import traceback
import time
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

# Estado do modelo — cargado en background despois de que Flask arrinque
model = None
model_ready = False
model_error = None


def load_model_background():
    """Load the Whisper model in a background thread after Flask has started."""
    global model, model_ready, model_error
    print(
        f"[{time.strftime('%H:%M:%S')}] Background thread: comezando carga do modelo Whisper...",
        flush=True,
    )
    try:
        t0 = time.time()
        model = whisper.load_model("medium")
        elapsed = time.time() - t0
        print(
            f"[{time.strftime('%H:%M:%S')}] Modelo listo en {elapsed:.1f}s. "
            f"Tipo: {type(model).__name__}, Device: {next(model.parameters()).device}",
            flush=True,
        )
        model_ready = True
    except Exception as e:
        model_error = str(e)
        print(f"[{time.strftime('%H:%M:%S')}] ERRO ao cargar o modelo Whisper: {e}", flush=True)
        traceback.print_exc(file=sys.stdout)
        sys.stdout.flush()


def update_job(job_id, **kwargs):
    jobs[job_id].update(kwargs)


def hex_to_ass_color(hex_color: str) -> str:
    """Convierte #RRGGBB a formato ASS &HBBGGRR (sin alfa)."""
    hex_color = hex_color.lstrip('#')
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f"&H{b:02X}{g:02X}{r:02X}"


def process_video(job_id, video_path: Path, font_name: str, font_size: int, font_color: str):
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

        # Guardar texto y timestamps de palabras para animación frontend
        full_text = transcription["text"].strip()
        word_timestamps = extract_word_timestamps(transcription["segments"])
        update_job(job_id, transcript=full_text, srt=srt_content, words=word_timestamps)

        update_job(job_id, status="burning", progress=80, message="Engadindo subtítulos ao vídeo...")

        # 4. Quemar subtítulos con FFmpeg — estilo limpio y configurable
        output_path = OUTPUT_DIR / f"{job_id}_subtitulado.mp4"

        ass_primary = hex_to_ass_color(font_color)
        # Outline negro suave, fondo casi transparente
        srt_fixed = str(srt_path).replace("\\", "/").replace(":", "\\:")

        subtitles_filter = (
            f"subtitles='{srt_fixed}':"
            f"force_style='FontName={font_name},"
            f"FontSize={font_size},"
            f"PrimaryColour={ass_primary},"
            f"OutlineColour=&H00000000,"   # outline negro puro
            f"BackColour=&H60000000,"      # fondo semitransparente suave
            f"BorderStyle=4,"              # caja de fondo
            f"Outline=2,"
            f"Shadow=0,"
            f"Bold=1,"
            f"Alignment=2,"               # centrado abajo
            f"MarginV=35'"
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


def extract_word_timestamps(segments):
    """Extrae timestamps de cada palabra para la animación del frontend."""
    words = []
    for seg in segments:
        if "words" in seg:
            for w in seg["words"]:
                words.append({
                    "word": w.get("word", "").strip(),
                    "start": round(w.get("start", 0), 3),
                    "end": round(w.get("end", 0), 3),
                })
    return words


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


@app.route("/health")
def health():
    """Immediate health check — always returns 200 so Railway knows the container is alive."""
    status = "ready" if model_ready else ("error" if model_error else "loading")
    return jsonify({"status": status, "model_ready": model_ready}), 200


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    if not model_ready:
        msg = f"Modelo aínda non está listo: {model_error}" if model_error else "Modelo cargando, agarda un momento..."
        return jsonify({"error": msg}), 503

    if "video" not in request.files:
        return jsonify({"error": "Non se recibiu ningún ficheiro"}), 400

    file = request.files["video"]
    if not file.filename:
        return jsonify({"error": "Nome de ficheiro baleiro"}), 400

    # Parámetros de estilo opcionales
    font_name = request.form.get("font", "Arial")
    font_size = int(request.form.get("size", 18))
    font_color = request.form.get("color", "#ffffff")

    # Validar tamaño razonable
    font_size = max(10, min(font_size, 40))

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
        "words": [],
        "output_file": ""
    }

    thread = threading.Thread(
        target=process_video,
        args=(job_id, video_path, font_name, font_size, font_color)
    )
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
    port = int(os.environ.get("PORT", 5000))
    print(f"[{time.strftime('%H:%M:%S')}] Flask arrincando na porta {port}...", flush=True)

    # Start model loading in background AFTER Flask is about to serve requests
    model_thread = threading.Thread(target=load_model_background, daemon=True)
    model_thread.start()

    print(f"[{time.strftime('%H:%M:%S')}] Flask listo. Modelo cargando en background.", flush=True)
    app.run(host="0.0.0.0", port=port)
