from flask import Flask, request, jsonify
from faster_whisper import WhisperModel
import tempfile, os

app = Flask(__name__)

# 首次运行会自动下载模型（约150MB），之后缓存
model = WhisperModel("small", device="cpu", compute_type="int8")

@app.route("/health")
def health():
    return jsonify({"status": "ok"})

@app.route("/transcribe", methods=["POST"])
def transcribe():
    if "audio" not in request.files:
        return jsonify({"error": "no audio file"}), 400

    audio_file = request.files["audio"]

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        audio_file.save(tmp.name)
        tmp_path = tmp.name

    try:
        segments, _ = model.transcribe(tmp_path, language="zh")
        text = "".join(s.text for s in segments).strip()
    finally:
        os.unlink(tmp_path)

    return jsonify({"text": text})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
