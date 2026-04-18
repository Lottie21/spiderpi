import io
import json
import tempfile
import time
import wave
import urllib.request

import pyaudio

# ====== 可配置参数 ======
WHISPER_HOST = "192.168.1.113"
WHISPER_PORT = 5001
RECORD_SECONDS = 4
SAMPLE_RATE    = 16000
CHUNK          = 1024
CHANNELS       = 1


def record_wav(seconds: int = RECORD_SECONDS) -> bytes:
    """用 pyaudio 录音，返回 wav 格式的 bytes"""
    pa = pyaudio.PyAudio()
    stream = pa.open(
        format=pyaudio.paInt16,
        channels=CHANNELS,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=CHUNK,
    )

    frames = []
    for _ in range(int(SAMPLE_RATE / CHUNK * seconds)):
        data = stream.read(CHUNK, exception_on_overflow=False)
        frames.append(data)

    stream.stop_stream()
    stream.close()
    pa.terminate()

    # 写入内存 wav
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(pa.get_sample_size(pyaudio.paInt16))
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(b"".join(frames))
    return buf.getvalue()


def transcribe(wav_bytes: bytes, host: str = WHISPER_HOST, port: int = WHISPER_PORT) -> str:
    """
    把 wav bytes 以 multipart/form-data 发给 Whisper 服务
    返回识别出的文字，失败返回空字符串
    """
    url = f"http://{host}:{port}/transcribe"

    boundary = "----WhisperBoundary7654321"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="audio"; filename="audio.wav"\r\n'
        f"Content-Type: audio/wav\r\n\r\n"
    ).encode("utf-8") + wav_bytes + f"\r\n--{boundary}--\r\n".encode("utf-8")

    req = urllib.request.Request(
        url=url,
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data.get("text", "").strip()
    except Exception as e:
        print(f"[whisper_client] 请求失败: {e}")
        return ""


def listen_and_transcribe(
    seconds: int = RECORD_SECONDS,
    host: str = WHISPER_HOST,
    port: int = WHISPER_PORT,
) -> str:
    
    print(f"[whisper_client] 开始录音 {seconds} 秒...")
    wav_bytes = record_wav(seconds)
    print("[whisper_client] 录音完成，发送至 Whisper...")
    text = transcribe(wav_bytes, host=host, port=port)
    print(f"[whisper_client] 识别结果: {text}")
    return text


if __name__ == "__main__":
    result = listen_and_transcribe()
    print("最终结果:", result)
