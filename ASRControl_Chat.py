import time
import import_path

import HiwonderSDK.TTS as TTS
import HiwonderSDK.ASR as ASR
import kinematics as kinematics
import HiwonderSDK.Board as Board

from ollama_client import ollama_generate
from whisper_client import listen_and_transcribe

# ====== 配置参数 ======
OLLAMA_HOST    = "192.168.1.113"
WHISPER_HOST   = "192.168.1.113"
MODEL_NAME     = "qwen2.5:1.5b"
TIMEOUT_S      = 5
RECORD_SECONDS = 4

WAKE_ID = 100  # hi

TTS_SIGN_ACK   = ""
TTS_SIGN_READY = "[h0][v10][m3]"

FALLBACK_TEXT  = "网络不稳"

ik = kinematics.IK()

MOTION_IDS = {2, 3, 4, 5}

try:
	asr = ASR.ASR()
	tts = TTS.TTS()

	debug = True
	if debug:
		asr.eraseWords()
		asr.setMode(2)
		asr.addWords(1, 'kai shi')
		asr.addWords(2, 'wang qian zou')
		asr.addWords(2, 'qian jin')
		asr.addWords(2, 'zhi zou')
		asr.addWords(3, 'wang hou tui')
		asr.addWords(4, 'xiang zuo yi dong')
		asr.addWords(5, 'xiang you yi dong')
		asr.addWords(WAKE_ID, 'hi')

	Board.setPWMServoPulse(1, 1500, 500)
	Board.setPWMServoPulse(2, 1500, 500)
	ik.stand(ik.initial_pos)
	tts.TTSModuleSpeak(TTS_SIGN_READY, '准备就绪')
	time.sleep(1)
except Exception:
	print('传感器初始化出错')
	raise


def speak(text: str, sign: str = TTS_SIGN_ACK):
	if not text:
		return
	tts.TTSModuleSpeak(sign, text)


def handle_motion(cmd_id: int):
	time.sleep(0.2)
	if cmd_id == 2:
		ik.go_forward(ik.initial_pos, 2, 100, 80, 2)
	elif cmd_id == 3:
		ik.back(ik.initial_pos, 2, 100, 80, 2)
	elif cmd_id == 4:
		ik.left_move(ik.initial_pos, 2, 100, 80, 2)
	elif cmd_id == 5:
		ik.right_move(ik.initial_pos, 2, 100, 80, 2)


def handle_chat():
	speak('我在')

	# 录音 → Whisper 转写
	user_text = listen_and_transcribe(
		seconds=RECORD_SECONDS,
		host=WHISPER_HOST,
	)

	if not user_text:
		speak('我没听清楚')
		return

	print(f'[chat] user: {user_text}')

	# 发给 Ollama
	prompt = f"你是SpiderPi电子宠物。用户说：{user_text}。用中文回复一句话，不超过20个字"
	reply = ollama_generate(
		prompt,
		host=OLLAMA_HOST,
		model=MODEL_NAME,
		timeout_s=TIMEOUT_S,
	)

	if not reply:
		reply = FALLBACK_TEXT
	print(f'[chat] ollama: {reply}')
	speak(reply)


while True:
	data = asr.getResult()
	if data:
		print('result:', data)
		if data == WAKE_ID:
			handle_chat()
		elif data in MOTION_IDS:
			handle_motion(data)
	time.sleep(0.01)
