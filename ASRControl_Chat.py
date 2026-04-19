#!/usr/bin/python3
# coding=utf-8
"""ASRControl_Chat_SpiderSpider.py

安全架构：LLM 不具备执行权限，只具备建议权，执行权在规则与确认层
降级策略：本地关键词层 → 远端服务层 → TTS 承认失败

完整流程：
  1. ASR 常驻监听
  2. 说 Hi 唤醒 → 健康检查 Whisper/Ollama
  3. 全部离线时走本地关键词层，否则进入完整流程
  4. LLM 意图建议 → 白名单+句式校验 → 语音确认 → 执行
"""

import json
import time
import urllib.request
import import_path

import HiwonderSDK.TTS as TTS
import HiwonderSDK.ASR as ASR
import kinematics as kinematics
import HiwonderSDK.Board as Board

from ollama_client import ollama_generate
from whisper_client import record_wav, transcribe, listen_and_transcribe

# ====== 可配置参数 ======
OLLAMA_HOST  = "192.168.1.113"
WHISPER_HOST = "192.168.1.113"
OLLAMA_PORT  = 11434
WHISPER_PORT = 5001
MODEL_NAME   = "qwen2.5:1.5b"
TIMEOUT_S    = 5
HEALTH_TIMEOUT = 2  # 健康检查超时秒数

WAKE_ID = 100  # hi

TTS_SIGN_ACK   = ""
TTS_SIGN_READY = "[h0][v10][m3]"

ik = kinematics.IK()
MOTION_IDS = {2, 3, 4, 5}

# ==============================
# 白名单
# ==============================
ALLOWED_ACTIONS = {"forward", "backward", "left", "right"}

ACTION_NAMES = {
    "forward":  "向前走",
    "backward": "向后退",
    "left":     "左移",
    "right":    "右移",
}

ACTION_ID_MAP = {
    "forward":  2,
    "backward": 3,
    "left":     4,
    "right":    5,
}

# 句式判断：包含这些词才像请求
REQUEST_PATTERNS = [
    "帮我", "可以", "能不能", "能否", "请",
    "麻烦", "走", "移", "过来", "去吧", "试试",
]

# ==============================
# 降级层：本地关键词匹配
# 用于 Whisper/Ollama 全部不可达时
# ==============================
LOCAL_MOTION_KEYWORDS = {
    "前进": 2, "向前": 2, "前进": 2,
    "后退": 3, "向后": 3, "退后": 3,
    "左移": 4, "左边": 4, "向左": 4,
    "右移": 5, "右边": 5, "向右": 5,
}

# Ollama system prompt
SYSTEM_PROMPT = """\
你是一个叫 SpiderPi 的六足机器人宠物。
不管用户说什么，你必须只输出一个 JSON 对象，格式如下：
{"action": "", "reply": "回复内容"}

action 只能是以下五种之一：
- "forward"  — 用户想要机器人向前走
- "backward" — 用户想要机器人向后退
- "left"     — 用户想要机器人左移
- "right"    — 用户想要机器人右移
- ""         — 纯对话

reply 是不超过 20 字的中文短句。
只输出 JSON，不要加任何解释文字。
"""

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


# ==============================
# 降级层：健康检查
# ==============================
def check_service(host: str, port: int, path: str = "/health") -> bool:
    """尝试连接服务，可达返回 True，否则 False"""
    try:
        url = f"{{http://{host}}}:{port}{path}"
        urllib.request.urlopen(url, timeout=HEALTH_TIMEOUT)
        return True
    except Exception:
        return False


def check_ollama(host: str, port: int) -> bool:
    """Ollama 用 /api/tags 接口做健康检查"""
    try:
        url = f"{{http://{host}}}:{port}/api/tags"
        urllib.request.urlopen(url, timeout=HEALTH_TIMEOUT)
        return True
    except Exception:
        return False


# ==============================
# 降级层：本地关键词匹配
# ==============================
def local_match_motion(text: str):
    """本地匹配返回动作 ID，未匹配返回 None"""
    for kw, cmd_id in LOCAL_MOTION_KEYWORDS.items():
        if kw in text:
            return cmd_id
    return None


# ==============================
# LLM 层：意图建议
# ==============================
def ask_llm_for_intent(user_text: str):
    prompt = f"{SYSTEM_PROMPT}\n用户说：{user_text}"
    raw = ollama_generate(prompt, host=OLLAMA_HOST, model=MODEL_NAME, timeout_s=TIMEOUT_S)
    if not raw:
        return "", ""
    print(f"[llm] raw: {raw}")
    try:
        start = raw.find('{')
        end   = raw.rfind('}') + 1
        data  = json.loads(raw[start:end])
        return data.get('action', '').strip(), data.get('reply', '').strip()
    except Exception as e:
        print(f"[llm] JSON 解析失败: {e}")
        return "", raw


# ==============================
# 校验层
# ==============================
def is_action_allowed(action: str) -> bool:
    return action in ALLOWED_ACTIONS

def looks_like_request(user_text: str) -> bool:
    return any(p in user_text for p in REQUEST_PATTERNS)


# ==============================
# 确认层
# ==============================
def ask_voice_confirm(action: str) -> bool:
    action_name = ACTION_NAMES.get(action, action)
    speak(f"要我{action_name}吗")
    time.sleep(0.3)
    wav = record_wav(max_seconds=3, silence_seconds=0.8)
    response = transcribe(wav, host=WHISPER_HOST)
    print(f"[confirm] 用户回复: {response}")
    for w in ["不", "否", "算了", "取消", "不要", "不用"]:
        if w in response:
            speak("好的，不动")
            return False
    for w in ["是", "对", "好", "要", "可以"]:
        if w in response:
            return True
    speak("没听清，取消")
    return False


# ==============================
# 主对话入口
# ==============================
def handle_chat():
    speak('我在')

    # --- 健康检查 ---
    whisper_ok = check_service(WHISPER_HOST, WHISPER_PORT)
    ollama_ok  = check_ollama(OLLAMA_HOST, OLLAMA_PORT)

    print(f"[health] whisper={'ok' if whisper_ok else 'OFFLINE'}  ollama={'ok' if ollama_ok else 'OFFLINE'}")

    # ==============================
    # 降级路径：全部离线
    # ==============================
    if not whisper_ok and not ollama_ok:
        speak("服务离线，请直接说动作词")
        # 用 ASR 录一段并尝试本地匹配
        # （这里简化处理：提示用户用硬动作词）
        return

    # ==============================
    # 降级路径：仅 Whisper 离线
    # ==============================
    if not whisper_ok:
        speak("语音识别离线，请直接说动作词")
        return

    # --- Step 1: 录音 → Whisper ---
    user_text = listen_and_transcribe(host=WHISPER_HOST)
    if not user_text:
        speak('我没听清楚')
        return
    print(f'[chat] user: {user_text}')

    # ==============================
    # 降级路径：仅 Ollama 离线
    # ==============================
    if not ollama_ok:
        speak("大脑离线，尝试本地匹配")
        cmd_id = local_match_motion(user_text)
        if cmd_id is not None:
            confirmed = ask_voice_confirm(
                next(k for k, v in ACTION_ID_MAP.items() if v == cmd_id)
            )
            if confirmed:
                handle_motion(cmd_id)
        else:
            speak("本地模式只支持动作指令")
        return

    # --- Step 2: LLM 建议 ---
    suggested_action, reply = ask_llm_for_intent(user_text)
    print(f'[chat] LLM 建议 action={suggested_action!r}, reply={reply!r}')

    # ==============================
    # 降级路径： Ollama 返回了也没回内容
    # ==============================
    if not suggested_action and not reply:
        speak("机器不太清醒，稍后再试")
        return

    # --- Step 3: 白名单 + 句式校验 ---
    if is_action_allowed(suggested_action) and looks_like_request(user_text):
        # --- Step 4: 语音确认 ---
        confirmed = ask_voice_confirm(suggested_action)
        if confirmed:
            handle_motion(ACTION_ID_MAP[suggested_action])
    else:
        if reply:
            speak(reply)
        elif suggested_action and not is_action_allowed(suggested_action):
            print(f'[security] 非法 action: {suggested_action!r}，已拦截')
            speak("我还不能做那个")


# ==============================
# 主循环
# ==============================
while True:
    data = asr.getResult()
    if data:
        print('result:', data)
        if data == WAKE_ID:
            handle_chat()
        elif data in MOTION_IDS:
            handle_motion(data)
    time.sleep(0.01)
