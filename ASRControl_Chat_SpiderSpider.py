#!/usr/bin/python3
# coding=utf-8
"""ASRControl_Chat_SpiderSpider.py  v7

变更：
- v5: detect_color 颜色识别
- v6: 报时功能（规则层直接读系统时钟，不过 LLM）
安全架构：LLM 不具备执行权限，只具备建议权，执行权在规则与确认层
降级策略：本地关键词层 → 远端服务层 → TTS 承认失败
- v7: 细节调试
"""

import json
import math
import signal
import subprocess
import sys
import time
import urllib.request
from datetime import datetime
from typing import Optional
sys.path.insert(0, '/home/pi/SpiderPi')
import import_path

import cv2
import numpy as np
import HiwonderSDK.TTS as TTS
import HiwonderSDK.ASR as ASR
import HiwonderSDK.Misc as Misc
import HiwonderSDK.Board as Board
import HiwonderSDK.ActionGroupControl as AGC
import kinematics as kinematics
import Camera
import yaml_handle

from ollama_client import ollama_generate
from whisper_client import record_wav, transcribe, listen_and_transcribe
from drivers.oled_face import OledFace, reply_to_face

# ====== 可配置参数 ======
OLLAMA_HOST  = "192.168.1.6"
WHISPER_HOST = "192.168.1.6"
OLLAMA_PORT  = 11434
WHISPER_PORT = 5001
MODEL_NAME   = "qwen2.5:1.5b"
TIMEOUT_S    = 5
HEALTH_TIMEOUT = 2

WAKE_ID = 100  # zai ma（在吗）
COLOR_ID = 7    # kan yan se（看颜色）
TIME_ID = 8     # ji dian le / xian zai ji dian（几点了）

TTS_SIGN_ACK   = ""
TTS_SIGN_READY = "[h0][v10][m3]"

WALK_GIF = "/home/pi/SpiderPi/assets/walk.gif"
BYE_GIF  = "/home/pi/SpiderPi/assets/bye.gif"

ik = kinematics.IK()
MOTION_IDS = {2, 3, 4, 5, 6}
DIRECT_COMMAND_IDS = MOTION_IDS | {COLOR_ID, TIME_ID}

ALLOWED_ACTIONS = {"forward", "backward", "left", "right", "wave", "detect_color"}

ACTION_NAMES = {
    "forward":       "向前走",
    "backward":      "向后退",
    "left":          "左移",
    "right":         "右移",
    "wave":          "挥手打招呼",
    "detect_color":  "看颜色",
}

ACTION_ID_MAP = {
    "forward":  2,
    "backward": 3,
    "left":     4,
    "right":    5,
    "wave":     6,
}

REQUEST_PATTERNS = [
    "帮我", "可以", "能不能", "能否", "请",
    "麻烦", "走", "移", "过来", "去吧", "试试",
]

# 繁体→简体常用字映射（Whisper 有时输出繁体）
_TRAD_TO_SIMP = str.maketrans(
    "後來說這時語無妳們點請為麼頭嗎吶嚮",
    "后来说这时语无你们点请为么头吗呐向"
)

def to_simplified(text: str) -> str:
    """将 Whisper 可能输出的繁体字转为简体，提高关键词匹配率"""
    return text.translate(_TRAD_TO_SIMP)

LOCAL_MOTION_KEYWORDS = {
    "前进": 2, "向前": 2, "往前": 2, "向前走": 2, "往前走": 2, "前面走": 2,
    "后退": 3, "向后": 3, "退后": 3, "往后": 3, "向后走": 3, "往后退": 3,
    "左移": 4, "左边": 4, "向左": 4, "往左": 4, "向左走": 4, "左移一下": 4,
    "右移": 5, "右边": 5, "向右": 5, "往右": 5, "向右走": 5, "右移一下": 5,
    "你好": 6, "嗨": 6, "哈喽": 6,
}

COLOR_NAME_ZH = {
    "red":   "红色",
    "green": "绿色",
    "blue":  "蓝色",
}

TIME_QUERY_WORDS = ["几点", "时间", "现在", "多少点"]
COLOR_QUERY_WORDS = ["颜色", "什么色", "看一下颜色", "看颜色", "识别颜色", "这是什么"]
FACE_QUERY_WORDS = ["我是谁", "你认识我", "认识我吗", "认得我吗", "识别我", "看我是谁", "看看我是谁", "我的名字"]
DASHBOARD_COLOR_URL = "http://127.0.0.1:5000/api/color"
FACE_STATUS_URL = "http://127.0.0.1:5000/api/status"
FACE_SETTINGS_URL = "http://127.0.0.1:5000/api/settings"
FACE_DASHBOARD_CMD = "cd /home/pi/SpiderPi/Functions && nohup /usr/bin/python3 -u face_dashboard.py >> face_dashboard.log 2>&1 &"

SYSTEM_PROMPT = """你是一个叫 SpiderPi 的六足机器人宠物。
不管用户说什么，你必须只输出一个 JSON 对象，格式如下：
{"action": "", "reply": "回复内容"}

action 只能是以下几种之一：
- "forward"       — 用户想要机器人向前走
- "backward"      — 用户想要机器人向后退
- "left"          — 用户想要机器人左移
- "right"         — 用户想要机器人右移
- "wave"          — 用户向机器人打招呼（你好、嗨、hi、hello 等）
- "detect_color"  — 用户想知道面前是什么颜色（这是什么颜色、看看颜色等）
- ""              — 纯对话

reply 是不超过 20 字的简体中文短句。必须使用简体中文，严禁繁体字（如：無→无，時→时，這→这，語→语）。
只输出 JSON，不要加任何解释文字。
"""

try:
    print("[BOOT] init ASR/TTS...", flush=True)
    asr = ASR.ASR()
    tts = TTS.TTS()
    print("[BOOT] init ASR/TTS done", flush=True)

    print("[ASR] programming words...", flush=True)
    asr.eraseWords()
    asr.setMode(1)
    asr.addWords(WAKE_ID, 'zai ma')
    asr.addWords(2, 'wang qian zou')
    asr.addWords(3, 'wang hou zou')
    asr.addWords(4, 'xiang zuo zou')
    asr.addWords(5, 'xiang you zou')
    asr.addWords(6, 'ni hao')
    asr.addWords(COLOR_ID, 'kan yan se')
    asr.addWords(COLOR_ID, 'shen me yan se')
    asr.addWords(TIME_ID, 'ji dian le')
    asr.addWords(TIME_ID, 'xian zai ji dian')
    asr.getResult()
    tts.TTSModuleSpeak(TTS_SIGN_READY, '你好')
    time.sleep(2)
    print("[ASR] ready", flush=True)

    Board.setPWMServoPulse(1, 1500, 500)
    Board.setPWMServoPulse(2, 1500, 500)
    ik.stand(ik.initial_pos)
    time.sleep(1)
except Exception:
    print('传感器初始化出错')
    raise

try:
    oled = OledFace()
    oled.show_face("happy")
except Exception as e:
    print(f'[oled] 初始化失败，跳过: {e}')
    oled = None


TTS_MAX_CHARS = 13
_SPLIT_PUNC   = '。！？；\n'
_SOFT_PUNC    = '，、'

# ====== 状态机 ======
_status = "IDLE"  # 当前状态，显示在 OLED 黄色带

def set_status(s: str, face: str = "happy"):
    """更新状态并刷新 OLED 显示"""
    global _status
    _status = s
    print(f"[status] → {s}", flush=True)
    if oled is not None:
        oled.show_face(face, status=s)
# ====================


def _split_for_tts(text: str):
    import re
    parts = re.split(f'([{_SPLIT_PUNC}])', text)
    segments = []
    buf = ""
    for tok in parts:
        if tok in _SPLIT_PUNC:
            buf += tok
        else:
            buf += tok
        if buf and (tok in _SPLIT_PUNC or len(buf) >= TTS_MAX_CHARS):
            segments.append(buf.strip())
            buf = ""
    if buf.strip():
        segments.append(buf.strip())

    result = []
    for seg in segments:
        while len(seg) > TTS_MAX_CHARS:
            cut = TTS_MAX_CHARS
            for i in range(min(len(seg)-1, TTS_MAX_CHARS), 0, -1):
                if seg[i] in _SOFT_PUNC:
                    cut = i + 1
                    break
            result.append(seg[:cut].strip())
            seg = seg[cut:].strip()
        if seg:
            result.append(seg)
    return [s for s in result if s]


def speak(text: str, sign: str = TTS_SIGN_ACK):
    if not text:
        return
    # 繁体字兜底：GB2312 编不了就换成备用回复，防止 TTS 崩溃
    try:
        text.encode("gb2312")
    except UnicodeEncodeError:
        print(f"[tts] 繁体字过滤，原文: {text!r}", flush=True)
        text = "我没听懂"
    face = reply_to_face(text)
    if oled is not None:
        oled.show_face(face, status=_status)
    try:
        asr.setMode(0)
        tts.TTSModuleSpeak(sign, text)
        time.sleep(max(1.2, min(2.6, 0.18 * len(text))))
    finally:
        asr.setMode(1)


def speak_long(text: str):
    for chunk in _split_for_tts(text):
        print(f'[tts] chunk: {chunk!r}', flush=True)
        speak(chunk)


def handle_motion(cmd_id: int):
    set_status("MOVE", face="happy")
    if oled is not None and cmd_id != 6:  # wave 不播 walk gif
        oled.play_gif(WALK_GIF, loops=1)
    time.sleep(0.2)
    if cmd_id == 2:
        ik.go_forward(ik.initial_pos, 2, 100, 80, 2)
    elif cmd_id == 3:
        ik.back(ik.initial_pos, 2, 100, 80, 2)
    elif cmd_id == 4:
        ik.left_move(ik.initial_pos, 2, 100, 80, 2)
    elif cmd_id == 5:
        ik.right_move(ik.initial_pos, 2, 100, 80, 2)
    elif cmd_id == 6:
        AGC.runActionGroup('wave')
        speak("你好")
    set_status("CHAT")


# ====== 颜色识别 ======
def detect_color_via_dashboard():  # type: () -> Optional[str]
    """Use the face dashboard's already-open camera frame to avoid camera conflicts."""
    try:
        req = urllib.request.Request(DASHBOARD_COLOR_URL, method="POST")
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        print(f"[color] dashboard unavailable: {e}", flush=True)
        return None

    if not data.get("ok"):
        print(f"[color] dashboard no result: {data}", flush=True)
        return None
    color = data.get("color_zh") or COLOR_NAME_ZH.get(data.get("color"))
    if color:
        print(f"[color] dashboard result: {color}", flush=True)
    return color


def _post_json(url: str, payload: dict, timeout: float = 3.0):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def stop_face_dashboard():
    """Keep ASR as the main process; face recognition is opened only on demand."""
    try:
        subprocess.call(["pkill", "-TERM", "-f", "face_dashboard.py"])
        time.sleep(1.0)
    except Exception as e:
        print(f"[face] stop failed: {e}", flush=True)


def start_face_dashboard():
    if subprocess.call("pgrep -f '[f]ace_dashboard.py' >/dev/null 2>&1", shell=True) != 0:
        print("[face] starting dashboard", flush=True)
        subprocess.Popen(["bash", "-lc", FACE_DASHBOARD_CMD])
        time.sleep(2.5)

    deadline = time.time() + 12
    last_error = ""
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(FACE_STATUS_URL, timeout=2) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            if data.get("camera_open"):
                return True
        except Exception as e:
            last_error = str(e)
        time.sleep(0.6)
    print(f"[face] dashboard not ready: {last_error}", flush=True)
    return False


def get_face_status():
    with urllib.request.urlopen(FACE_STATUS_URL, timeout=2) as resp:
        return json.loads(resp.read().decode("utf-8"))


def handle_face_query():
    set_status("VISION", face="hmm")
    speak("我看一下")
    if not start_face_dashboard():
        speak("我现在看不清")
        set_status("CHAT")
        return

    try:
        deadline = time.time() + 18
        best = None
        while time.time() < deadline:
            try:
                status = get_face_status()
            except Exception as e:
                print(f"[face] status failed: {e}", flush=True)
                time.sleep(0.6)
                continue

            identity = status.get("identity")
            stable = int(status.get("stable") or 0)
            target = int(status.get("stable_target") or 7)
            state = status.get("state")
            print(f"[face] state={state} identity={identity} stable={stable}/{target}", flush=True)

            if identity and identity not in ("no_face", "uncertain", "no_model"):
                best = status
            if state == "locked" and stable >= max(1, target - 1) and best:
                break
            time.sleep(0.7)

        if not best:
            speak("我没有看到人脸")
            return

        identity = best.get("identity")
        if identity == "unknown":
            speak("我还不认识你")
        elif best.get("is_known"):
            speak(f"你是{identity}")
        else:
            speak("我不太确定")
    finally:
        set_status("CHAT")


def detect_color_once():  # type: () -> Optional[str]
    """按需打开摄像头，采样识别主色，返回中文颜色名；无法识别返回 None"""
    try:
        lab_data = yaml_handle.get_yaml_data(yaml_handle.lab_file_path)
    except Exception as e:
        print(f"[color] 加载 lab_data 失败: {e}")
        return None

    cam = Camera.Camera()
    cam.camera_open()
    time.sleep(0.5)

    result = None
    try:
        for _ in range(15):
            img = cam.frame
            if img is None:
                time.sleep(0.05)
                continue
            frame = cv2.resize(img, (320, 240), interpolation=cv2.INTER_NEAREST)
            frame_gb = cv2.GaussianBlur(frame, (3, 3), 3)
            frame_lab = cv2.cvtColor(frame_gb, cv2.COLOR_BGR2LAB)

            max_area = 0
            best_color = None
            for color_name, ranges in lab_data.items():
                if color_name in ('black', 'white'):
                    continue
                mask = cv2.inRange(
                    frame_lab,
                    tuple(ranges['min']),
                    tuple(ranges['max'])
                )
                mask = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
                mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
                contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2]
                for c in contours:
                    area = math.fabs(cv2.contourArea(c))
                    if area > max_area and area >= 100:
                        max_area = area
                        best_color = color_name

            if best_color and max_area > 500:
                result = COLOR_NAME_ZH.get(best_color)
                print(f"[color] 识别结果: {best_color} ({max_area:.0f}px)")
                break
            time.sleep(0.05)
    finally:
        cam.camera_close()

    return result
# ======================


def handle_color_query():
    set_status("SCAN", face="hmm")
    speak("我看一下")
    color = detect_color_via_dashboard() or detect_color_once()
    if color:
        speak(f"这是{color}")
    else:
        speak("没识别到颜色")
    set_status("CHAT")


def handle_time_query():
    now = datetime.now()
    time_str = f"现在{now.hour}点{now.minute:02d}分"
    print(f"[time] {time_str}", flush=True)
    speak(time_str)
    return time_str


def check_service(host: str, port: int, path: str = "/health") -> bool:
    try:
        url = f"http://{host}:{port}{path}"
        urllib.request.urlopen(url, timeout=HEALTH_TIMEOUT)
        return True
    except Exception:
        return False


def check_ollama(host: str, port: int) -> bool:
    try:
        url = f"http://{host}:{port}/api/tags"
        urllib.request.urlopen(url, timeout=HEALTH_TIMEOUT)
        return True
    except Exception:
        return False


def local_match_motion(text: str):
    for kw, cmd_id in LOCAL_MOTION_KEYWORDS.items():
        if kw in text:
            return cmd_id
    return None


_chat_history: list = []


def ask_llm_with_history(user_text: str):
    history_lines = ""
    for msg in _chat_history[-6:]:
        role = "用户" if msg["role"] == "user" else "SpiderPi"
        history_lines += f"{role}：{msg['content']}\n"
    prompt = f"{SYSTEM_PROMPT}\n{history_lines}用户：{user_text}"
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


def is_action_allowed(action: str) -> bool:
    return action in ALLOWED_ACTIONS

def looks_like_request(user_text: str) -> bool:
    return any(p in user_text for p in REQUEST_PATTERNS)


def ask_voice_confirm(action: str) -> bool:
    set_status("CONFIRM", face="hmm")
    action_name = ACTION_NAMES.get(action, action)
    speak(f"要我{action_name}吗")
    time.sleep(0.3)
    wav = record_wav(max_seconds=5, silence_seconds=3)
    response = transcribe(wav, host=WHISPER_HOST)
    print(f"[confirm] 用户回复: {response}")
    for w in ["不", "否", "算了", "取消", "不要", "不用"]:
        if w in response:
            speak("好的，不动")
            set_status("CHAT")
            return False
    for w in ["是", "对", "好", "要", "可以"]:
        if w in response:
            set_status("CHAT")
            return True
    speak("没听清，取消")
    set_status("CHAT")
    return False


FAREWELL_WORDS = ["再见", "拜拜", "不聊了", "好了", "没事了", "结束"]


def _say_bye():
    """播放再见动画 + 说再见"""
    if oled is not None:
        oled.play_gif(BYE_GIF, loops=1)
        time.sleep(0.3)
    speak("再见")


def handle_chat():
    global _chat_history
    _chat_history = []

    set_status("CHAT", face="happy")
    speak('我在')
    if oled is not None:
        oled.blink("happy", status="CHAT")

    whisper_ok = check_service(WHISPER_HOST, WHISPER_PORT)
    ollama_ok  = check_ollama(OLLAMA_HOST, OLLAMA_PORT)
    print(f"[health] whisper={'ok' if whisper_ok else 'OFFLINE'}  ollama={'ok' if ollama_ok else 'OFFLINE'}")

    if not whisper_ok and not ollama_ok:
        set_status("OFFLINE", face="sad")
        speak("服务离线，请直接说动作词")
        _awake_until = time.time() + 5   # 说完后重新给 5 秒
        return
    if not whisper_ok:
        set_status("OFFLINE", face="sad")
        speak("语音识别离线，请直接说动作词")
        _awake_until = time.time() + 5   # 说完后重新给 5 秒
        return

    while True:
        user_text = listen_and_transcribe(host=WHISPER_HOST, silence_seconds=3)
        if not user_text:
            _say_bye()  # 超时退出
            break
        user_text = to_simplified(user_text)  # 繁→简，防止 Whisper 输出繁体导致匹配失败
        print(f'[chat] user: {user_text}')

        if any(w in user_text for w in FAREWELL_WORDS):
            _say_bye()  # 用户主动说再见
            break

        if any(w in user_text for w in FACE_QUERY_WORDS):
            handle_face_query()
            _chat_history.append({"role": "user", "content": user_text})
            _chat_history.append({"role": "assistant", "content": "人脸识别"})
            continue

        if any(w in user_text for w in COLOR_QUERY_WORDS):
            handle_color_query()
            _chat_history.append({"role": "user", "content": user_text})
            _chat_history.append({"role": "assistant", "content": "颜色识别"})
            continue

        # ====== 报时：规则层拦截，不过 LLM ======
        if any(w in user_text for w in TIME_QUERY_WORDS):
            time_str = handle_time_query()
            _chat_history.append({"role": "user", "content": user_text})
            _chat_history.append({"role": "assistant", "content": time_str})
            continue
        # ======================================

        cmd_id = local_match_motion(user_text)
        if cmd_id is not None:
            print(f"[local] direct motion cmd_id={cmd_id} from {user_text!r}", flush=True)
            speak("好的")
            handle_motion(cmd_id)
            _chat_history.append({"role": "user", "content": user_text})
            _chat_history.append({"role": "assistant", "content": "执行动作"})
            continue

        if not ollama_ok:
            set_status("OFFLINE", face="sad")
            speak("大脑离线，尝试本地匹配")
            speak("本地模式只支持动作指令")
            continue

        suggested_action, reply = ask_llm_with_history(user_text)
        print(f'[chat] LLM 建议 action={suggested_action!r}, reply={reply!r}')

        if not suggested_action and not reply:
            speak("我没太听懂")
            continue

        _chat_history.append({"role": "user", "content": user_text})
        _chat_history.append({"role": "assistant", "content": reply or suggested_action})

        # detect_color 直接执行，不走确认流程
        if suggested_action == "detect_color":
            handle_color_query()

        elif suggested_action == "wave" and any(w in user_text.lower() for w in ["你好", "嗨", "哈喽", "hi", "hello"]):
            if reply:
                speak_long(reply)
            handle_motion(ACTION_ID_MAP[suggested_action])

        elif is_action_allowed(suggested_action) and looks_like_request(user_text):
            confirmed = ask_voice_confirm(suggested_action)
            if confirmed:
                handle_motion(ACTION_ID_MAP[suggested_action])
        else:
            if reply:
                speak_long(reply)
            elif suggested_action and not is_action_allowed(suggested_action):
                print(f'[security] 非法 action: {suggested_action!r}，已拦截')
                speak("我还不能做那个")

    set_status("IDLE")  # 退出对话，回到待机


def _graceful_exit(signum, frame):
    print("\n[EXIT] 正常退出", flush=True)
    try:
        asr.setMode(0)
    except Exception:
        pass
    sys.exit(0)

signal.signal(signal.SIGINT, _graceful_exit)

set_status("IDLE")  # 启动完成，进入待机
_awake_until = 0.0

while True:
    data = asr.getResult()
    if data:
        print('result:', data, flush=True)
        if data == WAKE_ID:
            _awake_until = time.time() + 10
            handle_chat()
            time.sleep(0.3)
            asr.setMode(1)
        elif data in DIRECT_COMMAND_IDS:
            if time.time() < _awake_until:
                if data == COLOR_ID:
                    handle_color_query()
                elif data == TIME_ID:
                    handle_time_query()
                else:
                    handle_motion(data)
                _awake_until = 0.0
                time.sleep(0.3)
                asr.setMode(1)
            else:
                print(f'[guard] result {data} ignored, not awake', flush=True)
    time.sleep(0.01)
