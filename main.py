import json
import time
import queue
import threading
import concurrent.futures
from typing import Optional, List

import airsim
import ollama
import sounddevice as sd
from vosk import Model, KaldiRecognizer
from rapidfuzz import process, fuzz

OLLAMA_MODEL    = "qwen2.5:1.5b"
VOSK_MODEL_PATH = "model"
SAMPLE_RATE     = 16000
BLOCKSIZE       = 800
FORWARD_SPEED   = 3.0
SIDE_SPEED      = 2.5
LOOP_DT         = 0.03
FUZZY_THRESHOLD = 65
LLM_TIMEOUT     = 10.0

TRIGGER_WORD = {
    "drone",
    "road",
    "joe",
    "though",
    "troll",
    "hello",
}

TIME_KEYWORDS = {
    "second", "seconds", "sec", "minute", "minutes", "min",
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "for", "during", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
}

WORD_TO_NUM = {
    "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
}

COMMAND_MAP = {
    "forward": {"action": "set_direction", "direction": "forward"},
    "move forward": {"action": "set_direction", "direction": "forward"},
    "go forward": {"action": "set_direction", "direction": "forward"},
    "back": {"action": "set_direction", "direction": "backward"},
    "backward": {"action": "set_direction", "direction": "backward"},
    "go back": {"action": "set_direction", "direction": "backward"},
    "left": {"action": "set_direction", "direction": "left"},
    "go left": {"action": "set_direction", "direction": "left"},
    "right": {"action": "set_direction", "direction": "right"},
    "go right": {"action": "set_direction", "direction": "right"},
    "stop": {"action": "stop"},
    "take off": {"action": "takeoff"},
    "takeoff": {"action": "takeoff"},
    "land": {"action": "land"},
    "turn left": {"action": "turn", "yaw_delta_deg": -45},
    "turn right": {"action": "turn", "yaw_delta_deg": 45},
}

class DroneState:

    def __init__(self):

        self.lock = threading.Lock()
        self.current_direction = "idle"
        self.speed = FORWARD_SPEED
        self.takeoff_requested = False
        self.did_takeoff = False
        self.land_requested = False
        self.is_turning = False
        self.target_yaw = 0.0
        self.emergency_stop = False
        self.running = True
        self.timed_move_direction = None
        self.timed_move_until = 0.0
        self.cmd_queue: List[dict] = []
        self.cmd_executing = False

def normalize(text: str) -> str:
    t = text.lower().strip()

    replacements = {
        "take of": "take off",
        "takeof": "take off",
        "they of": "take off",
        "they off": "take off",
        "they are": "take off",
        "they call": "take off",
        "day of": "take off",

        "lend": "land",
        "lent": "land",
        "lead": "land",
        "live": "land",
        "len": "land",
        "let": "land",
        "like": "land",

        "move for word": "move forward",
        "for word": "forward",
        "what about": "move forward",

        "so low": "turn left",
        "song low": "turn left",
        "thorn low": "turn left",

        "arrive i": "turn right",
        "thorn arrive": "turn right",
        "the arrive": "turn right",

        "move where": "move left",

        "road": "drone",
        "though": "drone",
        "troll": "drone",
        "hello": "drone",
        "joe": "drone",
    }

    for wrong, correct in replacements.items():
        t = t.replace(wrong, correct)

    return t

def replace_word_numbers(text: str) -> str:
    return " ".join(WORD_TO_NUM.get(w, w) for w in text.split())

def has_time_keywords(text: str) -> bool:
    return bool(set(text.lower().split()) & TIME_KEYWORDS)

def starts_with_trigger(text: str) -> bool:
    words = normalize(text).split()

    if not words:
        return False

    return words[0] in TRIGGER_WORD

def fast_parse(text: str) -> Optional[dict]:

    t = normalize(text)

    if has_time_keywords(t):
        return None

    if "and" in t:
        return None

    if t in COMMAND_MAP:

        return dict(COMMAND_MAP[t])

    if t in {
        "lend",
        "lent",
        "lead",
        "live",
        "len",
    }:

        return {
            "action": "land"
        }

    if len(t.split()) > 3:

        return None

    if len(t) <= 3:

        return None

    BAD_WORDS = {
        "oh",
        "okay",
        "most",
        "part",
        "about",
        "portable",
        "forever",
    }

    if any(
        w in BAD_WORDS
        for w in t.split()
    ):

        return None

    result = process.extractOne(
        t,
        COMMAND_MAP.keys(),
        scorer=fuzz.partial_ratio
    )

    if not result:
        return None

    match, score, _ = result

    if score < 85:

        return None

    first_input = t.split()[0]

    first_match = match.split()[0]

    first_score = fuzz.ratio(
        first_input,
        first_match
    )

    if first_score < 70:

        return None

    print(
        f"[FUZZY] {t} -> {match} ({score})"
    )

    return dict(COMMAND_MAP[match])

SYSTEM_PROMPT = """You convert drone commands to JSON. Rules:
- Output ONLY a raw JSON array, no markdown, no backticks, no explanation
- Each item has "action" field
- Actions and fields:
  takeoff -> {"action":"takeoff"}
  land -> {"action":"land"}
  stop -> {"action":"stop"}
  move direction -> {"action":"set_direction","direction":"forward|backward|left|right"}
  move direction N sec -> {"action":"move_timed","direction":"...","duration_seconds":N}
  turn left/right -> {"action":"turn","yaw_delta_deg":-45 or 45}

Input: take off
Output: [{"action":"takeoff"}]

Input: move left 3 seconds
Output: [{"action":"move_timed","direction":"left","duration_seconds":3}]

Input: take off and move forward 5 seconds
Output: [{"action":"takeoff"},{"action":"move_timed","direction":"forward","duration_seconds":5}]

Input: turn left then land
Output: [{"action":"turn","yaw_delta_deg":-45},{"action":"land"}]"""


def warmup_llm():
    print("LLM warming up...")
    t0 = time.time()
    try:
        ollama.generate(
            model=OLLAMA_MODEL,
            prompt=f"{SYSTEM_PROMPT}\n\nInput: take off\nOutput:",
            options={"temperature": 0, "num_predict": 20},
        )
        ollama.generate(
            model=OLLAMA_MODEL,
            prompt=f"{SYSTEM_PROMPT}\n\nInput: take off and move forward 5 seconds\nOutput:",
            options={"temperature": 0, "num_predict": 80},
        )
        print(f"[LLM] ready ({time.time()-t0:.1f}s)")
    except Exception as e:
        print(f"[LLM] warmup failed: {e}")


def call_llm(text: str) -> List[dict]:
    text = replace_word_numbers(text)
    try:
        resp = ollama.generate(
            model=OLLAMA_MODEL,
            prompt=f"{SYSTEM_PROMPT}\n\nInput: {text}\nOutput:",
            options={"temperature": 0, "num_predict": 80},
        )
        content = resp["response"].strip()
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()
        start = content.find("[")
        end   = content.rfind("]") + 1
        if start == -1 or end == 0:
            print(f"LLM bad response: {content[:80]}")
            return []
        return json.loads(content[start:end])
    except Exception as e:
        print(f"LLM ERROR: {e}")
        return []

def apply_cmd(cmd: dict, state: DroneState):

    with state.lock:

        a = cmd.get("action")

        if a != "stop" and state.emergency_stop:

            print("[STOP CLEARED]")

            state.emergency_stop = False

        if a == "set_direction":

            state.current_direction = cmd["direction"]

            state.timed_move_direction = None

            state.timed_move_until = 0.0

        elif a == "move_timed":

            direction = cmd.get(
                "direction",
                "forward"
            )

            duration = float(
                cmd.get(
                    "duration_seconds",
                    1.0
                )
            )

            print(
                f"[TIMED] "
                f"{direction} for {duration}s"
            )

            state.timed_move_direction = direction

            state.timed_move_until = (
                time.time() + duration
            )

            state.current_direction = "idle"

        elif a == "stop":

            state.emergency_stop = True

            state.current_direction = "idle"

            state.timed_move_direction = None

            state.timed_move_until = 0.0

            state.cmd_queue.clear()

            state.cmd_executing = False

        elif a == "takeoff":

            state.takeoff_requested = True

        elif a == "land":

            state.current_direction = "idle"

            state.land_requested = True

        elif a == "turn":

            state.target_yaw += cmd.get(
                "yaw_delta_deg",
                0
            )

            state.is_turning = True


def enqueue_cmds(cmds: List[dict], state: DroneState):
    with state.lock:
        state.cmd_queue.extend(cmds)


def is_cmd_done(cmd: dict, state: DroneState) -> bool:
    a = cmd.get("action")
    with state.lock:
        if a == "move_timed":
            return time.time() >= state.timed_move_until
        if a == "turn":
            return not state.is_turning
        if a == "takeoff":
            return state.did_takeoff

        return True


audio_q = queue.Queue()
text_q  = queue.Queue()

def audio_callback(indata, frames, time_info, status):
    audio_q.put(bytes(indata))

def vosk_thread(state: DroneState):
    model = Model(VOSK_MODEL_PATH)
    rec   = KaldiRecognizer(model, SAMPLE_RATE)

    with sd.RawInputStream(
        samplerate=SAMPLE_RATE,
        blocksize=BLOCKSIZE,
        dtype="int16",
        channels=1,
        callback=audio_callback
    ):
        while state.running:
            data = audio_q.get()
            partial = json.loads(rec.PartialResult()).get("partial", "")
            if partial and not has_time_keywords(partial) and not starts_with_trigger(partial):
                cmd = fast_parse(partial)
                if cmd:
                    print("[REALTIME]", partial, "->", cmd)
                    apply_cmd(cmd, state)

            if rec.AcceptWaveform(data):
                result = json.loads(rec.Result())
                text = result.get("text", "")
                if text:
                    print("[VOICE]", text)
                    if starts_with_trigger(text):
                        text_q.put(text)
                    elif has_time_keywords(text):
                        text_q.put(text)
                    else:
                        cmd = fast_parse(text)
                        if cmd:
                            apply_cmd(cmd, state)
                        else:
                            text_q.put(text)

def llm_thread(state: DroneState):

    executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=1
    )

    while state.running:

        text = text_q.get()
        clean_text = text.lower().strip()

        if starts_with_trigger(clean_text):

            words = normalize(clean_text).split()
            clean_text = " ".join(words[1:]).strip()

        if not clean_text:
            print("[LLM] empty input")
            continue

        print(
            "[LLM] processing:",
            clean_text
        )

        future = executor.submit(
            call_llm,
            clean_text
        )

        t0 = time.time()

        try:

            cmds = future.result(
                timeout=LLM_TIMEOUT
            )

            elapsed = time.time() - t0

            print(
                f"[LLM] done in "
                f"{elapsed:.2f}s -> {cmds}"
            )

            if not cmds:
                continue

            if len(cmds) == 1:

                apply_cmd(
                    cmds[0],
                    state
                )

            else:

                enqueue_cmds(
                    cmds,
                    state
                )

        except concurrent.futures.TimeoutError:

            print(
                f"[LLM] timeout after "
                f"{time.time()-t0:.2f}s"
            )

        except Exception as e:

            print(
                f"[LLM] error: {e}"
            )

def main():

    state = DroneState()

    warmup_llm()

    client = airsim.MultirotorClient()

    client.confirmConnection()

    threading.Thread(
        target=vosk_thread,
        args=(state,),
        daemon=True
    ).start()

    threading.Thread(
        target=llm_thread,
        args=(state,),
        daemon=True
    ).start()

    client.enableApiControl(True)

    client.armDisarm(True)

    current_queued_cmd = None

    try:

        while state.running:

            with state.lock:

                direction   = state.current_direction
                speed       = state.speed
                takeoff     = (
                    state.takeoff_requested
                    and not state.did_takeoff
                )
                turning     = state.is_turning
                yaw         = state.target_yaw
                emergency   = state.emergency_stop
                timed_dir   = state.timed_move_direction
                timed_until = state.timed_move_until
                land_req    = state.land_requested

            if emergency:

                current_queued_cmd = None

                client.moveByVelocityBodyFrameAsync(
                    0,
                    0,
                    0,
                    LOOP_DT
                ).join()

                time.sleep(LOOP_DT)

                continue

            if current_queued_cmd is not None:

                done = is_cmd_done(
                    current_queued_cmd,
                    state
                )

                if done:

                    print(
                        f"[QUEUE] done: "
                        f"{current_queued_cmd['action']}"
                    )

                    current_queued_cmd = None

            if current_queued_cmd is None:

                with state.lock:

                    if state.cmd_queue:

                        current_queued_cmd = (
                            state.cmd_queue.pop(0)
                        )

                        print(
                            "[QUEUE] executing:",
                            current_queued_cmd
                        )

                if current_queued_cmd is not None:

                    apply_cmd(
                        current_queued_cmd,
                        state
                    )

                    time.sleep(0.05)


            if takeoff:

                client.takeoffAsync().join()

                client.moveToZAsync(-3, 2).join()

                with state.lock:

                    state.did_takeoff = True

                    state.takeoff_requested = False


            if land_req:

                client.landAsync().join()

                with state.lock:

                    state.land_requested = False

                    state.did_takeoff = False


            if turning:

                client.rotateToYawAsync(yaw).join()

                with state.lock:

                    state.is_turning = False

            now = time.time()

            if timed_dir and now < timed_until:

                vx, vy = 0, 0

                if timed_dir == "forward":
                    vx = speed

                if timed_dir == "backward":
                    vx = -speed

                if timed_dir == "left":
                    vy = -SIDE_SPEED

                if timed_dir == "right":
                    vy = SIDE_SPEED

                client.moveByVelocityBodyFrameAsync(
                    vx,
                    vy,
                    0,
                    LOOP_DT
                )

                time.sleep(LOOP_DT)

                continue

            if timed_dir and now >= timed_until:

                with state.lock:

                    state.timed_move_direction = None

                    state.timed_move_until = 0.0

                client.moveByVelocityBodyFrameAsync(
                    0,
                    0,
                    0,
                    LOOP_DT
                ).join()

                time.sleep(LOOP_DT)

                continue

            vx, vy = 0, 0

            if direction == "forward":
                vx = speed

            elif direction == "backward":
                vx = -speed

            elif direction == "left":
                vy = -SIDE_SPEED

            elif direction == "right":
                vy = SIDE_SPEED

            client.moveByVelocityBodyFrameAsync(
                vx,
                vy,
                0,
                LOOP_DT
            )

            time.sleep(LOOP_DT)

    except KeyboardInterrupt:

        pass

    if state.did_takeoff:

        client.landAsync().join()

    client.armDisarm(False)

    client.enableApiControl(False)

if __name__ == "__main__":

    main()