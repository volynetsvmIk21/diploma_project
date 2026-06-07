import os
import sys
import json
import time
import wave
import csv
import concurrent.futures
import tempfile
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from collections import Counter
from dataclasses import dataclass, asdict
from collections import defaultdict

try:
    from vosk import Model, KaldiRecognizer
except ImportError:
    sys.exit("Install vosk: pip install vosk")

try:
    import ollama
except ImportError:
    sys.exit("Install ollama: pip install ollama")

try:
    from rapidfuzz import process, fuzz
except ImportError:
    sys.exit("Install rapidfuzz: pip install rapidfuzz")

DATASETS = {
    "clean": "audio/clean",
    "noise10": "audio/noise10",
    "noise20": "audio/noise20",
    "noise30": "audio/noise30",
    "noise40": "audio/noise40",
}
VOSK_MODEL_PATH = "model"

OLLAMA_MODEL    = "qwen2.5:1.5b"

LLM_TIMEOUT     = 6.0

TRIGGER_WORD    = "drone"

FUZZY_THRESHOLD = 85

EXPECTED_COMMANDS = {
    "take-off.wav": "takeoff",
    "land.wav": "land",
    "stop.wav": "stop",

    "move-forward.wav": "forward",
    "move-backward.wav": "backward",
    "move-left.wav": "left",
    "move-right.wav": "right",

    "turnLeft.wav": "turn_left",
    "turnRight.wav": "turn_right",

    "forward-5sec.wav": "move_timed",
    "left-5secs.wav": "move_timed",

    "drone-moveright-for-2seconds-turnleft.wav": "multi_step",
    "drone-takeoff_moveforward.wav": "multi_step",
    "drone-turnleft-moveforward.wav": "multi_step",
    "drone-back5seconds-land.wav": "multi_step",
}

CATEGORY_NAMES = {
    "takeoff": "Take-off",
    "land": "land",
    "stop": "Stop",

    "forward": "Move forward",
    "backward": "Move backward",
    "left": "Move left",
    "right": "Move right",

    "turn_left": "Turn left",
    "turn_right": "Turn right",

    "move_timed": "Timed move",
    "multi_step": "Multi-step",
}

TIME_KEYWORDS = {
    "second",
    "seconds",
    "sec",
    "minute",
    "minutes",
    "min",

    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",

    "for",
    "during",

    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "10",
}

WORD_TO_NUM = {
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
}

COMMAND_MAP = {

    "forward": {
        "action": "set_direction",
        "direction": "forward"
    },

    "move forward": {
        "action": "set_direction",
        "direction": "forward"
    },

    "go forward": {
        "action": "set_direction",
        "direction": "forward"
    },

    "back": {
        "action": "set_direction",
        "direction": "backward"
    },

    "backward": {
        "action": "set_direction",
        "direction": "backward"
    },

    "go back": {
        "action": "set_direction",
        "direction": "backward"
    },

    "left": {
        "action": "set_direction",
        "direction": "left"
    },

    "go left": {
        "action": "set_direction",
        "direction": "left"
    },

    "right": {
        "action": "set_direction",
        "direction": "right"
    },

    "go right": {
        "action": "set_direction",
        "direction": "right"
    },

    "stop": {
        "action": "stop"
    },

    "take off": {
        "action": "takeoff"
    },

    "takeoff": {
        "action": "takeoff"
    },

    "land": {
        "action": "land"
    },

    "turn left": {
        "action": "turn",
        "yaw_delta_deg": -45
    },

    "turn right": {
        "action": "turn",
        "yaw_delta_deg": 45
    },
}

def normalize(text):

    t = text.lower().strip()

    replacements = {

        "take of": "take off",
        "takeof": "take off",
        "they of": "take off",
        "they are": "take off",
        "they call": "take off",
        "day of": "take off",

        "lend": "land",
        "lent": "land",
        "lead": "land",
        "live": "land",
        "len": "land",
        "like": "land",

        "so low": "turn left",
        "song low": "turn left",
        "thorn low": "turn left",

        "arrive i": "turn right",
        "thorn arrive": "turn right",
        "the arrive": "turn right",

        "move where": "move left",

        "what about": "move forward",

        "road": "drone",
        "though": "drone",
        "troll": "drone",
        "hello": "drone",

    }

    for k, v in replacements.items():

        t = t.replace(k, v)

    return t

def replace_word_numbers(text: str) -> str:

    return " ".join(
        WORD_TO_NUM.get(w, w)
        for w in text.split()
    )

def has_time_keywords(text: str) -> bool:

    return bool(
        set(text.lower().split())
        & TIME_KEYWORDS
    )

TRIGGER_VARIANTS = [
    "drone",
    "road",
    "joe",
    "though",
]

def starts_with_trigger(text: str) -> bool:

    words = text.lower().split()

    if not words:
        return False

    return words[0] in TRIGGER_VARIANTS

def fast_parse(text: str):

    t = normalize(text)

    if t in COMMAND_MAP:

        return dict(COMMAND_MAP[t])

    if has_time_keywords(t):
        return None

    if "and" in t:
        return None

    if t in {
        "land",
        "lend",
        "lent",
        "lead",
        "live",
        "len",
        "let",
        "like",
    }:
        return {
            "action": "land"
        }

    if t in COMMAND_MAP:
        return dict(COMMAND_MAP[t])

    if t in {
        "lend",
        "lead",
        "len",
        "live",
    }:
        return {
            "action": "land"
        }

    if len(t.split()) > 4:
        return None

    if len(t) <= 3 and t not in {
        "land",
        "len",
        "let",
    }:
        return None

 
    BAD_WORDS = {
        "oh",
        "okay",
        "most",
        "part",
        "about",
        "forever",
        "portable",
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

    if result:

        match, score, _ = result

        if score >= 85:

            print(
                f"[FUZZY] {t} -> {match} ({score})"
            )

            return dict(COMMAND_MAP[match])

    return None

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
Output: [{"action":"turn","yaw_delta_deg":-45},{"action":"land"}]
"""

def call_llm(text: str) -> list:

    text = replace_word_numbers(text)

    try:

        resp = ollama.generate(
            model=OLLAMA_MODEL,
            prompt=f"{SYSTEM_PROMPT}\n\nInput: {text}\nOutput:",
            options={
                "temperature": 0,
                "num_predict": 80,
            }
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

            print(f"[LLM] bad response: {content[:80]}")

            return []

        return json.loads(content[start:end])

    except Exception as e:

        print(f"[LLM ERROR] {e}")

        return []

_vosk_model = None

def get_vosk_model():

    global _vosk_model

    if _vosk_model is None:

        print(f"[ASR] loading Vosk model from '{VOSK_MODEL_PATH}' ...")

        _vosk_model = Model(VOSK_MODEL_PATH)

        print("[ASR] model loaded.\n")

    return _vosk_model


from pydub import AudioSegment
from pydub.effects import normalize as audio_normalize

def preprocess_audio(filepath: str):

    audio = AudioSegment.from_wav(filepath)

    audio = audio_normalize(audio)

    audio = audio.high_pass_filter(120)

    audio = audio.low_pass_filter(4000)

    with tempfile.NamedTemporaryFile(
        suffix=".wav",
        delete=False
    ) as tmp:

        temp_path = tmp.name

    audio.export(
        temp_path,
        format="wav"
    )

    return temp_path

def recognize_wav(filepath: str):

    temp_path = preprocess_audio(filepath)
    model = get_vosk_model()
    t0 = time.time()
    wf = wave.open(temp_path, "rb")
    rec = KaldiRecognizer(
        model,
        wf.getframerate()
    )

    while True:
        data = wf.readframes(4000)

        if not data:
            break

        rec.AcceptWaveform(data)

    result = json.loads(
        rec.FinalResult()
    )

    text = result.get("text", "")

    elapsed = round(
        time.time() - t0,
        3
    )

    wf.close()

    try:
        os.remove(temp_path)
    except:
        pass

    return text, elapsed

def label_from_llm(cmds: list) -> str:

    if not cmds:
        return "unknown"

    if len(cmds) > 1:

        actions = {
            c.get("action")
            for c in cmds
        }

        if actions == {
            "move_timed",
            "set_direction"
        }:
            return "move_timed"

        return "multi_step"

    cmd = cmds[0]

    action = cmd.get(
        "action",
        "unknown"
    )

    if action == "move_timed":
        return "move_timed"

    if action == "set_direction":

        return cmd.get(
            "direction",
            "unknown"
        )

    if action == "turn":

        yaw = cmd.get(
            "yaw_delta_deg",
            0
        )

        return (
            "turn_left"
            if yaw < 0
            else "turn_right"
        )

    return action

def run_llm_with_timeout(text: str):

    executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=1
    )

    future = executor.submit(
        call_llm,
        text
    )

    try:

        cmds = future.result(
            timeout=LLM_TIMEOUT
        )

        detected = label_from_llm(cmds)

        path = "llm"

    except concurrent.futures.TimeoutError:

        detected = "timeout"
        path = "timeout"

    finally:

        executor.shutdown(wait=False)

    return detected, path

def detect_command(text: str):

    t0 = time.time()
    norm = normalize(text)

    if starts_with_trigger(norm) and len(norm.split()) >= 4:

        words = norm.split()

        clean = " ".join(words[1:]).strip()

        detected, path = run_llm_with_timeout(clean)

        elapsed = round(
            time.time() - t0,
            3
        )

        return detected, path, elapsed

    if has_time_keywords(norm):
        detected, path = run_llm_with_timeout(norm)

        elapsed = round(
            time.time() - t0,
            3
        )

        return detected, path, elapsed

    cmd = fast_parse(norm)

    if cmd:

        action = cmd.get(
            "action",
            "unknown"
        )

        if action == "set_direction":

            action = cmd.get(
                "direction",
                "unknown"
            )

        elif action == "turn":

            yaw = cmd.get(
                "yaw_delta_deg",
                0
            )

            if yaw < 0:
                action = "turn_left"
            else:
                action = "turn_right"

        elif action in {
            "land",
            "takeoff",
            "stop",
        }:

            pass

        elapsed = round(
            time.time() - t0,
            3
        )

        return action, "fast_parse", elapsed

    elapsed = round(
        time.time() - t0,
        3
    )

    return "unknown", "unknown", elapsed

@dataclass
class TestResult:
    dataset: str
    file: str
    expected: str
    recognized_text: str
    detected: str

    success: bool

    path: str

    asr_time: float
    cmd_time: float
    total_time: float

def run_one(dataset_name: str, filename: str, expected: str):

    filepath = os.path.join(
        AUDIO_FOLDER,
        filename
    )

    if not os.path.exists(filepath):

        print(f"[MISSING] {filename}")

        return None

    recognized_text, asr_time = recognize_wav(filepath)

    detected, path, cmd_time = detect_command(
        recognized_text
    )

    total_time = round(
        asr_time + cmd_time,
        3
    )

    return TestResult(
        dataset=dataset_name,
        file=filename,
        expected=expected,
        recognized_text=recognized_text,
        detected=detected,

        success=(detected == expected),

        path=path,

        asr_time=asr_time,
        cmd_time=cmd_time,
        total_time=total_time,
    )

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def print_header():

    print(
        f"\n{'FILE':32}"
        f"{'EXPECTED':15}"
        f"{'DETECTED':15}"
        f"{'PATH':12}"
        f"{'ASR(s)':10}"
        f"{'CMD(s)':10}"
        f"OK"
    )

    print("─" * 97)

def print_row(r: TestResult):

    ok = (
        f"{GREEN}✓{RESET}"
        if r.success
        else f"{RED}✗{RESET}"
    )

    det = (
        f"{GREEN}{r.detected:15}{RESET}"
        if r.success
        else f"{RED}{r.detected:15}{RESET}"
    )

    print(
        f"{r.file:32}"
        f"{r.expected:15}"
        f"{det}"
        f"{YELLOW}{r.path:12}{RESET}"
        f"{r.asr_time:<10}"
        f"{r.cmd_time:<10}"
        f"{ok}"
    )

    print(
        f"  {CYAN}asr:{RESET} "
        f"{r.recognized_text!r}"
    )

def show_summary(results: list):

    total = len(results)

    correct = sum(
        r.success
        for r in results
    )

    accuracy = (
        correct / total * 100
        if total else 0
    )

    avg_total = (
        sum(r.total_time for r in results)
        / total
    )

    avg_asr = (
        sum(r.asr_time for r in results)
        / total
    )

    avg_cmd = (
        sum(r.cmd_time for r in results)
        / total
    )

    print(f"\n{BOLD}{'=' * 60}{RESET}")

    print(f"{BOLD}SUMMARY{RESET}")

    print("=" * 60)

    print("Total tests :", total)

    print("Correct     :", correct)

    print(
        "Accuracy    :",
        f"{accuracy:.1f}%"
    )

    print(
        f"Avg total   : {avg_total:.3f}s "
        f"(asr {avg_asr:.3f}s + cmd {avg_cmd:.3f}s)"
    )

def save_csv(results: list):

    rows = [
        asdict(r)
        for r in results
    ]

    with open(
        "test_results.csv",
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=rows[0].keys()
        )

        writer.writeheader()

        writer.writerows(rows)

    print("\n[CSV] saved -> test_results.csv")


def run_dataset(dataset_name, folder):

    global AUDIO_FOLDER

    AUDIO_FOLDER = folder

    print(f"\n{'=' * 25}")
    print(f"DATASET: {dataset_name}")
    print(f"{'=' * 25}")

    print_header()

    results = []

    for filename, expected in EXPECTED_COMMANDS.items():

        r = run_one(
            dataset_name,
            filename,
            expected
        )

        if r:

            results.append(r)

            print_row(r)

    if not results:

        print("[ERROR] no results")

        return []

    show_summary(results)

    return results

def generate_graphs(results):

    os.makedirs("results", exist_ok=True)

    df = pd.DataFrame([
        asdict(r)
        for r in results
    ])

    df.to_excel(
        "results/full_results.xlsx",
        index=False
    )

    print("[XLSX] saved -> results/full_results.xlsx")

    dataset_stats = []

    for dataset_name in DATASETS.keys():

        subset = df[
            df["file"].apply(
                lambda x: True
            )
        ]

        total = len(subset)
        correct = subset["success"].sum()
        acc = (
            correct / total * 100
            if total else 0
        )

        dataset_stats.append(
            {
                "dataset": dataset_name,
                "accuracy": acc
            }
        )

    ds_df = pd.DataFrame(dataset_stats)

    plt.figure(figsize=(8, 5))

    plt.plot(
        ds_df["dataset"],
        ds_df["accuracy"],
        marker="o"
    )

    plt.title("Accuracy vs Noise Level")

    plt.xlabel("Dataset")

    plt.ylabel("Accuracy (%)")

    plt.grid(True)

    plt.savefig(
        "results/accuracy_vs_noise.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    path_counts = (
        df["path"]
        .value_counts()
    )

    plt.figure(figsize=(7, 5))
    path_counts.plot(
        kind="bar"
    )

    plt.title("Fast Parser vs LLM Usage")
    plt.xlabel("Path")
    plt.ylabel("Count")
    plt.grid(True)
    plt.savefig(
        "results/parser_usage.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    latency = {

        "ASR": df["asr_time"].mean(),
        "Command": df["cmd_time"].mean(),
        "Total": df["total_time"].mean(),
    }

    plt.figure(figsize=(7, 5))

    plt.bar(
        latency.keys(),
        latency.values()
    )

    plt.title("Average Processing Time")

    plt.ylabel("Seconds")

    plt.grid(True)

    plt.savefig(
        "results/latency.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    category_rows = []

    for category in set(df["expected"]):
        subset = df[
            df["expected"] == category
        ]

        total = len(subset)
        correct = subset["success"].sum()
        acc = (
            correct / total * 100
            if total else 0
        )

        category_rows.append(
            {
                "category": category,
                "accuracy": acc
            }
        )

    cat_df = pd.DataFrame(category_rows)
    cat_df.to_excel(
        "results/category_accuracy.xlsx",
        index=False
    )

    plt.figure(figsize=(10, 5))
    plt.bar(
        cat_df["category"],
        cat_df["accuracy"]
    )
    plt.xticks(rotation=20)
    plt.title("Accuracy by Command Category")
    plt.ylabel("Accuracy (%)")
    plt.grid(True)
    plt.savefig(
        "results/category_accuracy.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    labels = sorted(
        list(
            set(df["expected"])
            | set(df["detected"])
        )
    )

    matrix = np.zeros(
        (
            len(labels),
            len(labels)
        ),
        dtype=int
    )

    label_to_idx = {
        l: i
        for i, l in enumerate(labels)
    }

    for _, row in df.iterrows():

        i = label_to_idx[row["expected"]]

        j = label_to_idx[row["detected"]]

        matrix[i][j] += 1

    plt.figure(figsize=(10, 8))
    plt.imshow(
        matrix,
        interpolation="nearest"
    )

    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.xticks(
        range(len(labels)),
        labels,
        rotation=45
    )
    plt.yticks(
        range(len(labels)),
        labels
    )
    plt.xlabel("Predicted")
    plt.ylabel("Expected")

    for i in range(len(labels)):

        for j in range(len(labels)):

            plt.text(
                j,
                i,
                matrix[i, j],
                ha="center",
                va="center"
            )

    plt.tight_layout()

    plt.savefig(
        "results/confusion_matrix.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    summary = {

        "total_tests":
            len(df),

        "accuracy":
            round(
                df["success"].mean() * 100,
                2
            ),

        "avg_asr":
            round(
                df["asr_time"].mean(),
                3
            ),

        "avg_cmd":
            round(
                df["cmd_time"].mean(),
                3
            ),

        "avg_total":
            round(
                df["total_time"].mean(),
                3
            ),
    }

    pd.DataFrame(
        [summary]
    ).to_excel(
        "results/summary.xlsx",
        index=False
    )

    print("[GRAPH] saved -> results/")

def main():

    all_results = []

    for dataset_name, folder in DATASETS.items():

        dataset_results = run_dataset(
            dataset_name,
            folder
        )

        all_results.extend(
            dataset_results
        )

    save_csv(all_results)

    generate_graphs(all_results)


if __name__ == "__main__":

    main()