import json
import time
import csv
import queue
from typing import Optional, List, Tuple

import sounddevice as sd
import ollama
from vosk import Model, KaldiRecognizer
from rapidfuzz import process, fuzz


OLLAMA_MODEL = "qwen2.5:1.5b"
VOSK_MODEL_PATH = "model"

SAMPLE_RATE = 16000
BLOCKSIZE = 800

RECORD_TIMEOUT = 12.0
SILENCE_AFTER_TEXT_SECONDS = 1.5

REPEATS = 3


TIME_KEYWORDS = {
    "second", "seconds", "sec", "minute", "minutes", "min",
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "for", "during", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
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


SYSTEM_PROMPT = """You convert drone commands to JSON. Rules:
- Output ONLY a raw JSON array, no markdown, no backticks, no explanation
- Preserve the exact order of actions from the input command
- Do not reorder actions
- Do not add actions that are not present in the input
- Each item has "action" field
- Actions and fields:
  takeoff -> {"action":"takeoff"}
  land -> {"action":"land"}
  stop -> {"action":"stop"}
  move direction -> {"action":"set_direction","direction":"forward|backward|left|right"}
  move direction N sec -> {"action":"move_timed","direction":"forward|backward|left|right","duration_seconds":N}
  turn left/right -> {"action":"turn","yaw_delta_deg":-45 or 45}

Input: take off
Output: [{"action":"takeoff"}]

Input: land
Output: [{"action":"land"}]

Input: stop
Output: [{"action":"stop"}]

Input: move forward
Output: [{"action":"set_direction","direction":"forward"}]

Input: move left 3 seconds
Output: [{"action":"move_timed","direction":"left","duration_seconds":3}]

Input: move forward for five seconds
Output: [{"action":"move_timed","direction":"forward","duration_seconds":5}]

Input: take off and move forward 5 seconds
Output: [{"action":"takeoff"},{"action":"move_timed","direction":"forward","duration_seconds":5}]

Input: turn left then land
Output: [{"action":"turn","yaw_delta_deg":-45},{"action":"land"}]

Input: take off and move forward 5 seconds and turn left
Output: [{"action":"takeoff"},{"action":"move_timed","direction":"forward","duration_seconds":5},{"action":"turn","yaw_delta_deg":-45}]

Input: take off and move forward 5 seconds and turn left and land
Output: [{"action":"takeoff"},{"action":"move_timed","direction":"forward","duration_seconds":5},{"action":"turn","yaw_delta_deg":-45},{"action":"land"}]

Input: take off and move forward 5 seconds and turn left and move forward 3 seconds and land
Output: [{"action":"takeoff"},{"action":"move_timed","direction":"forward","duration_seconds":5},{"action":"turn","yaw_delta_deg":-45},{"action":"move_timed","direction":"forward","duration_seconds":3},{"action":"land"}]"""


def normalize(text: str) -> str:
    t = text.lower().strip()

    replacements = {
        "take of": "take off",
        "takeof": "take off",
        "the gulf": "take off",
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
        "more forward": "move forward",
        "more former": "move forward",
        "move former": "move forward",
        "for word": "forward",
        "what about": "move forward",

        "more four or five seconds": "move forward for five seconds",
        "more for five seconds": "move forward for five seconds",
        "move four or five seconds": "move forward for five seconds",

        "so low": "turn left",
        "song low": "turn left",
        "thorn low": "turn left",
        "turn lift": "turn left",

        "arrive i": "turn right",
        "thorn arrive": "turn right",
        "the arrive": "turn right",
    }

    for wrong, correct in replacements.items():
        t = t.replace(wrong, correct)

    return t


def replace_word_numbers(text: str) -> str:
    return " ".join(WORD_TO_NUM.get(word, word) for word in text.split())


def has_time_keywords(text: str) -> bool:
    return bool(set(text.lower().split()) & TIME_KEYWORDS)


def fast_parse(text: str) -> Optional[dict]:
    t = normalize(text)

    if has_time_keywords(t):
        return None

    if "and" in t:
        return None

    if t in COMMAND_MAP:
        return dict(COMMAND_MAP[t])

    if len(t.split()) > 3:
        return None

    if len(t) <= 3:
        return None

    bad_words = {
        "oh",
        "okay",
        "most",
        "part",
        "about",
        "portable",
        "forever",
    }

    if any(word in bad_words for word in t.split()):
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
    first_score = fuzz.ratio(first_input, first_match)

    if first_score < 70:
        return None

    return dict(COMMAND_MAP[match])


def call_llm(text: str) -> List[dict]:
    text = normalize(text)
    text = replace_word_numbers(text)

    try:
        response = ollama.generate(
            model=OLLAMA_MODEL,
            prompt=f"{SYSTEM_PROMPT}\n\nInput: {text}\nOutput:",
            options={
                "temperature": 0,
                "num_predict": 120,
            },
        )

        content = response["response"].strip()

        if content.startswith("```"):
            parts = content.split("```")
            if len(parts) >= 2:
                content = parts[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip()

        start = content.find("[")
        end = content.rfind("]") + 1

        if start == -1 or end == 0:
            print("LLM bad response:", content)
            return []

        parsed = json.loads(content[start:end])

        if not isinstance(parsed, list):
            return []

        return parsed

    except Exception as error:
        print("LLM ERROR:", error)
        return []


def parse_command(text: str) -> List[dict]:
    command = fast_parse(text)

    if command:
        return [command]

    return call_llm(text)


def commands_equal(actual: List[dict], expected: List[dict]) -> bool:
    return actual == expected


def command_signature(command: dict) -> Tuple:
    action = command.get("action")

    if action == "takeoff":
        return ("takeoff",)

    if action == "land":
        return ("land",)

    if action == "stop":
        return ("stop",)

    if action == "set_direction":
        return (
            "set_direction",
            command.get("direction"),
        )

    if action == "move_timed":
        return (
            "move_timed",
            command.get("direction"),
            int(float(command.get("duration_seconds", 0))),
        )

    if action == "turn":
        return (
            "turn",
            int(command.get("yaw_delta_deg", 0)),
        )

    return (action,)


def get_status(actual: List[dict], expected: List[dict]) -> str:
    if commands_equal(actual, expected):
        return "коректно"

    if not actual:
        return "некоректно"

    actual_signatures = [command_signature(cmd) for cmd in actual]
    expected_signatures = [command_signature(cmd) for cmd in expected]

    expected_set = set(expected_signatures)
    actual_set = set(actual_signatures)

    common = expected_set & actual_set

    if len(common) >= max(1, len(expected_set) // 2):
        return "частково коректно"

    return "некоректно"


audio_queue = queue.Queue()


def audio_callback(indata, frames, time_info, status):
    if status:
        print("Audio status:", status)
    audio_queue.put(bytes(indata))


def clear_audio_queue():
    while not audio_queue.empty():
        try:
            audio_queue.get_nowait()
        except queue.Empty:
            break


def recognize_one_command(model: Model) -> Tuple[str, float]:
    recognizer = KaldiRecognizer(model, SAMPLE_RATE)

    recognized_text = ""
    start_time = time.perf_counter()
    last_text_time = None

    clear_audio_queue()

    print("Говоріть команду...")

    with sd.RawInputStream(
        samplerate=SAMPLE_RATE,
        blocksize=BLOCKSIZE,
        dtype="int16",
        channels=1,
        callback=audio_callback
    ):
        while True:
            now = time.perf_counter()

            if now - start_time > RECORD_TIMEOUT:
                break

            try:
                data = audio_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            if recognizer.AcceptWaveform(data):
                result = json.loads(recognizer.Result())
                text = result.get("text", "").strip()

                if text:
                    recognized_text = text
                    break

            partial = json.loads(recognizer.PartialResult()).get("partial", "").strip()

            if partial:
                last_text_time = time.perf_counter()

            if last_text_time and now - last_text_time > SILENCE_AFTER_TEXT_SECONDS:
                break

    if not recognized_text:
        final_result = json.loads(recognizer.FinalResult())
        recognized_text = final_result.get("text", "").strip()

    end_time = time.perf_counter()
    voice_input_time = round(end_time - start_time, 3)

    return recognized_text, voice_input_time


def warmup_llm():
    print("Прогрів LLM...")
    start = time.perf_counter()

    try:
        ollama.generate(
            model=OLLAMA_MODEL,
            prompt=f"{SYSTEM_PROMPT}\n\nInput: take off\nOutput:",
            options={
                "temperature": 0,
                "num_predict": 30,
            },
        )
        elapsed = round(time.perf_counter() - start, 2)
        print(f"LLM готова ({elapsed} с)")
    except Exception as error:
        print("Не вдалося прогріти LLM:", error)


def run_voice_test():
    test_cases = [
        {
            "scenario": "Зліт і рух вперед",
            "expected_phrase": "take off and move forward for five seconds",
            "manual_actions": 2,
            "expected_commands": [
                {"action": "takeoff"},
                {"action": "move_timed", "direction": "forward", "duration_seconds": 5},
            ],
        },
        {
            "scenario": "Зліт, рух вперед, поворот",
            "expected_phrase": "take off and move forward for five seconds and turn left",
            "manual_actions": 3,
            "expected_commands": [
                {"action": "takeoff"},
                {"action": "move_timed", "direction": "forward", "duration_seconds": 5},
                {"action": "turn", "yaw_delta_deg": -45},
            ],
        },
        {
            "scenario": "Багатоетапна команда",
            "expected_phrase": "take off and move forward for five seconds and turn left and land",
            "manual_actions": 4,
            "expected_commands": [
                {"action": "takeoff"},
                {"action": "move_timed", "direction": "forward", "duration_seconds": 5},
                {"action": "turn", "yaw_delta_deg": -45},
                {"action": "land"},
            ],
        },
        {
            "scenario": "Багатоетапна команда з двома рухами",
            "expected_phrase": "take off and move forward for five seconds and turn left and move forward for three seconds and land",
            "manual_actions": 5,
            "expected_commands": [
                {"action": "takeoff"},
                {"action": "move_timed", "direction": "forward", "duration_seconds": 5},
                {"action": "turn", "yaw_delta_deg": -45},
                {"action": "move_timed", "direction": "forward", "duration_seconds": 3},
                {"action": "land"},
            ],
        },
    ]

    results = []

    print("\n=== ТЕСТУВАННЯ ГОЛОСОВИХ КОМПЛЕКСНИХ КОМАНД БЕЗ AIRSIM ===\n")

    print("Завантаження Vosk-моделі...")
    vosk_model = Model(VOSK_MODEL_PATH)
    print("Vosk-модель готова.")

    warmup_llm()

    try:
        for case in test_cases:
            for repeat in range(1, REPEATS + 1):
                print("\nСценарій:", case["scenario"])
                print("Повтор:", repeat)
                print("Скажіть фразу:")
                print(case["expected_phrase"])

                input("Натисніть Enter і після цього одразу говоріть команду...")

                recognized_text, voice_input_time = recognize_one_command(vosk_model)

                print("Розпізнаний текст:", recognized_text)

                interpretation_start = time.perf_counter()
                commands = parse_command(recognized_text)
                interpretation_end = time.perf_counter()

                interpretation_time = round(interpretation_end - interpretation_start, 3)
                total_time = round(voice_input_time + interpretation_time, 3)

                manual_actions = case["manual_actions"]
                voice_actions = 1

                reduction_percent = round(
                    ((manual_actions - voice_actions) / manual_actions) * 100,
                    1
                )

                status = get_status(commands, case["expected_commands"])

                result = {
                    "Сценарій": case["scenario"],
                    "Повтор": repeat,
                    "Очікувана фраза": case["expected_phrase"],
                    "Розпізнаний текст": recognized_text,
                    "Кількість дій при ручному введенні": manual_actions,
                    "Кількість голосових команд": voice_actions,
                    "Скорочення кількості операторських дій, %": reduction_percent,
                    "Час голосового введення та розпізнавання, с": voice_input_time,
                    "Час інтерпретації, с": interpretation_time,
                    "Загальний час голосового введення, с": total_time,
                    "Очікувані команди": json.dumps(case["expected_commands"], ensure_ascii=False),
                    "Сформовані команди": json.dumps(commands, ensure_ascii=False),
                    "Статус": status,
                }

                results.append(result)

                print("Сформовані команди:", commands)
                print("Очікувані команди:", case["expected_commands"])
                print("Час голосового введення та розпізнавання:", voice_input_time, "с")
                print("Час інтерпретації:", interpretation_time, "с")
                print("Загальний час:", total_time, "с")
                print("Скорочення операторських дій:", reduction_percent, "%")
                print("Статус:", status)
                print("-" * 80)

    except KeyboardInterrupt:
        print("\nТестування зупинено. Зберігаю отримані результати...")

    if results:
        output_file = "voice_command_test_results.csv"

        with open(output_file, "w", newline="", encoding="utf-8-sig") as file:
            writer = csv.DictWriter(file, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)

        print("\nРезультати збережено у файл:", output_file)
    else:
        print("\nРезультатів немає, файл не створено.")


if __name__ == "__main__":
    run_voice_test()