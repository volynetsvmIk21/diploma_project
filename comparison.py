import time
import csv
import tkinter as tk
from tkinter import messagebox


REPEATS = 3

TEST_SCENARIOS = [
    {
        "scenario": "Зліт і рух вперед",
        "expected_input": "take off and move forward for five seconds",
        "manual_actions": 2,
    },
    {
        "scenario": "Зліт, рух вперед, поворот",
        "expected_input": "take off and move forward for five seconds and turn left",
        "manual_actions": 3,
    },
    {
        "scenario": "Зліт, рух вперед, поворот, посадка",
        "expected_input": "take off and move forward for five seconds and turn left and land",
        "manual_actions": 4,
    },
    {
        "scenario": "Багатоетапна команда з двома рухами",
        "expected_input": "take off and move forward for five seconds and turn left and move forward for three seconds and land",
        "manual_actions": 5,
    },
]


class ManualNaturalTextTestApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Тестування ручного введення комплексних команд")
        self.root.geometry("1000x500")

        self.results = []

        self.scenario_index = 0
        self.repeat_index = 1
        self.start_time = None
        self.current_scenario = None

        self.title_label = tk.Label(
            root,
            text="Тестування ручного введення комплексної команди природною мовою",
            font=("Times New Roman", 18, "bold")
        )
        self.title_label.pack(pady=10)

        self.info_label = tk.Label(
            root,
            text="Натисніть кнопку, щоб почати перший сценарій.",
            font=("Times New Roman", 13),
            wraplength=920,
            justify="center"
        )
        self.info_label.pack(pady=10)

        self.expected_label = tk.Label(
            root,
            text="",
            font=("Consolas", 13),
            wraplength=920,
            justify="center"
        )
        self.expected_label.pack(pady=10)

        self.command_entry = tk.Entry(
            root,
            font=("Consolas", 15),
            width=90,
            state="disabled"
        )
        self.command_entry.pack(pady=10)
        self.command_entry.bind("<Return>", self.finish_scenario)

        self.timer_label = tk.Label(
            root,
            text="Час: 0.000 с",
            font=("Times New Roman", 14)
        )
        self.timer_label.pack(pady=5)

        self.status_label = tk.Label(
            root,
            text="",
            font=("Times New Roman", 12),
            wraplength=920,
            justify="center"
        )
        self.status_label.pack(pady=10)

        self.start_button = tk.Button(
            root,
            text="Почати сценарій",
            font=("Times New Roman", 14),
            command=self.start_scenario
        )
        self.start_button.pack(pady=10)

        self.save_button = tk.Button(
            root,
            text="Зберегти результати",
            font=("Times New Roman", 12),
            command=self.save_results,
            state="disabled"
        )
        self.save_button.pack(pady=5)

    def start_scenario(self):
        if self.scenario_index >= len(TEST_SCENARIOS):
            self.save_button.config(state="normal")
            messagebox.showinfo("Готово", "Усі сценарії виконано.")
            return

        self.current_scenario = TEST_SCENARIOS[self.scenario_index]
        self.start_time = time.perf_counter()

        scenario_name = self.current_scenario["scenario"]
        expected_input = self.current_scenario["expected_input"]

        self.info_label.config(
            text=(
                f"Сценарій: {scenario_name}\n"
                f"Повтор: {self.repeat_index} з {REPEATS}\n\n"
                "Введіть повну команду природною мовою одним рядком і натисніть Enter."
            )
        )

        self.expected_label.config(
            text=f"Очікуваний текст:\n{expected_input}"
        )

        self.status_label.config(text="Таймер запущено. Вводьте команду.")
        self.command_entry.config(state="normal")
        self.command_entry.delete(0, tk.END)
        self.command_entry.focus()

        self.start_button.config(state="disabled")
        self.update_timer()

    def update_timer(self):
        if self.start_time is not None:
            elapsed = time.perf_counter() - self.start_time
            self.timer_label.config(text=f"Час: {elapsed:.3f} с")
            self.root.after(50, self.update_timer)

    def normalize_input(self, text):
        return " ".join(text.lower().strip().split())

    def count_words(self, text):
        normalized = self.normalize_input(text)
        if not normalized:
            return 0
        return len(normalized.split())

    def finish_scenario(self, event=None):
        if self.current_scenario is None or self.start_time is None:
            return

        end_time = time.perf_counter()
        elapsed_time = round(end_time - self.start_time, 3)

        entered_text = self.normalize_input(self.command_entry.get())
        expected_text = self.normalize_input(self.current_scenario["expected_input"])

        if entered_text == expected_text:
            status = "коректно"
        elif entered_text:
            status = "з помилками"
        else:
            status = "порожнє введення"

        word_count = self.count_words(expected_text)

        if elapsed_time > 0:
            actual_wpm = round((word_count / elapsed_time) * 60, 1)
        else:
            actual_wpm = 0

        result = {
            "Сценарій": self.current_scenario["scenario"],
            "Повтор": self.repeat_index,
            "Очікуваний текст": expected_text,
            "Введений текст": entered_text,
            "Кількість ручних дій": self.current_scenario["manual_actions"],
            "Кількість слів у команді": word_count,
            "Час ручного введення, с": elapsed_time,
            "Фактична швидкість введення, WPM": actual_wpm,
            "Статус": status,
        }

        self.results.append(result)

        self.command_entry.config(state="disabled")
        self.start_time = None

        self.status_label.config(
            text=(
                f"Сценарій завершено.\n"
                f"Час ручного введення: {elapsed_time} с\n"
                f"Фактична швидкість: {actual_wpm} WPM\n"
                f"Статус: {status}"
            )
        )
        self.timer_label.config(text=f"Час: {elapsed_time:.3f} с")

        self.repeat_index += 1

        if self.repeat_index > REPEATS:
            self.repeat_index = 1
            self.scenario_index += 1

        if self.scenario_index >= len(TEST_SCENARIOS):
            self.info_label.config(
                text="Усі сценарії завершено. Натисніть “Зберегти результати”."
            )
            self.expected_label.config(text="")
            self.start_button.config(state="disabled")
            self.save_button.config(state="normal")
        else:
            self.start_button.config(state="normal")
            self.start_button.config(text="Почати наступний сценарій")

    def save_results(self):
        if not self.results:
            messagebox.showwarning("Немає даних", "Немає результатів для збереження.")
            return

        output_file = "manual_natural_text_input_results.csv"

        with open(output_file, "w", newline="", encoding="utf-8-sig") as file:
            writer = csv.DictWriter(file, fieldnames=self.results[0].keys())
            writer.writeheader()
            writer.writerows(self.results)

        messagebox.showinfo(
            "Збережено",
            f"Результати збережено у файл:\n{output_file}"
        )


if __name__ == "__main__":
    root = tk.Tk()
    app = ManualNaturalTextTestApp(root)
    root.mainloop()