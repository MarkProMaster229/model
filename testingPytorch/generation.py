# dataset_gen_onefile.py
import random
import json
from pathlib import Path

random.seed(42)

ops = ["+", "-", "*", "/", "**"]
examples = []

NUM_EXAMPLES = 5000
NEG_RATIO = 0.2   # доля негативных примеров

def make_expr(a, op, b):
    return f"{a} {op} {b}"

for _ in range(NUM_EXAMPLES):
    if random.random() < NEG_RATIO:
        # Негативный пример — строка без \calc
        a = random.randint(1, 999)
        sentence = random.choice([
            f"У меня {a} яблок.",
            f"В сумке было {a} предметов.",
            f"Число: {a}.",
            f"Я набрал {a} баллов в тесте."
        ])
        s = sentence
    else:
        a, b = random.randint(1, 99), random.randint(1, 99)
        op = random.choice(ops)
        expr = make_expr(a, op, b)
        sentence = random.choice([
            f"Пример: {expr}. Ответ:",
            f"Реши {expr}. Ответ:",
            f"{expr} — это что? Ответ:",
            f"Найди значение {expr}. Ответ:",
            f"Пожалуйста, посчитай: {expr}. Ответ:"
        ])
        s = f"{sentence} \\calc {expr} \\calc"

    examples.append(s)


out_file = Path(__file__).parent / "datasetCalc.json"
with open(out_file, "w", encoding="utf-8") as f:
    json.dump(examples, f, ensure_ascii=False, indent=2)

# sanity
print("Saved file:", out_file)
print("Total examples:", len(examples))
print("Sample 0:", examples[0])
