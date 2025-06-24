import json

# Настройки
INPUT_FILE = 'dialogs.json'       # входной файл с твоим JSON-массивом
OUTPUT_FILE = 'dialogs_clean.json'  # куда сохранить отфильтрованный результат

bad_words = [

    # сюда можно добавить свои нежелательные слова
]

MIN_LENGTH = 20      # мин длина строки (в символах)
MAX_LENGTH = 1000    # макс длина строки (в символах)

def contains_bad_words(text):
    text_lower = text.lower()
    return any(bad_word in text_lower for bad_word in bad_words)

def filter_dialogs(dialogs):
    filtered = []
    for dialog in dialogs:
        if contains_bad_words(dialog):
            continue
        if len(dialog) < MIN_LENGTH:
            continue
        if len(dialog) > MAX_LENGTH:
            continue
        filtered.append(dialog)
    return filtered

def main():
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        dialogs = json.load(f)

    filtered_dialogs = filter_dialogs(dialogs)

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(filtered_dialogs, f, ensure_ascii=False, indent=2)

    print(f"Отфильтровано {len(filtered_dialogs)} из {len(dialogs)} записей.")

if __name__ == "__main__":
    main()
