# imgdownload

Скрипт `main.py` скачивает главы по ссылкам из `config.yaml` и склеивает картинки в один длинный файл `merged.png`.

После скачивания скрипт **автоматически архивирует** папку тайтла в `.zip`.

## Важно

- Поддерживается: **Naver Comic (в т.ч. залогиненный)**, **Webtoons**, **Mangalib**, **DemonicScans**
- **Newtoki / Manatoki / Booktoki запрещены и не поддерживаются** (скрипт такие ссылки блокирует)

## Установка Python

Нужен **Python 3.10+**.

- Скачать Python (официально): `https://www.python.org/downloads/`
- Windows: `https://www.python.org/downloads/windows/`
- macOS: `https://www.python.org/downloads/macos/`

На Windows при установке отметьте галочку **Add Python to PATH**.

Проверьте в терминале:

```bash
python --version
pip --version
```

## Установка библиотек

В папке проекта:

```bash
python -m venv .venv

# Windows:
.\.venv\Scripts\activate

# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
python -m playwright install chromium
```

Документация Playwright (Python): `https://playwright.dev/python/`

## Настройка

Откройте `config.yaml` и добавьте ссылки в `urls:`.
Форматы для нужных сайтов указаны в закомментированных строках, комментарий ##
Табуляция ВАЖНА! (пробелы), следуйте примерам закомментированным

### Naver (залогиненный)

При первом запуске для Naver откроется окно браузера: войдите в аккаунт и **закройте окно**. Профиль сохранится в `naver_profile/` (можно поменять через `naver_user_data_dir` в `config.yaml`).

## Запуск

```bash
python main.py
```

## Результат

- Склеенный скан: `Downloads/<Title>/Chapter-<N>/merged.png` (или папка из `download_dir`)
- Архив тайтла: `Downloads/<Title>.zip`

