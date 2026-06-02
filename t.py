from openai import OpenAI
from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

from prompt_toolkit import PromptSession
from prompt_toolkit.key_binding import KeyBindings

import json
import logging
import os
import re
import sys
import threading
import time
from pathlib import Path
from typing import Iterable


def get_base_dir() -> Path:
    """返回程序所在目录，兼容 PyInstaller 打包后的 exe。"""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent

    return Path(__file__).resolve().parent


BASE_DIR = get_base_dir()
CURRENT_DIR = Path.cwd()
console = Console()


DEFAULT_CONFIG = {
    "api_base_url": "https://api.deepseek.com",
    "models": ["deepseek-v4-pro", "deepseek-v4", "deepseek-v3"],
    "temperature": 0.7,
    "timeout": 60,
    "retry_times": 2,
    "knowledge_path": "knowledge_test.txt",
    "max_history_messages": 20,
    "knowledge_top_k": 4,
    "logs_dir": "logs",
}


def load_config() -> dict:
    """读取 config.json；不存在时使用默认配置。"""
    config = DEFAULT_CONFIG.copy()
    for path in (CURRENT_DIR / "config.json", BASE_DIR / "config.json"):
        if not path.exists():
            continue

        try:
            with path.open("r", encoding="utf-8") as f:
                user_config = json.load(f)
            config.update(user_config)
            break
        except Exception as exc:
            console.print(f"[yellow]config.json 读取失败，已使用默认配置：{exc}[/yellow]")

    return config


CONFIG = load_config()


def setup_environment() -> None:
    for path in (CURRENT_DIR / ".env", BASE_DIR / ".env"):
        if path.exists():
            load_dotenv(path)
            return

    load_dotenv()


def setup_logging() -> None:
    logs_dir = CURRENT_DIR / CONFIG["logs_dir"]
    logs_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        filename=logs_dir / "assistant.log",
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        encoding="utf-8",
    )


setup_environment()
setup_logging()


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return CURRENT_DIR / path


def load_knowledge() -> str:
    candidates = [
        resolve_path(CONFIG["knowledge_path"]),
        BASE_DIR / CONFIG["knowledge_path"],
    ]

    for path in candidates:
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                return f.read()

    console.print("[yellow]未找到 knowledge_test.txt，知识库内容将为空。[/yellow]")
    return ""


KNOWLEDGE = load_knowledge()


BASE_SYSTEM_PROMPT = """
你是一名大学学习助手。

回答要求：
1. 解答大学问题，通俗易懂，像老师一样讲解。
2. 多举例，使用 Markdown 格式。
3. 如果有代码，请使用 Markdown 代码块。
4. 如果有数学公式，请使用 LaTeX，例如 $E=mc^2$ 或 $$a^2+b^2=c^2$$。

知识库使用规则：
1. 如果本轮问题检索到了知识库片段，必须优先参考这些内容。
2. 可以在知识库基础上补充解释。
3. 如果没有检索到相关内容，再使用你的通用知识回答。

代码文件输出要求：
当用户明确需要完整的 .py、.cpp、.md 文件时：
1. 必须给出完整文件内容。
2. 必须使用代码块，不要省略 import、main 函数、类定义等必要部分。
3. 不要写“其他部分不变”“这里省略”“同上”。
4. 如果用户指定文件名，请在代码块开头写成：
   ```python filename=main.py
   ```cpp filename=main.cpp
   ```markdown filename=README.md
"""


CHAT_FILE = CURRENT_DIR / "chat_history.json"
MAX_HISTORY_MESSAGES = int(CONFIG["max_history_messages"])


def normalize_words(text: str) -> set[str]:
    return set(re.findall(r"[\w\u4e00-\u9fff]+", text.lower()))


def split_knowledge(knowledge: str) -> list[str]:
    chunks = [chunk.strip() for chunk in re.split(r"\n\s*\n", knowledge) if chunk.strip()]
    if chunks:
        return chunks
    return [knowledge.strip()] if knowledge.strip() else []


def retrieve_knowledge(query: str, knowledge: str, top_k: int) -> str:
    """轻量 RAG：按关键词重叠度选出最相关的知识片段。"""
    query_words = normalize_words(query)
    if not query_words:
        return ""

    scored_chunks = []
    for chunk in split_knowledge(knowledge):
        chunk_words = normalize_words(chunk)
        score = len(query_words & chunk_words)
        if score:
            scored_chunks.append((score, len(chunk), chunk))

    scored_chunks.sort(key=lambda item: (-item[0], item[1]))
    return "\n\n".join(chunk for _, _, chunk in scored_chunks[:top_k])


def build_system_prompt(user_input: str) -> str:
    relevant_knowledge = retrieve_knowledge(
        user_input,
        KNOWLEDGE,
        int(CONFIG["knowledge_top_k"]),
    )

    if not relevant_knowledge:
        return BASE_SYSTEM_PROMPT

    return (
        BASE_SYSTEM_PROMPT
        + "\n\n========================\n【本轮相关知识库片段】\n========================\n"
        + relevant_knowledge
    )


def load_chat_history() -> list[dict]:
    if not CHAT_FILE.exists():
        return []

    try:
        with CHAT_FILE.open("r", encoding="utf-8") as f:
            history_messages = json.load(f)

        console.print("[green]历史聊天记录读取成功[/green]")
        return [
            item
            for item in history_messages
            if isinstance(item, dict) and item.get("role") in {"user", "assistant"}
        ]
    except Exception as exc:
        console.print(f"[red]聊天记录读取失败：{exc}[/red]")
        logging.exception("Failed to load chat history")
        return []


def trim_history(history: list[dict]) -> list[dict]:
    if len(history) <= MAX_HISTORY_MESSAGES:
        return history
    return history[-MAX_HISTORY_MESSAGES:]


def save_chat_history(history: list[dict]) -> None:
    with CHAT_FILE.open("w", encoding="utf-8") as f:
        json.dump(trim_history(history), f, ensure_ascii=False, indent=2)


history_messages = trim_history(load_chat_history())


class LLMClient:
    def __init__(self, config: dict):
        self.models = config["models"]
        self.retry_times = int(config["retry_times"])
        self.temperature = float(config["temperature"])
        self.client = OpenAI(
            api_key=os.getenv("API_KEY"),
            base_url=config["api_base_url"],
            timeout=float(config["timeout"]),
        )

    def chat_stream(self, messages: list[dict]) -> Iterable:
        last_error = None

        for model in self.models:
            for attempt in range(1, self.retry_times + 2):
                try:
                    logging.info("Requesting model=%s attempt=%s", model, attempt)
                    return self.client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=self.temperature,
                        stream=True,
                    )
                except Exception as exc:
                    last_error = exc
                    logging.exception("LLM request failed: model=%s attempt=%s", model, attempt)
                    time.sleep(min(2 * attempt, 5))

        raise RuntimeError(f"所有模型请求均失败：{last_error}")


llm_client = LLMClient(CONFIG)


def build_messages(user_input: str, history: list[dict]) -> list[dict]:
    return [
        {"role": "system", "content": build_system_prompt(user_input)},
        *trim_history(history),
        {"role": "user", "content": user_input},
    ]


def multiline_input() -> str:
    kb = KeyBindings()

    @kb.add("enter")
    def _(event):
        event.app.exit(result=event.app.current_buffer.text)

    # @kb.add("s-enter")
    # def _(event):
    #     event.app.current_buffer.insert_text("\n")

    @kb.add("c-j")
    def _(event):
        event.app.current_buffer.insert_text("\n")

    @kb.add("escape", "enter")
    def _(event):
        event.app.current_buffer.insert_text("\n")

    session = PromptSession(multiline=True, key_bindings=kb)
    return session.prompt(
        "Sean: ",
        bottom_toolbar="Enter 发送；Shift+Enter 换行；Ctrl+J/Alt+Enter 备用换行；输入 exit 退出",
    )


class SpaceStopper:
    """监听空格中断 AI 流式输出。"""

    def __init__(self):
        self.stop_event = threading.Event()
        self.thread = None
        self.running = False
        self.old_terminal_settings = None

    def start(self) -> None:
        self.running = True
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._listen, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=0.2)
        self._restore_terminal()

    def stopped(self) -> bool:
        return self.stop_event.is_set()

    def _listen(self) -> None:
        if os.name == "nt":
            self._listen_windows()
        else:
            self._listen_unix()

    def _listen_windows(self) -> None:
        import msvcrt

        while self.running:
            if msvcrt.kbhit() and msvcrt.getwch() == " ":
                self.stop_event.set()
                self.running = False
                break
            time.sleep(0.03)

    def _listen_unix(self) -> None:
        import select
        import termios
        import tty

        try:
            self.old_terminal_settings = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())

            while self.running:
                readable, _, _ = select.select([sys.stdin], [], [], 0.03)
                if readable and sys.stdin.read(1) == " ":
                    self.stop_event.set()
                    self.running = False
                    break
        except Exception:
            logging.exception("Space stopper failed")
        finally:
            self._restore_terminal()

    def _restore_terminal(self) -> None:
        if os.name == "nt" or self.old_terminal_settings is None:
            return

        try:
            import termios

            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_terminal_settings)
            self.old_terminal_settings = None
        except Exception:
            logging.exception("Failed to restore terminal settings")


def render_formatted_reply(text: str) -> None:
    pattern = r"\$\$(.*?)\$\$"
    parts = re.split(pattern, text, flags=re.S)

    for i, part in enumerate(parts):
        if i % 2 == 0:
            if part.strip():
                console.print(Markdown(part))
        else:
            formula = part.strip()
            console.print(
                Panel(
                    Text(formula, style="bold magenta"),
                    title="LaTeX",
                    border_style="magenta",
                )
            )


def extract_filenames_from_user_input(user_input: str) -> list[str]:
    pattern = r"([A-Za-z0-9_\-\u4e00-\u9fff]+?\.(?:py|cpp|md))"
    names = re.findall(pattern, user_input, flags=re.I)

    result = []
    for name in names:
        if name not in result:
            result.append(name)
    return result


def safe_filename(filename: str) -> str:
    filename = filename.strip().strip('"').strip("'")
    filename = filename.replace("\\", "/").split("/")[-1]
    filename = re.sub(r'[<>:"/\\|?*]', "_", filename)
    return filename or "generated.md"


def extract_filename_from_code_info(info: str) -> str | None:
    info = info.strip()

    filename_match = re.search(
        r'filename\s*=\s*["\']?([^"\'\s]+)["\']?',
        info,
        flags=re.I,
    )
    if filename_match:
        return safe_filename(filename_match.group(1))

    direct_match = re.search(
        r"([A-Za-z0-9_\-\u4e00-\u9fff]+?\.(?:py|cpp|md))",
        info,
        flags=re.I,
    )
    if direct_match:
        return safe_filename(direct_match.group(1))

    return None


def language_to_extension(language: str) -> str | None:
    language = language.lower().strip()
    if language in {"python", "py"}:
        return ".py"
    if language in {"cpp", "c++", "cplusplus"}:
        return ".cpp"
    if language in {"markdown", "md"}:
        return ".md"
    return None


def save_code_blocks(reply: str, user_input: str) -> list[Path]:
    code_block_pattern = re.compile(r"```([^\n`]*)\n(.*?)```", flags=re.S)
    matches = list(code_block_pattern.finditer(reply))
    if not matches:
        return []

    requested_names = extract_filenames_from_user_input(user_input)
    requested_index = 0
    saved_files = []
    counters = {".py": 1, ".cpp": 1, ".md": 1}

    for match in matches:
        info = match.group(1).strip()
        code = match.group(2)
        info_parts = info.split()
        language = info_parts[0].lower() if info_parts else ""
        extension = language_to_extension(language)

        if extension is None:
            continue

        filename = extract_filename_from_code_info(info)
        if filename is None and requested_index < len(requested_names):
            candidate = safe_filename(requested_names[requested_index])
            if candidate.lower().endswith(extension):
                filename = candidate
                requested_index += 1

        if filename is None:
            filename = f"generated_{counters[extension]}{extension}"
            counters[extension] += 1

        filename = safe_filename(filename)
        if not filename.lower().endswith(extension):
            filename += extension

        save_path = CURRENT_DIR / filename
        with save_path.open("w", encoding="utf-8", newline="\n") as f:
            f.write(code.rstrip() + "\n")

        saved_files.append(save_path)

    return saved_files


def stream_ai_reply(request_messages: list[dict], user_input: str):
    full_reply = ""
    cancelled = False
    start_time = time.time()

    console.print()
    console.print("[bold cyan]AI 正在思考...[/bold cyan]")
    console.print("[dim]提示：AI 思考时按空格可以中止。[/dim]")

    stopper = SpaceStopper()
    stopper.start()
    stream = None

    try:
        stream = llm_client.chat_stream(request_messages)

        for chunk in stream:
            if stopper.stopped():
                cancelled = True
                try:
                    stream.close()
                except Exception:
                    logging.exception("Failed to close stream after cancellation")
                break

            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta
            if delta.content is not None:
                full_reply += delta.content

    except KeyboardInterrupt:
        cancelled = True
    finally:
        stopper.stop()
        if stream is not None:
            try:
                stream.close()
            except Exception:
                logging.exception("Failed to close stream")

    elapsed = time.time() - start_time
    console.print()
    console.print(f"[dim]思考耗时: {elapsed:.2f} 秒[/dim]")
    logging.info("AI reply completed cancelled=%s elapsed=%.2fs chars=%s", cancelled, elapsed, len(full_reply))

    if cancelled:
        console.print("[bold red]已中止 AI 输出。[/bold red]")
        console.print("[yellow]这次回复没有保存到聊天记录。[/yellow]")
        return full_reply, True, []

    console.print("\n[bold green]AI:[/bold green]\n")
    render_formatted_reply(full_reply)

    saved_files = save_code_blocks(full_reply, user_input)
    if saved_files:
        console.print()
        console.print("[bold green]已自动保存文件到当前目录：[/bold green]")
        for file_path in saved_files:
            console.print(f"[cyan]{file_path}[/cyan]")

    return full_reply, False, saved_files


def main() -> None:
    global history_messages

    console.print("[bold blue]=== My Assistant ===[/bold blue]")
    console.print("[dim]输入 exit 退出[/dim]")
    console.print("[dim]需要换行：Shift+Enter；如果不生效，用 Ctrl+J 或 Alt+Enter。[/dim]")
    console.print("[dim]AI 思考时按空格可以中止。[/dim]")
    console.print()

    while True:
        try:
            user_input = multiline_input()
        except KeyboardInterrupt:
            console.print("\n[red]已退出。[/red]")
            break

        if user_input.strip().lower() == "exit":
            break

        if not user_input.strip():
            continue

        request_messages = build_messages(user_input, history_messages)

        try:
            reply, cancelled, _ = stream_ai_reply(request_messages, user_input)
            if cancelled:
                continue

            history_messages.extend(
                [
                    {"role": "user", "content": user_input},
                    {"role": "assistant", "content": reply},
                ]
            )
            history_messages = trim_history(history_messages)
            save_chat_history(history_messages)

        except Exception as exc:
            logging.exception("Chat loop failed")
            console.print(f"\n[bold red]出错了：{exc}[/bold red]")


if __name__ == "__main__":
    main()
