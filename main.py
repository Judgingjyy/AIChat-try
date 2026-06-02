from openai import OpenAI
from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

from prompt_toolkit import PromptSession
from prompt_toolkit.key_binding import KeyBindings

import time
import os
import json
import re
import sys
import threading


# =========================
# 基础路径
# =========================
def get_base_dir():
    """
    返回程序所在目录。
    普通运行：返回 main.py 所在目录。
    PyInstaller 打包后：返回 exe 所在目录。
    """
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)

    return os.path.dirname(os.path.abspath(__file__))


BASE_DIR = get_base_dir()
CURRENT_DIR = os.getcwd()


# =========================
# Rich 控制台
# =========================
console = Console()


# =========================
# 读取 .env
# 优先读取当前目录，其次读取程序目录
# =========================
env_path_current = os.path.join(CURRENT_DIR, ".env")
env_path_base = os.path.join(BASE_DIR, ".env")

if os.path.exists(env_path_current):
    load_dotenv(env_path_current)
elif os.path.exists(env_path_base):
    load_dotenv(env_path_base)
else:
    load_dotenv()


# =========================
# 读取知识库
# 优先读取当前目录，其次读取程序目录
# =========================
def load_knowledge():
    possible_paths = [
        os.path.join(CURRENT_DIR, "knowledge_test.txt"),
        os.path.join(BASE_DIR, "knowledge_test.txt"),
    ]

    for path in possible_paths:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return f.read()

    console.print("[yellow]未找到 knowledge_test.txt，知识库内容将为空。[/yellow]")
    return ""


knowledge = load_knowledge()


# =========================
# 创建 AI 客户端
# =========================
client = OpenAI(
    api_key=os.getenv("API_KEY"),
    base_url="https://api.deepseek.com"
)


# =========================
# AI 人设 / System Prompt
# =========================
SYSTEM_PROMPT = f"""
你是一名大学学习助手。

========================
【重要知识库】
========================

你必须优先依据下面知识库回答问题。

如果知识库中存在相关内容：

1. 必须优先使用知识库内容
2. 不允许忽略知识库
3. 回答时应体现知识库观点
4. 可以在知识库基础上补充解释

知识库内容：

{knowledge}

========================
【回答要求】
========================

1. 解答大学问题
2. 通俗易懂
3. 像老师一样讲解
4. 多举例
5. 使用 Markdown 格式
6. 如果有代码，请使用 Markdown 代码块
7. 如果有数学公式，请使用 LaTeX，例如 $E=mc^2$ 或 $$a^2+b^2=c^2$$

========================
【代码文件输出要求】
========================

当用户明确需要完整的 .py、.cpp、.md 文件时：

1. 必须给出完整文件内容
2. 必须使用代码块
3. 不要省略 import、main 函数、类定义、头文件等必要部分
4. 不要写“其余部分不变”“这里省略”“同上”
5. 代码必须能直接复制粘贴使用
6. 如果是 Python 文件，代码块语言使用 ```python
7. 如果是 C++ 文件，代码块语言使用 ```cpp
8. 如果是 Markdown 文件，代码块语言使用 ```markdown
9. 如果用户指定了文件名，请在代码块开始处写成：
   ```python filename=main.py
   ```cpp filename=main.cpp
   ```markdown filename=README.md
"""


# =========================
# 聊天记录文件
# =========================
CHAT_FILE = os.path.join(CURRENT_DIR, "chat_history.json")


# =========================
# 每次启动重新创建 system prompt
# =========================
messages = [
    {
        "role": "system",
        "content": SYSTEM_PROMPT
    }
]


# =========================
# 读取历史聊天记录
# 只读取 user / assistant
# =========================
if os.path.exists(CHAT_FILE):
    try:
        with open(CHAT_FILE, "r", encoding="utf-8") as f:
            history_messages = json.load(f)
            messages.extend(history_messages)

        console.print("[green]历史聊天记录读取成功[/green]")

    except Exception as e:
        console.print(f"[red]聊天记录读取失败：{e}[/red]")


# =========================
# 保存历史记录
# =========================
def save_chat_history():
    save_messages = messages[1:]

    with open(CHAT_FILE, "w", encoding="utf-8") as f:
        json.dump(
            save_messages,
            f,
            ensure_ascii=False,
            indent=2
        )


# =========================
# 多行输入：Enter 发送，Shift+Enter 换行
# =========================
def multiline_input():
    """
    目标：
    1. Enter 发送
    2. Shift+Enter 换行
    3. Ctrl+J 作为备用换行
    4. Alt+Enter 作为备用换行

    注意：
    有些终端无法识别 Shift+Enter。
    如果你的终端不支持，就用 Ctrl+J 或 Alt+Enter 换行。
    """

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

    session = PromptSession(
        multiline=True,
        key_bindings=kb
    )

    text = session.prompt(
        "Sean: ",
        bottom_toolbar="Enter 发送｜Shift+Enter 换行｜Ctrl+J/Alt+Enter 备用换行｜输入 exit 退出"
    )

    return text


# =========================
# 空格中止监听
# =========================
class SpaceStopper:
    """
    在 AI 思考 / 接收流式内容时监听空格。
    用户按空格后，把 stop_event 设为 True。

    Windows 使用 msvcrt。
    macOS / Linux 使用 termios + tty + select。
    """

    def __init__(self):
        self.stop_event = threading.Event()
        self.thread = None
        self.running = False
        self.old_terminal_settings = None

    def start(self):
        self.running = True
        self.stop_event.clear()

        self.thread = threading.Thread(
            target=self._listen,
            daemon=True
        )

        self.thread.start()

    def stop(self):
        self.running = False

        if self.thread is not None:
            self.thread.join(timeout=0.2)

        self._restore_terminal()

    def stopped(self):
        return self.stop_event.is_set()

    def _listen(self):
        if os.name == "nt":
            self._listen_windows()
        else:
            self._listen_unix()

    def _listen_windows(self):
        import msvcrt

        while self.running:
            if msvcrt.kbhit():
                key = msvcrt.getwch()

                if key == " ":
                    self.stop_event.set()
                    self.running = False
                    break

            time.sleep(0.03)

    def _listen_unix(self):
        import select
        import termios
        import tty

        try:
            self.old_terminal_settings = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())

            while self.running:
                readable, _, _ = select.select([sys.stdin], [], [], 0.03)

                if readable:
                    key = sys.stdin.read(1)

                    if key == " ":
                        self.stop_event.set()
                        self.running = False
                        break

        except Exception:
            pass

        finally:
            self._restore_terminal()

    def _restore_terminal(self):
        if os.name != "nt" and self.old_terminal_settings is not None:
            try:
                import termios

                termios.tcsetattr(
                    sys.stdin,
                    termios.TCSADRAIN,
                    self.old_terminal_settings
                )

                self.old_terminal_settings = None

            except Exception:
                pass


# =========================
# Markdown + LaTeX 渲染
# 只输出格式化后的内容
# =========================
def render_formatted_reply(text: str):
    """
    只显示格式化后的内容：
    1. 普通 Markdown 用 Rich Markdown 渲染
    2. $$...$$ 公式用 Panel 单独展示
    """

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
                    border_style="magenta"
                )
            )


# =========================
# 从用户输入中提取文件名
# =========================
def extract_filenames_from_user_input(user_input: str):
    """
    从用户输入里提取用户明确提到的文件名。
    支持：
    main.py
    main.cpp
    README.md
    """
    pattern = r"([A-Za-z0-9_\-\u4e00-\u9fff]+?\.(?:py|cpp|md))"
    names = re.findall(pattern, user_input, flags=re.I)

    result = []
    for name in names:
        if name not in result:
            result.append(name)

    return result


# =========================
# 清理文件名
# =========================
def safe_filename(filename: str):
    """
    防止模型输出危险路径。
    只允许保存到当前目录，不允许 ../ 或绝对路径。
    """
    filename = filename.strip().strip('"').strip("'")
    filename = filename.replace("\\", "/")
    filename = filename.split("/")[-1]

    filename = re.sub(r'[<>:"/\\|?*]', "_", filename)

    if not filename:
        filename = "generated.md"

    return filename


# =========================
# 从代码块信息中提取文件名
# =========================
def extract_filename_from_code_info(info: str):
    """
    支持这些写法：
    ```python filename=main.py
    ```python main.py
    ```cpp filename="main.cpp"
    ```markdown README.md
    """
    info = info.strip()

    filename_match = re.search(
        r'filename\s*=\s*["\']?([^"\'\s]+)["\']?',
        info,
        flags=re.I
    )

    if filename_match:
        return safe_filename(filename_match.group(1))

    direct_match = re.search(
        r"([A-Za-z0-9_\-\u4e00-\u9fff]+?\.(?:py|cpp|md))",
        info,
        flags=re.I
    )

    if direct_match:
        return safe_filename(direct_match.group(1))

    return None


# =========================
# 根据代码语言决定扩展名
# =========================
def language_to_extension(language: str):
    language = language.lower().strip()

    if language in ["python", "py"]:
        return ".py"

    if language in ["cpp", "c++", "cplusplus"]:
        return ".cpp"

    if language in ["markdown", "md"]:
        return ".md"

    return None


# =========================
# 提取并保存 AI 回复中的 py/cpp/md 代码块
# =========================
def save_code_blocks(reply: str, user_input: str):
    """
    自动保存 AI 回复中的完整代码块到当前目录。
    只保存 python / py / cpp / c++ / markdown / md。
    """

    code_block_pattern = re.compile(
        r"```([^\n`]*)\n(.*?)```",
        flags=re.S
    )

    matches = list(code_block_pattern.finditer(reply))

    if not matches:
        return []

    requested_names = extract_filenames_from_user_input(user_input)
    requested_index = 0

    saved_files = []
    counters = {
        ".py": 1,
        ".cpp": 1,
        ".md": 1
    }

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

        save_path = os.path.join(CURRENT_DIR, filename)

        with open(save_path, "w", encoding="utf-8", newline="\n") as f:
            f.write(code.rstrip() + "\n")

        saved_files.append(save_path)

    return saved_files


# =========================
# 真正流式接收函数
# 但不输出原始 token
# 完成后只输出格式化内容
# =========================
def stream_ai_reply(messages, user_input):
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
        stream = client.chat.completions.create(
            model="deepseek-v4-pro",
            messages=messages,
            stream=True
        )

        for chunk in stream:
            if stopper.stopped():
                cancelled = True

                try:
                    stream.close()
                except Exception:
                    pass

                break

            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta

            if delta.content is None:
                continue

            full_reply += delta.content

    except KeyboardInterrupt:
        cancelled = True

    finally:
        stopper.stop()

        if stream is not None:
            try:
                stream.close()
            except Exception:
                pass

    end_time = time.time()

    console.print()
    console.print(f"[dim]思考耗时: {end_time - start_time:.2f} 秒[/dim]")

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


# =========================
# 启动提示
# =========================
console.print("[bold blue]=== My Assistant ===[/bold blue]")
console.print("[dim]输入 exit 退出[/dim]")
console.print("[dim]需要换行：Shift+Enter；如果不生效，用 Ctrl+J 或 Alt+Enter。[/dim]")
console.print("[dim]AI 思考时按空格可以中止。[/dim]")
console.print()


# =========================
# 主循环
# =========================
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

    messages.append({
        "role": "user",
        "content": user_input
    })

    try:
        reply, cancelled, saved_files = stream_ai_reply(messages, user_input)

        if cancelled:
            messages.pop()
            continue

        messages.append({
            "role": "assistant",
            "content": reply
        })

        save_chat_history()

    except Exception as e:
        console.print(f"\n[bold red]出错了：{e}[/bold red]")

        if messages and messages[-1]["role"] == "user":
            messages.pop()