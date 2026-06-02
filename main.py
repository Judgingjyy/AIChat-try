from openai import OpenAI
from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
import os
import json

# 读取.env
load_dotenv()
# 读取知识库
with open("knowledge_test.txt", "r", encoding="utf-8") as f:
    knowledge = f.read()
# 创建AI客户端
client = OpenAI(
    api_key=os.getenv("API_KEY"),
    base_url="https://open.bigmodel.cn/api/paas/v4/"
)

# AI人设
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
5. 使用Markdown格式
"""

# 聊天记录文件
CHAT_FILE = "chat_history.json"

# 如果聊天记录存在
# 尝试读取聊天记录
try:

    if os.path.exists(CHAT_FILE):

        with open(CHAT_FILE, "r", encoding="utf-8") as f:
            messages = json.load(f)

    else:

        raise Exception("没有聊天记录")

# 如果读取失败
except:

    messages = [
        {
            "role":"system",
            "content":SYSTEM_PROMPT
        }
    ]

print("=== My Assistant ===")
print("Enter exit To exit")
print()

while True:

    user_input = input("Sean: ")

    if user_input.lower() == "exit":
        break

    # 保存用户消息
    messages.append({
        "role":"user",
        "content":user_input
    })

    # 调用AI
    response = client.chat.completions.create(
        model="glm-4-flash",
        messages=messages
    )

    reply = response.choices[0].message.content

    print("\nAI:")
    console=Console()
    md=Markdown(reply)
    console.print(md)
    print()

    # 保存AI回复
    messages.append({
        "role":"assistant",
        "content":reply
    })

    # 写入文件
    with open(CHAT_FILE, "w", encoding="utf-8") as f:
        json.dump(messages, f, ensure_ascii=False, indent=2)