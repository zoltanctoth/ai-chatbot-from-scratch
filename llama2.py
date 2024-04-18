import multiprocessing
import sys

import chainlit as cl
from ctransformers import AutoModelForCausalLM

llm = AutoModelForCausalLM.from_pretrained(
    "TheBloke/Llama-2-7b-Chat-GGUF",
    model_file="llama-2-7b-chat.Q8_0.gguf",
    threads=multiprocessing.cpu_count(),
)

SYSTEM = """You are an AI assistant that follows instruction extremely well. Help as much as you can. Give short answers. Use the context provided."""


def get_prompt(instruction: str, conversation_history=[]) -> str:
    context = ". ".join(conversation_history)
    prompt = f"<s>[INST] <<SYS>>\n{SYSTEM}. Context: {context}\n<</SYS>>\n\n{instruction} [/INST]"
    print(f"Prompt created: {prompt}")
    return prompt


@cl.on_chat_start
def on_chat_start():
    cl.user_session.set("history", [])


@cl.on_message
async def on_message(message: cl.Message):
    history = cl.user_session.get("history")

    prompt = get_prompt(message.content, history)
    history.append(message.content)

    msg = cl.Message(content="")
    await msg.send()

    answer = ""
    for token in llm(prompt, stream=True):
        answer += token
        await msg.stream_token(token)
    await msg.update()
    history.append(answer)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        print(llm("Hi, how are you doing?"))
