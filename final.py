import chainlit as cl
from ctransformers import AutoModelForCausalLM

llm = AutoModelForCausalLM.from_pretrained(
    "zoltanctoth/orca_mini_3B-GGUF", model_file="orca-mini-3b.q4_0.gguf"
)

SYSTEM = """
You are an AI assistant that follows instruction extremely well. Help as much as you can. Give short answers. Use the context provided and don't mention that you use the context."
"""


def get_prompt(instruction, conversation_history=[]):
    context = ". ".join(conversation_history)
    prompt = f"### System:\n{SYSTEM}. Context: {context}\n\n### User:\n{instruction}\n\n### Response:\n"
    print(f"\n----------------------------------\nPROMPT: {prompt}")
    return prompt


history = []


@cl.on_message
async def on_message(message: cl.Message):
    prompt = get_prompt(message.content, history)
    history.append(message.content)

    msg = cl.Message(content="")
    await msg.send()

    for token in llm(prompt, stream=True):
        await msg.stream_token(token)
    await msg.update()
