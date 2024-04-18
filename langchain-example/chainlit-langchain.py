import chainlit as cl
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.llms import CTransformers
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate


class StreamHandler(BaseCallbackHandler):
    def __init__(self):
        self.msg = cl.Message(content="")

    async def on_llm_new_token(self, token: str, **kwargs):
        await self.msg.stream_token(token)

    async def on_llm_end(self, response: str, **kwargs):
        await self.msg.send()
        self.msg = cl.Message(content="")


# Load quantized Llama2
llm = CTransformers(
    model="TheBloke/Llama-2-7B-Chat-GGUF",
    model_file="llama-2-7b-chat.Q2_K.gguf",
    model_type="llama2",
    config={
        "max_new_tokens": 2000,
        "temperature": 0.01,
        "context_length": 2000,
        "threads": 8,
        "gpu_layers": 1,
    },
)

template = """
[INST] <<SYS>>
You are a helpful, respectful and honest assistant. Always provide a concise answer and use the Context provided in the question. If you don't know the answer, you can say so.
<</SYS>>
Context:
{context}
User:
{question}[/INST]"""


prompt = PromptTemplate(template=template, input_variables=["context", "question"])
memory = ConversationBufferMemory(memory_key="context")


@cl.on_chat_start
def on_chat_start():
    llm_chain = LLMChain(prompt=prompt, llm=llm, verbose=False, memory=memory)
    cl.user_session.set("llm_chain", llm_chain)


@cl.on_message
async def on_message(message: cl.Message):
    llm_chain = cl.user_session.get("llm_chain")

    await llm_chain.acall(
        message.content, callbacks=[cl.AsyncLangchainCallbackHandler(), StreamHandler()]
    )
