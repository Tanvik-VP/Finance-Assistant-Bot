import os
from langchain.chat_models import ChatOpenAI

from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
import chainlit as cl
import torch

DB_FAISS_PATH = 'vectorstores/db_faiss'

custom_prompt_template = """You are a sophisticated and diligent financial assistant, programmed to deliver accurate, respectful, and constructive responses. Your primary function is to provide information and insights related to finance and economics, with a focus on financial knowledge.

You are designed to adhere strictly to ethical guidelines, ensuring all your responses are free from harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. You maintain a socially unbiased stance and promote positivity in all interactions.

If you encounter a question that is unclear, nonsensical, or factually inconsistent, you are to clarify the confusion respectfully and guide the inquirer towards a coherent understanding, instead of providing incorrect or misleading information. In instances where you lack sufficient data or knowledge to respond accurately, you are to acknowledge the limitation openly, avoiding speculation or the dissemination of falsehoods.

Your ultimate aim is to educate, inform, and assist users in understanding financial concepts, and economic principles, empowering them with reliable information to make informed decisions.


Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['context', 'question'])
    return prompt

# loading model

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY") or "<YourOPENAIAPIKEY"

chat = ChatOpenAI(
    openai_api_key=os.environ["OPENAI_API_KEY"],
    model='gpt-4',max_tokens=512 ,temperature= 0.5)

#Retrieval QA Chain
def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                       chain_type='stuff',
                                       retriever=db.as_retriever(search_kwargs={'k': 2}),
                                       return_source_documents=True,
                                       chain_type_kwargs={'prompt': prompt}
                                       )
    return qa_chain


#QA Model Function
def qa_bot():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'device': device})
    db = FAISS.load_local(DB_FAISS_PATH, embeddings)
    llm = chat
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)

    return qa

#output function
def final_result(query):
    qa_result = qa_bot()
    response = qa_result({'query': query})
    return response

#chainlit code
@cl.on_chat_start
async def start():
    chain = qa_bot()
    msg = cl.Message(content="Starting the bot...")
    await msg.send()
    msg.content = "Hi, Welcome to Financial Assistant Bot. What is your query?"
    await msg.update()

    cl.user_session.set("chain", chain)

@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain") 
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    cb.answer_reached = True
    res = await chain.acall(message.content, callbacks=[cb])
    answer = res["result"]
    sources = res["source_documents"]

    if sources:
        answer += f"\nSources:" + str(sources)
    else:
        answer += "\nNo sources found"

    await cl.Message(content=answer).send()
