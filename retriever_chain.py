from langchain.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document

import os
os.environ["OPENAI_API_KEY"] = "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"

def build_chain():
    with open("data/live_feed.txt", "r") as f:
        logs = f.read()

    docs = [Document(page_content=logs)]
    splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=20)
    chunks = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings()
    vectordb = FAISS.from_documents(chunks, embeddings)

    llm = ChatOpenAI(model="gpt-4-turbo")
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=vectordb.as_retriever())

    return qa