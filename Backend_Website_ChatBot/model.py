import os
from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

urls = [
    'https://askadvi.org/',
    'https://askadvi.org/students/',
    'https://askadvi.org/counselors/',
    'https://askadvi.org/faq/'
]

from langchain.document_loaders import UnstructuredURLLoader
loaders = UnstructuredURLLoader(urls=urls)
data = loaders.load()

# Text Splitter
from langchain.text_splitter import CharacterTextSplitter

text_splitter = CharacterTextSplitter(separator='\n',
                                      chunk_size=1000,
                                      chunk_overlap=300)

docs = text_splitter.split_documents(data)

import pickle
import faiss
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()

import tiktoken
vectorStore_openAI = FAISS.from_documents(docs, embeddings)

with open("faiss_store_openai.pkl", "wb") as f:
  pickle.dump(vectorStore_openAI, f)

with open("faiss_store_openai.pkl", "rb") as f:
    VectorStore = pickle.load(f)

from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.question_answering import load_qa_chain
from langchain import OpenAI

llm=OpenAI(temperature=0)

chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=VectorStore.as_retriever())

output = chain({"question": "what is student resources?"}, return_only_outputs=True)
print(output.get('answer'))
print("DONE")
#-----------------------------------------------------------
# import os
# import pickle
#
# import faiss
# from langchain.chains import RetrievalQAWithSourcesChain, llm
# from langchain.document_loaders import UnstructuredURLLoader
# from langchain.vectorstores import VectorStore
#
# urls = [
#     'https://askadvi.org/',
#     'https://askadvi.org/students/',
#     'https://askadvi.org/counselors/',
#     'https://askadvi.org/faq/'
# ]
#
# loaders = UnstructuredURLLoader(urls=urls)
# data = loaders.load()
#
# # Text Splitter
# from langchain.text_splitter import CharacterTextSplitter
#
# text_splitter = CharacterTextSplitter(separator='\n',
#                                       chunk_size=1000,
#                                       chunk_overlap=300)
#
# docs = text_splitter.split_documents(data)
#
# import azure.storage.blob as blob
#
# container_name = "my-container"
# blob_name = "faiss_store_openai.pkl"
#
# with blob.ContainerClient(account_url="my-account-url", container_name="my-container") as container_client:
#     with container_client.get_blob_client(blob_name) as blob_client:
#         vectorStore_openAI = pickle.loads(blob_client.download_blob().content)
#
# chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=VectorStore.as_retriever())
#
# output = chain({"question": "what is student resources?"}, return_only_outputs=True)
# print(output.get('answer'))
# print("DONE")
