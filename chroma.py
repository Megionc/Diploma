from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma
from typing import List
import os
import shutil

os.environ['OPENAI_API_KEY'] = 'dummy_key'

CHROMA_PATH = "chroma"
# DATA_PATH = "data/fincert/txt"
DATA_PATH = "data/big_set/txt"
CHUNK_SIZE = 750
CHUNK_OVERLAP = 200

def get_embeddings():
   model_kwargs = {'device': 'cuda'}
   embeddings_hf = HuggingFaceEmbeddings(
       model_name='intfloat/multilingual-e5-large',
       model_kwargs=model_kwargs
   )
   return embeddings_hf
   
   
def load_documents():
   loader = DirectoryLoader(DATA_PATH, glob="*.txt")
   documents = loader.load()
   return documents
   
   
def split_text(documents: list[Document]):
   text_splitter = RecursiveCharacterTextSplitter(
       chunk_size=CHUNK_SIZE,
       chunk_overlap=CHUNK_OVERLAP,
       length_function=len,
       add_start_index=True,
   )
   chunks = text_splitter.split_documents(documents)
   print(f"Разбили {len(documents)} документов на {len(chunks)} чанков.")

   return chunks
   
def save_to_chroma(chunks: list[Document]):
    # Clear out the database first.
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    # Create a new DB from the documents.
    db = Chroma.from_documents(
        chunks, get_embeddings(), 
        persist_directory=CHROMA_PATH
    )
    db.persist()
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")

   
   
def main():
   documents = load_documents()
   chunks = split_text(documents)
   save_to_chroma(chunks)
   
   
if __name__ == "__main__":
    main()

