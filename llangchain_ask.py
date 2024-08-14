from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

# Загружаем документ
loader = TextLoader('data/big_ti/Text/1_T.txt', encoding='utf-8')
docs = loader.load()

# Разбиваем документ на части
text_splitter = RecursiveCharacterTextSplitter(chunk_size=3300, chunk_overlap=500)
splits = text_splitter.split_documents(docs)

# Переводим части документа в эмбединги и помещаем в хранилище
embedding = HuggingFaceEmbeddings(
	model_name='sentence-transformers/paraphrase-multilingual-mpnet-base-v2',
	model_kwargs={'device':'cuda:0'},
	encode_kwargs={'normalize_embeddings': False}
)
vectorstore = Chroma.from_documents(documents=splits, 
                                    embedding=embedding,
                                    collection_metadata={"hnsw:space": "cosine"})

retriever = vectorstore.as_retriever(search_type="similarity_score_threshold", 
                                     search_kwargs={"score_threshold": 0.7}, 
                                     k=3)

# Работаем с локальной моделью
llm = ChatOpenAI(temperature=0.7, base_url="http://localhost:1234/v1", api_key="not-needed")

# Формируем шаблон запроса к llm
template = """Используй только следующие фрагменты контекста, чтобы в конце ответить на вопрос.
Если ты не нашел ответа, просто скажи, что не знаешь ответа. Не пытайся выдумывать ответ.
Старайся отвечать максимально кратко и только на русском языке. Используй не больше 5 предложений.
{context}
Вопрос: {question}
Полезный ответ: """

prompt = PromptTemplate.from_template(template)

def format_docs(docs):
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


# Формируем конвеер
rag_chain = (
	{"context": retriever | format_docs, "question": RunnablePassthrough()}
	| prompt
	| llm
	| StrOutputParser()
)

# Задаем вопрос по документу
out = rag_chain.invoke("Какие были тренды кибератак в России в прошлом году?")

print(out)

