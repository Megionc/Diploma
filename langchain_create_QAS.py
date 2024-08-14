from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

import pathlib
import collections


import pandas as pd

import tqdm

CHUNK_SIZE = 2500
CHUNK_OVERLAP = 500


# Формируем шаблон запроса к llm
template = """
Используй только следующие фрагменты контекста, чтобы в конце ответить на вопрос.
Если ты не нашел ответа, просто скажи, что не знаешь ответа. Не пытайся выдумывать ответ.
Старайся отвечать максимально кратко и только на русском языке. Используй не больше 5 предложений.
{context}
Вопрос: {question}
Полезный ответ: 
"""


def load_document(path_to_file):
    loader = TextLoader(str(path_to_file), encoding='utf-8')
    document = loader.load()
    return document


def split_text(document):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        add_start_index=True,
    )
    splits = text_splitter.split_documents(document)
    print(f"Разбили документ на {len(splits)} чанков.")

    return splits


def get_embeddings():
    embeddings = HuggingFaceEmbeddings(
        model_name='sentence-transformers/paraphrase-multilingual-mpnet-base-v2',
        # model_name='DeepPavlov/rubert-base-cased',
        model_kwargs={'device': 'cuda:0'},
        encode_kwargs={'normalize_embeddings': False},
    )
    return embeddings


def save_to_chroma(splits):
    embedding = get_embeddings()
    vectorstore = Chroma.from_documents(documents=splits,
                                        embedding=embedding,
                                        collection_metadata={
                                            "hnsw:space": "cosine"},
                                        )

    return vectorstore


def create_retriever(vectorstore):
    retriever = vectorstore.as_retriever(search_type="similarity_score_threshold",
                                         search_kwargs={
                                             "score_threshold": 0.7},
                                         k=3,
                                         )
    return retriever


def format_docs(docs):
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


def create_df_from_dir(path_to_parsed_txt_dir):
    files = collections.defaultdict(list)
    for path_to_file in path_to_parsed_txt_dir.iterdir():
        if path_to_file.is_file():
            try:
                with open(path_to_file, 'r', encoding='utf8') as file_open:
                    files['file_name'].append(path_to_file.name)
                    files['full_file_name'].append(path_to_file)
                    files['raw_text'].append(file_open.read())
            except:
                print(f'{path_to_file} not found')
    return pd.DataFrame(files)


# df = create_df_from_dir(pathlib.Path('.', 'data', 'big_ti', 'Text'))
# df_q = create_df_from_dir(pathlib.Path('.', 'data', 'big_ti', 'Questions'))
# df_a = create_df_from_dir(pathlib.Path('.', 'data', 'big_ti', 'Answers'))

df = create_df_from_dir(pathlib.Path('.', 'data', 'big_med', 'Text'))
df_q = create_df_from_dir(pathlib.Path('.', 'data', 'big_med', 'Questions'))
df_a = create_df_from_dir(pathlib.Path('.', 'data', 'big_med', 'Answers'))


def parse_q_num(file_name):
    qs = file_name.split('Q')[1]
    num = int(qs.split('.')[0])
    return num


def parse_a_num(file_name):
    qs = file_name.split('A')[1]
    num = int(qs.split('.')[0])
    return num


df['text_num'] = df['file_name'].apply(
    lambda file_name: int(file_name.split('_')[0]))

df_q['text_num'] = df_q['file_name'].apply(
    lambda file_name: int(file_name.split('_')[0]))
df_q['q_num'] = df_q['file_name'].apply(
    lambda file_name: parse_q_num(file_name))

df_a['text_num'] = df_a['file_name'].apply(
    lambda file_name: int(file_name.split('_')[0]))
df_a['q_num'] = df_a['file_name'].apply(
    lambda file_name: parse_a_num(file_name))


result = pd.merge(df,
                  df_q,
                  on='text_num',
                  suffixes=('_t', '_q'))

df_full = pd.merge(result,
                   df_a,
                   on=['text_num', 'q_num'],
                   suffixes=('_r', '_a'),
                   )

del df, df_a, df_q

def generate_answer(row):
    f_name = row['full_file_name_t']
    print(f'...process {f_name}')
    doc = load_document(row['full_file_name_t'])
    chunks = split_text(doc)
    vectorstore = save_to_chroma(chunks)
    retriever = create_retriever(vectorstore)

    prompt = PromptTemplate.from_template(template)

    # Работаем с локальной моделью
    llm = ChatOpenAI(
        temperature=0.7,
        base_url="http://localhost:1234/v1",
        api_key="not-needed")

    # Формируем конвеер
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # Задаем вопрос по документу
    return rag_chain.invoke(row['raw_text_q'])


df_full['RAG_answer'] = df_full.apply(generate_answer, axis=1)

df_full.to_excel('RAG.xlsx')

