from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma
from langchain.prompts.chat import ChatPromptTemplate
# from langchain_community.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI

# Example: reuse your existing OpenAI setup
from openai import OpenAI

from typing import List
import chroma
import argparse
import os
import shutil

os.environ['OPENAI_API_KEY'] = 'dummy_key'

CHROMA_PATH = "chroma"

# PROMPT_TEMPLATE = """
# Ответь на вопрос базируясь только на этом контексте:

# {context}

# ---

# Ответь на вопрос, используя только контекст: {question}
# """


PROMPT_TEMPLATE = """
Используй следующие фрагменты контекста, чтобы в конце ответить на вопрос.
Если ты не нашел ответа, просто скажи, что не знаешь ответа. Не пытайся выдумывать ответ.
Используй максимум три предложения и старайся отвечать максимально кратко.
{context}
Вопрос: {question}
Полезный ответ: 
"""

def main():
    # Читаем аргументы запуска
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str)
    args = parser.parse_args()
    query_text = args.query_text

    # Создаем БД
    db = Chroma(persist_directory=chroma.CHROMA_PATH,
                embedding_function=chroma.get_embeddings())

    # Ищем по БД
    # Мы будем использовать 3 чанка из БД, которые наиболее похожи на наш вопрос
    # c этим количеством можете экспериментировать как угодно, главное, не переборщите, ваша LLM
    # должна поддерживать такое количество контекста, чтобы уместить весь полученный промпт
    results = db.similarity_search_with_relevance_scores(query_text, k=5)
    if len(results) == 0 or results[0][1] < 0.7:
        print(f"Нет фрагментов текста, на которые можно опираться для ответа.")
        return
    
    print(results[0])

    # Собираем запрос к LLM, объединяя наши чанки. Их мы записываем через пропуск строки и ---
    # помещаем мы контекст в переменную context, которую обозначали еще в самом промпте
    # ну и по аналогии вопрос просто записываем в переменную question.
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    print(f"Полученный промпт {prompt}")
   
    # Подключение к LM Studio и отправка запроса
    model = ChatOpenAI(temperature=0.7, base_url="http://localhost:1234/v1", api_key="not-needed")
    response_text = model.invoke(prompt)

    # Выводим результаты ответа
    sources = [doc.metadata.get("source", None) for doc, _score in results]
    formatted_response = f"Ответ: {response_text}\nДанные взяты из: {sources}"
    print(formatted_response)
   
   
if __name__ == "__main__":
    main()

