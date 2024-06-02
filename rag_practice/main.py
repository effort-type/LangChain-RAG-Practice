def docs_load():
    """
    문서 읽는 함수
    """

    from langchain_community.document_loaders import TextLoader

    loader = TextLoader("corpus/정시 모집요강(동의대) 전처리 결과.txt", encoding="utf-8").load()

    print(loader)

    return loader


def c_text_split(corpus):
    """
    CharacterTextSplitter를 사용하여 문서를 분할하도록 하는 함수
    :param corpus: 전처리 완료된 말뭉치
    :return: 분리된 청크
    """

    from langchain.text_splitter import CharacterTextSplitter

    c_text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="---",
        chunk_size=1500,
        chunk_overlap=0,
    )

    text_documents = c_text_splitter.split_documents(corpus)

    return text_documents


def embedding_model():
    """
    문서 임베딩 모델을 생성하는 함수
    :return: 허깅페이스 임베딩 모델
    """

    from langchain.embeddings import HuggingFaceEmbeddings

    model_name = "jhgan/ko-sroberta-multitask"  # 한국어 모델
    model_kwargs = {'device': 'cpu'}  # cpu를 사용하기 위해 설정
    encode_kwargs = {'normalize_embeddings': True}
    model = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

    return model


def document_embedding(docs, model, save_directory):
    """
    Embedding 모델을 사용하여 문서 임베딩하여 Chroma 벡터저장소(VectorStore)에 저장하는 함수
    :param model: 임베딩 모델 종류
    :param save_directory: 벡터저장소 저장 경로
    :param docs: 분할된 문서
    :return:
    """

    from langchain_community.vectorstores import Chroma
    import os
    import shutil

    print("\n잠시만 기다려주세요.\n\n")

    # 벡터저장소가 이미 존재하는지 확인
    if os.path.exists(save_directory):
        shutil.rmtree(save_directory)
        print(f"디렉토리 {save_directory}가 삭제되었습니다.\n")

    print("문서 벡터화를 시작합니다. ")
    db = Chroma.from_documents(docs, model, persist_directory=save_directory)
    print("새로운 Chroma 데이터베이스가 생성되었습니다.\n")

    return db


def chat_llm():
    """
    채팅에 사용되는 거대언어모델 생성 함수
    :return: 답변해주는 거대언어모델
    """

    import os
    from dotenv import load_dotenv
    from langchain.chat_models import ChatOpenAI
    from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

    load_dotenv('.env')

    llm = ChatOpenAI(
        # base_url=os.getenv("LM_URL"),
        base_url=os.getenv("LM_LOCAL_URL"),
        api_key="lm-studio",
        model="teddylee777/EEVE-Korean-Instruct-10.8B-v1.0-gguf",
        temperature=0,
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandler()]
    )

    return llm


def qna(llm, db):
    qna_result = []

    check = 'Y'  # 0이면 질문 가능

    while check == 'Y' or check == 'y':
        query = input("질문을 입력하세요 : ")
        response = db_qna(llm, db, query)  # 기본 검색기

        qna_result.append({'query': query, 'response': response})

        check = input("\n\nY: 계속 질문한다.\nN: 프로그램 종료\n입력: ")

    return qna_result


def db_qna(llm, db, query, ):
    """
    벡터저장소에서 질문을 검색해서 적절한 답변을 찾아서 답하도록 하는 함수
    :param llm: 거대 언어 모델
    :param db: 벡터스토어
    :param query: 사용자 질문
    :return: 거대언어모델(LLM) 응답 결과
    """

    from langchain.prompts import ChatPromptTemplate
    from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
    from langchain_core.output_parsers import StrOutputParser

    db = db.as_retriever(
        # search_kwargs={'k': 3},
        search_type="mmr",
        search_kwargs={'k': 3, 'fetch_k': 10},
        # search_type='similarity_score_threshold',
        # search_kwargs={'k': 3, 'score_threshold': 0.45},
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                You are a specialized AI for question-and-answer tasks.
                You must answer questions based solely on the Context provided.
                If no Context is provided, you must instruct to inquire at "https://ipsi.deu.ac.kr/main.do".

                Context: {context}
                """,
            ),
            ("human", "Question: {question}"),
        ]
    )

    chain = {
                "context": db | RunnableLambda(format_docs),
                "question": RunnablePassthrough()
            } | prompt | llm | StrOutputParser()

    response = chain.invoke(query)

    return response


def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)


def run():
    """
    챗봇 시작
    Document Load -> Text Splitter -> Ducument Embedding -> VectorStore save -> QA
    """

    # 문서 업로드
    loader = docs_load()

    # 문서 분할
    chunk = c_text_split(loader)

    print(chunk)

    # 임베딩 모델 생성
    model = embedding_model()

    # 문서 임베딩
    db = document_embedding(chunk, model, save_directory="./chroma_db")

    # 채팅에 사용할 거대언어모델(LLM) 선택
    llm = chat_llm()

    # 질의응답
    qna_list = qna(llm, db)

    print(qna_list)


if __name__ == "__main__":
    run()
