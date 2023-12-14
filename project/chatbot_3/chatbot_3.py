import openai
import os
import re
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts.chat import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.tools import Tool
from langchain.utilities import GoogleSearchAPIWrapper

openai.api_key = os.environ['API_KEY']
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["GOOGLE_API_KEY"]
os.environ["GOOGLE_CSE_ID"]


def load_data(filepath):
    # txt 파일을 읽어서 데이터를 생성한다.
    file = open(filepath, "r", encoding="utf-8")
    data = file.read()
    file.close()

    return data


def generate_format_data(data):
    # 테이블, json 형태가 좋음
    splitData = re.split(r'#([ㄱ-ㅣ가-힣\s]+\n)', data.strip())
    dataset = {}

    subTitle = '제목'  # default 제목
    for i in range(len(splitData)):
        data = splitData[i]
        if "\n\n" not in data:
            subTitle = data.strip()
        else:
            dataset[subTitle] = data.strip()
    return dataset


def generate_vector_db():
    dir_path = "../dataset/"
    datasets = {}
    persist_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma")
    embedding = OpenAIEmbeddings()
    _db = None

    for file in os.listdir(dir_path):
        if file.endswith(".txt"):
            category = file.split(".txt")[0].split("_")[3]
            data = load_data(os.path.join(dir_path, file))
            datasets[category] = generate_format_data(data)

    for category in datasets:
        print(f"DB 생성 중... {category}")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0)

        for key in datasets[category]:
            pages = text_splitter.split_text(datasets[category][key])
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=10)
            texts = text_splitter.create_documents(pages)

        _db = Chroma.from_documents(texts, embedding, collection_name=category,
                                    persist_directory=persist_directory)
    _db.persist()
    print("DB 생성 완료")


def read_prompt_template(file_path: str) -> str:
    with open(file_path, "r") as f:
        prompt_template = f.read()
    return prompt_template


def read_intent_list(file_path: str) -> dict:
    data = ""
    with open(file_path, "r") as f:
        data = f.read()

    intent_dict = {}
    for line in data.strip().split("\n"):
        kv = line.strip().split(":")
        if len(kv) != 2:
            continue
        intent_dict[kv[0]] = kv[1]
    return intent_dict


def call_db(intent, param):
    DATA_DIR = os.path.dirname(os.path.abspath('./chroma'))
    CHROMA_PERSIST_DIR = os.path.join(DATA_DIR, "chroma")
    CHROMA_COLLECTION_NAME = intent
    _db = Chroma(
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=OpenAIEmbeddings(),
        collection_name=CHROMA_COLLECTION_NAME,
    )

    docs = _db.similarity_search(param)
    str_docs = [doc.page_content for doc in docs]
    return str_docs


def request_gpt_api(prompt: str, gpt_model="gpt-3.5-turbo", max_token: int = 500, temperature=0.1) -> str:
    response = openai.ChatCompletion.create(
        model=gpt_model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_token,
        temperature=temperature,
    )
    return response.choices[0].message.content


def query_web_search(user_message: str) -> str:
    llm = ChatOpenAI(temperature=0.1, max_tokens=200, model="gpt-3.5-turbo")
    SEARCH_VALUE_CHECK_PROMPT_TEMPLATE = os.path.join("../template/search_value_check.txt")
    SEARCH_COMPRESSION_PROMPT_TEMPLATE = os.path.join("../template/search_compress.txt")

    search = GoogleSearchAPIWrapper(
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        google_cse_id=os.getenv("GOOGLE_CSE_ID")
    )

    search_tool = Tool(
        name="Google Search",
        description="Search Google for recent results.",
        func=search.run,
    )

    context = {"user_message": user_message}
    context["related_web_search_results"] = search_tool.run(user_message)

    search_value_check_chain = create_chain(llm, SEARCH_VALUE_CHECK_PROMPT_TEMPLATE, "output")
    search_compression_chain = create_chain(llm, SEARCH_COMPRESSION_PROMPT_TEMPLATE, "output")

    has_value = search_value_check_chain.run(context)

    print(has_value)
    if has_value == "Y":
        return search_compression_chain.run(context)
    else:
        return ""


def generate_answer(param):
    INTENT_PROMPT_TEMPLATE = os.path.join("../template/parse_intent.txt")
    BOT_PROMPT_TEMPLATE = os.path.join("../template/bot_prompt_1.txt")
    DEFAULT_PROMPT_TEMPLATE = os.path.join("../template/bot_prompt_1.txt")
    INTENT_LIST_TXT = os.path.join("../template/intent_list.txt")

    # ----------------- langchain 방식 -----------------##
    llm = ChatOpenAI(temperature=0.1, max_tokens=300, model="gpt-3.5-turbo")

    # TODO
    # 인텐트 생성
    # memory 구성
    # web search 붙여보기(default)

    # 인텐트 맞는 템플릿으로 연결되게 함
    parse_intent_chain = create_chain(llm, INTENT_PROMPT_TEMPLATE, "intent")
    bot_prompt_chain = create_chain(llm, BOT_PROMPT_TEMPLATE, 'bot_output')
    # extra_info_chain = create_chain(llm, '../template/kakao_sync_prompt_2.txt', 'extra_output')
    default_chain = create_chain(llm, DEFAULT_PROMPT_TEMPLATE, "output")
    context = dict(user_message=param)
    context["intent_list"] = read_prompt_template(INTENT_LIST_TXT)
    intent = parse_intent_chain.run(context)
    if intent != 'None' and ":" in intent:
        intent = intent.split(":")[0]
    intent_list = read_intent_list(INTENT_LIST_TXT)
    context["intent_desc"] = intent_list[intent]
    # return intent

    if intent != 'None':
        context["related_documents"] = call_db(intent, context["user_message"])
        answer = ""
        for step in [bot_prompt_chain]:
            context = step(context)
            answer += context[step.output_key]
            answer += "\n\n"
    else:
        context["related_documents"] = request_gpt_api(context["user_message"])
        context["compressed_web_search_results"] = query_web_search(context["user_message"])
        answer = default_chain.run(context)
    return answer


def create_chain(llm, template_path, output_key):
    return LLMChain(
        llm=llm,
        prompt=ChatPromptTemplate.from_template(
            template=read_prompt_template(template_path),
        ),
        output_key=output_key,
        verbose=True,
    )


def main():
    print("프로젝트 3단계")
    # generate_vector_db() // DB 생성할때만 사용하는 함수 (한번만 실행)
    # call_db("social","기능은 뭐야?")
    result = generate_answer("카카오싱크 기능이 무엇이 있는지 설명해주세요")
    # result = generate_answer("카카오톡 채널 고객 관리에 대해 알려주세요")
    print(result)


if __name__ == "__main__":
    main()
