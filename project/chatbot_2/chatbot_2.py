import openai
import os
import re
import chromadb
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts.chat import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain

openai.api_key = os.environ['API_KEY']
os.environ["TOKENIZERS_PARALLELISM"] = "false"


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


def generate_vector_db(dataset):
    # TODO 한글이라 잘 안되는 것일까?
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=50)
    client = chromadb.PersistentClient()
    collection = client.get_or_create_collection(
        name="kakao_sync_bot")
    ids = []
    documents = []

    for key in dataset:
        k = key.lower().replace(' ', '-')
        texts = text_splitter.split_text(dataset[key])
        for text_index in range(len(texts)):
            ids.append(f"{k}_{text_index}")
            documents.append(f"{texts[text_index].strip().lower()}")

    collection.add(
        documents=documents,
        ids=ids,
    )

    print("DB 생성 완료")


def read_prompt_template(file_path: str) -> str:
    with open(file_path, "r") as f:
        prompt_template = f.read()

    return prompt_template


def call_db(param):
    client = chromadb.PersistentClient()
    collection = client.get_or_create_collection(
        name="kakao_sync_bot")
    searchResult = collection.query(
        query_texts=[param],
        n_results=3)

    return searchResult['documents']


def request_gpt_api(prompt: str, gpt_model="gpt-3.5-turbo", max_token: int = 500, temperature=0.1) -> str:
    response = openai.ChatCompletion.create(
        model=gpt_model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_token,
        temperature=temperature,
    )
    return response.choices[0].message.content


def generate_sync_bot(param):
    # ----------------- langchain 방식 -----------------##
    lim = ChatOpenAI(temperature=0.8, max_tokens=300, model="gpt-3.5-turbo")

    context_chain = create_chain(lim, '../template/kakao_sync_prompt_1.txt', 'context_output')
    extra_info_chain = create_chain(lim, '../template/kakao_sync_prompt_2.txt', 'extra_output')

    preprocess_chain = SequentialChain(
        chains=[
            context_chain,
            extra_info_chain,
        ],
        input_variables=["kakao_sync_data", "command", "extra_link"],
        output_variables=["context_output", "extra_output"],
        verbose=True,
    )

    context = dict(
        kakao_sync_data=call_db(param),
        command=param,
        extra_link=request_gpt_api("카카오 싱크 대표 링크 알려줘"),
    )
    context = preprocess_chain(context)
    return context["extra_output"]


# ----------------- prompt 방식 (이전) -----------------##
# prompt_template = read_prompt_template('../template/bot_prompt_1.txt')
# prompt = prompt_template.format(
#     kakao_sync_data=call_db(param),
#     command=param,
# )
# return request_gpt_api(prompt)


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
    print("프로젝트 2단계")
    data = load_data("../dataset/project_data_kakao_sync.txt")
    dataset = generate_format_data(data)
    generate_vector_db(dataset)
    result = generate_sync_bot("기능은 뭐야?")
    print(result)


if __name__ == "__main__":
    main()
