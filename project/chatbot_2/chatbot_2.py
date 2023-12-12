import json
import openai
import os
import re
import chromadb

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
    # 한글이라 잘 안되는 것일까?
    client = chromadb.PersistentClient()
    collection = client.get_or_create_collection(
        name="kakao_sync_bot")
    ids = []
    documents = []
    for key in dataset:
        ids.append(key.lower().replace(' ', '-'))
        document = f"{key}:{dataset[key].strip().lower()}"
        documents.append(document)
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
    prompt_template = read_prompt_template('../template/kakao_sync_prompt_1.txt')
    prompt = prompt_template.format(
        kakao_sync_data=call_db(param),
        command=param,
    )
    return request_gpt_api(prompt)


def main():
    print("프로젝트 2단계")
    data = load_data("../dataset/project_data_카카오싱크.txt")
    dataset = generate_format_data(data)
    generate_vector_db(dataset)
    result = generate_sync_bot("카카오싱크 기능이 무엇이 있는지 설명해주세요")
    print(result)


if __name__ == "__main__":
    main()
