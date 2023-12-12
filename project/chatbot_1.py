import json
import openai
import os
import re
import tkinter as tk
from tkinter import scrolledtext
import chromadb

openai.api_key = os.environ['API_KEY']

def prompt():
    message_log = [
        {
            "role": "system",
            "content": '''
            카카오챗봇으로 카카오챗봇에 대한 궁금한 점을 질문하는 봇입니다. 
            예를 들어, "카카오챗봇은 무엇인가요?" 또는 "카카오챗봇은 어떻게 사용하나요?"와 같은 질문을 할 수 있습니다.
            질문은 "quit"를 입력할 때까지 계속할 수 있습니다.
            # function_call 호출 조건 : 질문이나 요구사항은 command 로 function_call 로 전달됩니다.
            '''
        }
    ]
    functions = [
        {
            "name": "kakao_chatbot",
            "description": "kakaotalk chatbot information",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "질문을 하면 답변을 해줄 수 있습니다.",
                    },
                },
                "required": ["command"],
            },
        }
    ]
    return message_log, functions

def loadData(filepath):
    # txt 파일을 읽어서 데이터를 생성한다.
    file = open(filepath, "r", encoding="utf-8")
    data = file.read()
    file.close()

    return data


def generateFormatData(data):
    # 테이블, json 형태가 좋음
    splitData = re.split(r'#([ㄱ-ㅣ가-힣\s]+\n)', data.strip())
    dataset = {}

    subTitle = '제목' # default 제목
    for i in range(len(splitData)):
        data = splitData[i]
        if "\n\n" not in data:
            subTitle = data.strip()
        else:
            dataset[subTitle] = data.strip()
    return dataset


def kakao_chatbot(command):
    client = chromadb.PersistentClient()
    collection = client.get_or_create_collection(
        name="kakao_chatbot")
    searchResult = collection.query(
        query_texts=[command],
        n_results=3)

    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": f"{searchResult['documents']}"},
        ],
        max_tokens=1024
    )

    return completion["choices"][0]["message"]["content"]


def show_popup_message(window, message):
    popup = tk.Toplevel(window)
    popup.title("")

    # 팝업 창의 내용
    label = tk.Label(popup, text=message, font=("맑은 고딕", 12))
    label.pack(expand=True, fill=tk.BOTH)

    # 팝업 창의 크기 조절하기
    window.update_idletasks()
    popup_width = label.winfo_reqwidth() + 20
    popup_height = label.winfo_reqheight() + 20
    popup.geometry(f"{popup_width}x{popup_height}")

    # 팝업 창의 중앙에 위치하기
    window_x = window.winfo_x()
    window_y = window.winfo_y()
    window_width = window.winfo_width()
    window_height = window.winfo_height()

    popup_x = window_x + window_width // 2 - popup_width // 2
    popup_y = window_y + window_height // 2 - popup_height // 2
    popup.geometry(f"+{popup_x}+{popup_y}")

    popup.transient(window)
    popup.attributes('-topmost', True)

    popup.update()
    return popup


def send_message_to_gpt(message_log, functions, gpt_model="gpt-3.5-turbo", temperature=0.1):
    response = openai.ChatCompletion.create(
        model=gpt_model,
        messages=message_log,
        temperature=temperature,
        functions=functions,
        function_call='auto',
    )

    response_message = response["choices"][0]["message"]

    if response_message.get("function_call"):
        available_functions = {
            "kakao_chatbot": kakao_chatbot,
        }
        function_name = response_message["function_call"]["name"]
        fuction_to_call = available_functions[function_name]
        function_args = json.loads(response_message["function_call"]["arguments"])
        # 사용하는 함수에 따라 사용하는 인자의 개수와 내용이 달라질 수 있으므로
        # **function_args로 처리하기
        function_response = fuction_to_call(**function_args)

        # 함수를 실행한 결과를 GPT에게 보내 답을 받아오기 위한 부분
        message_log.append(response_message)  # GPT의 지난 답변을 message_logs에 추가하기
        message_log.append(
            {
                "role": "function",
                "name": function_name,
                "content": function_response,
            }
        )  # 함수 실행 결과도 GPT messages에 추가하기
        response = openai.ChatCompletion.create(
            model=gpt_model,
            messages=message_log,
            temperature=temperature,
        )  # 함수 실행 결과를 GPT에 보내 새로운 답변 받아오기
    return response.choices[0].message.content


def on_send(message_log, user_entry, window, conversation, functions):
    user_input = user_entry.get()
    user_entry.delete(0, tk.END)

    if user_input.lower() == "quit":
        window.destroy()
        return

    message_log.append({"role": "user", "content": user_input})
    conversation.config(state=tk.NORMAL)  # 이동
    conversation.insert(tk.END, f"You: {user_input}\n", "user")  # 이동
    thinking_popup = show_popup_message(window, "처리중...")
    window.update_idletasks()
    # '생각 중...' 팝업 창이 반드시 화면에 나타나도록 강제로 설정하기
    response = send_message_to_gpt(message_log, functions)
    thinking_popup.destroy()

    message_log.append({"role": "assistant", "content": response})

    # 태그를 추가한 부분(1)
    conversation.insert(tk.END, f"gpt assistant: {response}\n", "assistant")
    conversation.config(state=tk.DISABLED)
    # conversation을 수정하지 못하게 설정하기
    conversation.see(tk.END)

def chatbot_window():
    messageLog, functions = prompt()

    window = tk.Tk()
    window.title("GPT AI")
    font = ("맑은 고딕", 10)
    conversation = scrolledtext.ScrolledText(window, wrap=tk.WORD, bg='#f0f0f0', font=font)
    # width, height를 없애고 배경색 지정하기(2)
    conversation.tag_configure("user", background="#c9daf8")
    # 태그별로 다르게 배경색 지정하기(3)
    conversation.tag_configure("assistant", background="#e4e4e4")
    # 태그별로 다르게 배경색 지정하기(3)
    conversation.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    # 창의 폭에 맞추어 크기 조정하기(4)
    input_frame = tk.Frame(window)  # user_entry와 send_button을 담는 frame(5)
    input_frame.pack(fill=tk.X, padx=10, pady=10)  # 창의 크기에 맞추어 조절하기(5)
    user_entry = tk.Entry(input_frame)
    user_entry.pack(fill=tk.X, side=tk.LEFT, expand=True)
    send_button = tk.Button(input_frame, text="Send",
                            command=on_send(messageLog, user_entry, window, conversation, functions))
    send_button.pack(side=tk.RIGHT)
    window.bind('<Return>',
                lambda event: on_send(messageLog, user_entry, window, conversation, functions))
    window.mainloop()


def generateVectorDB(dataset):
    # 한글이라 잘 안되는 것일까?
    client = chromadb.PersistentClient()
    collection = client.get_or_create_collection(
        name="kakao_chatbot")
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

def main():
    print("프로젝트 1단계 - 호출하기")
    data = loadData("./dataset/project_data_카카오톡채널.txt")
    dataset = generateFormatData(data)
    generateVectorDB(dataset)
    chatbot_window()


if __name__ == "__main__":
    main()
