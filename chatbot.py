
from datetime import datetime
import inspect
import json
import os
import re
import time
from guardrails import Guard, OnFailAction
from guardrails_grhub_llamaguard_7b import LlamaGuard7B
from guardrails_grhub_toxic_language import ToxicLanguage
import numpy as np
from openai import OpenAI
import pandas as pd
from pandasai import SmartDataframe
from pydantic import ValidationError
from qdrant_client import QdrantClient
from qdrant_client.http import models
from dotenv import load_dotenv
from pandasai_openai import OpenAI as pandasOpenai
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
import logging
logging.basicConfig(filename='chatbot.log', encoding='utf-8', level=logging.INFO)

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)
os.environ["OPENAI_API_KEY"] = api_key

qdrant_client = QdrantClient("localhost", port=6333)
df = pd.read_csv("data/countries-of-the-world.csv")
# llm = pandasOpenai(api_token=api_key)
# agent = SmartDataframe(df,config={"llm":llm})
agent = create_pandas_dataframe_agent(
    ChatOpenAI(temperature=0, model="gpt-4o-mini"),
    df,
    verbose=True,
    agent_type=AgentType.OPENAI_FUNCTIONS,
    allow_dangerous_code=True
)
guard = Guard()
guard.name = "ChatBotGuard"
guard.use(ToxicLanguage())





def search_agent_pandasAI(query):
    try:
        # Tinh chỉnh truy vấn để yêu cầu chỉ trả về số
        # refined_query = f"{query}"
        answer = agent.invoke(query)
        print(answer)
        return answer
    except Exception as e:
        return str(e)

def create_collection():
    print("Creating collection 'chatbot_data'...")
    if qdrant_client.collection_exists("chatbot_world_data"):
        print("Collection 'chatbot_world_data' already exists!")
    else:
        qdrant_client.create_collection(
        collection_name="chatbot_world_data",
        vectors_config=models.VectorParams(
            size=1536,
            distance=models.Distance.COSINE
        )
        
    )
        print("Collection 'chatbot_world_data' created!")
        add_data_to_qdrant()

def get_embedding(text):
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding


def add_data_to_qdrant():
    texts = []
    try:
        with open("data/country_detailed_specialties.txt", "r", encoding="utf-8") as file: # Thêm encoding utf-8
            for line in file:
                texts.append(line.strip())  # Thêm từng dòng vào texts, loại bỏ khoảng trắng thừa
    except UnicodeDecodeError as e:
        print(f"Error decoding file: {e}")
        return

    vectors = [get_embedding(text) for text in texts]
    ids = list(range(len(texts)))

    if vectors:
        print("Adding data to Qdrant...")
        qdrant_client.upsert(
            collection_name="chatbot_world_data",
            points=models.Batch(
                ids=ids,
                vectors=vectors,
                payloads=[{"text": text} for text in texts]
            )
        )
        print("Data added to Qdrant!")
    else:
        print("No vectors generated, Qdrant upsert skipped.")

def search_vector_database(query):
    
    query_vector = get_embedding(query)
    search_result = qdrant_client.search(
        collection_name="chatbot_world_data",
        query_vector=query_vector,
        limit=3  
    )
    summorize = summarize_results(search_result,query)
    return summorize


def summarize_results(results,query):

    combined_text = "\n".join([result.payload["text"] for result in results])
   
    prompt = (
        f"Câu hỏi của người dùng: '{query}'. "
        f"Nếu không có thông tin phù hợp trong dữ liệu sau, trả lời 'Không tìm thấy thông tin phù hợp'. "
        f"Nếu có, tóm tắt ngắn gọn và đúng trọng tâm:\n{combined_text}"
    )
 
    response = client.chat.completions.create(
        model="gpt-4o-mini", 
        messages=[
            {"role": "system", "content": "Bạn là một trợ lý tóm tắt thông tin."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.5,
        max_tokens=500,  
    )
    
    summary = response.choices[0].message.content
    return summary


# def chatbot(query):
#     results = search_qdrant(query)
    
#     summary = summarize_results(results,query)
#     return summary

def check_args(function, args):
    sig = inspect.signature(function)
    params = sig.parameters

    # Check if there are extra arguments
    for name in args:
        if name not in params:
            return False
    # Check if the required arguments are provided 
    for name, param in params.items():
        if param.default is param.empty and name not in args:
           
            return False

    return True

def draw_plot_pandasAI(query):
    try:
        answer = agent.invoke(query)

        return answer
    except Exception as e:
        return str(e)
    
def initialize_data():
    create_collection()
    
def get_current_date():
    return datetime.now().strftime("%Y-%m-%d")
# initialize_data()
PERSONA = f"""
You are a friendly and knowledgeable global AI guide. 
The current date/time is {get_current_date()}. 
Your mission is to assist users with questions about countries worldwide. 
Prioritize searching vector database (search_vector_database) for general inquiries about country specialties (culture, history, cuisine, geography). 
Use search_agent_pandasAI for numerical or statistical data questions, expecting a numeric result when appropriate.
For visualization requests, use draw_plot_pandasAI and ensure the query explicitly includes 'plot' or 'draw' (e.g., 'Plot population of the 5 largest countries') to generate a chart. 
Provide accurate answers based on the database. If an answer is unavailable and dont have information, say 'Sorry, I don’t have that information' and do not make up an answer. 
Respond in the user’s language, keeping answers brief and concise.
"""

AVAILABLE_FUNCTIONS = {
            "search_agent_pandasAI": search_agent_pandasAI,
            "search_vector_database": search_vector_database,
            "draw_plot_pandasAI": draw_plot_pandasAI
        } 

FUNCTIONS_SPEC = [
    {
        "type": "function",
        "function": {
            "name": "search_agent_pandasAI",
            "description": "A search tool for a pandas DataFrame with country data: Country, Region, Population, Area (sq. mi.), Pop. Density (per sq. mi.), Coastline, Net Migration, Infant Mortality (per 1000 births), GDP ($ per capita), Literacy (%), Phones (per 1000), Arable (%), Crops (%), Other (%), Climate, Birthrate, Deathrate, Agriculture, Industry, Service.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query for a country's data, the figure, the number, or the statistic"
                    }
                },
                "required": ["query"],
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_vector_database",
            "description": "A search tool for finding information about the specialties of a country, including its unique cultural, historical, culinary, and geographical features.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query for a country's information such like geography, position, famous with,etc."
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "draw_plot_pandasAI",
            "description": "A tool to draw plots from a pandas DataFrame with country data: Country, Region, Population, Area (sq. mi.), etc. The query must explicitly include 'plot' or 'draw' (e.g., 'Plot population of the 5 largest countries') to generate a visualization.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The query for drawing a plot, must include 'plot' or 'draw'"
                    }
                },
                "required": ["query"]
            }
        }
    }


]  

import json



class SmartAgent:
    def __init__(self, persona, functions_spec, functions_list, name=None, init_message=None):
        if init_message is not None:
            init_hist = [{"role": "system", "content": persona}, {"role": "assistant", "content": init_message}]
        else:
            init_hist = [{"role": "system", "content": persona}]
        self.init_history = init_hist
        self.persona = persona
        self.name = name
        self.functions_spec = functions_spec
        self.functions_list = functions_list

    def run(self, user_input, conversation=None):
        display_pictures = False
        if user_input is None:
            return False, self.init_history, self.init_history[1]["content"], display_pictures

        if conversation is None:
            conversation = self.init_history.copy()
        start_time = time.perf_counter()
      
        # user_input,error, pass_flag =  validate_guardrails(guard, user_input)
        # Kiểm tra nếu input không hợp lệ
        # if pass_flag == False:
        #     end_time = time.perf_counter()
        #     response_time = end_time - start_time  # Tính thời gian phản hồi
           
        #     logging.info(f"User Input: {user_input}")
        #     logging.info(f"Response Time: {response_time:.4f} seconds")
        #     logging.info(f"Response: {error}")
        #     return False, conversation, error, display_pictures
        
        

        # Nếu hợp lệ, tiếp tục xử lý
        conversation.append({"role": "user", "content": user_input})
        global guard
    
        while True:
            # Gọi API OpenAI
            
            try:
                check_input = guard.validate(user_input)
                print("Check input:", check_input)
            except Exception as e:
                end_time = time.perf_counter()
                response_time = end_time - start_time  # Tính thời gian phản hồi
            
                logging.info(f"User Input: {user_input}")
                logging.info(f"Response Time: {response_time:.4f} seconds")
                if isinstance(e, ValidationError):
                    return False, conversation, "I'm sorry, I can't answer that question.", display_pictures
                return False, conversation, "I'm sorry there was a problem, I can't answer that question.", display_pictures
                            
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=conversation,
                tools=self.functions_spec,
                tool_choice="auto",
                max_tokens=600,
            )
            # **Kết thúc đo thời gian phản hồi**
            end_time = time.perf_counter()
            response_time = end_time - start_time  # Tính thời gian phản hồi
            response_message = response.choices[0].message
            print("API Response:", response_message)
            # Ghi log thời gian phản hồi vào file
        
            # Kiểm tra xem có tool_calls không
            tool_calls = response_message.tool_calls
            if tool_calls:
                # Thêm tin nhắn assistant với tool_calls vào hội thoại
                assistant_message = {
                    "role": "assistant",
                    "content": response_message.content if response_message.content else None,
                    "tool_calls": [
                        {
                            "id": tool_call.id,
                            "function": {
                                "name": tool_call.function.name,
                                "arguments": tool_call.function.arguments
                            },
                            "type": "function"
                        } for tool_call in tool_calls
                    ]
                }
                conversation.append(assistant_message)
                print("Added assistant message with tool_calls:", assistant_message)

                # Xử lý từng tool call
                for tool_call in tool_calls:
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)

                    if function_name not in self.functions_list:
                        print(f"Hàm {function_name} không tồn tại")
                        function_response = f"Lỗi: Hàm {function_name} không tồn tại."
                    else:
                        function_to_call = self.functions_list[function_name]
                        print(f"Hàm {function_name} được gọi với args: {function_args}")
                        try:
                            function_response = function_to_call(**function_args)
                            print(f"Phản hồi từ hàm {function_name}: {function_response}")
                        except Exception as e:
                            function_response = f"Lỗi khi thực thi hàm {function_name}: {str(e)}"

                    # Thêm phản hồi công cụ với tool_call_id tương ứng
                    tool_message = {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": function_name,
                        "content": str(function_response),
                    }
                    conversation.append(tool_message)
                    print("Added tool message:", tool_message)

                    # Kiểm tra nếu công cụ liên quan đến vẽ biểu đồ
                    if function_name == "draw_plot_pandasAI":
                        display_pictures = True
                        break
                    if function_name == "search_vector_database":
                        break

                # Tiếp tục vòng lặp để mô hình xử lý phản hồi từ công cụ
                continue
            else:
                # Không có tool_calls, thêm phản hồi cuối cùng và kết thúc
                conversation.append({"role": "assistant", "content": response_message.content})
                break

        # Trả về nội dung phản hồi cuối cùng
        logging.info(f"User Input: {user_input}")
        logging.info(f"Response Time: {response_time:.4f} seconds")
        logging.info(f"Response: {response_message.content}")
        return False, conversation, response_message.content, display_pictures
    
        
                
    




