
from dotenv import load_dotenv

import pandas as pd
import streamlit as st


import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from chatbot import SmartAgent, PERSONA, AVAILABLE_FUNCTIONS, FUNCTIONS_SPEC

import os
from pathlib import Path  
agent = SmartAgent(persona=PERSONA,functions_list=AVAILABLE_FUNCTIONS, functions_spec=FUNCTIONS_SPEC, init_message="Hi there, this is Mỹ Hiệp, what can I do for you?")

st.set_page_config(layout="wide",page_title="Enterprise Copilot- A demo of Copilot application using GPT")
styl = f"""
<style>
    .stTextInput {{
      position: fixed;
      bottom: 3rem;
    }}
</style>
"""
st.markdown(styl, unsafe_allow_html=True)


MAX_HIST= 3
# Sidebar contents
with st.sidebar:
    st.title(' AI Copilot')
    st.markdown('''
    This is a demo of RAG AI Assistant.
    ''')
    add_vertical_space(5)
    st.write('Copyright SCG company 2024.')
    if st.button('Clear Chat'):

        if 'history' in st.session_state:
            st.session_state['history'] = []

    if 'history' not in st.session_state:
        st.session_state['history'] = []
    if 'input' not in st.session_state:
        st.session_state['input'] = ""
    if 'question_count' not in st.session_state:
        st.session_state['question_count'] = 0


user_input= st.chat_input("You:")

## Conditional display of AI generated responses as a function of user provided prompts
history = st.session_state['history']
question_count=st.session_state['question_count']

if len(history) > 0:
    idx=0
    removal_indices =[]
    running_question_count=0
    start_counting=False # flag to start including history items in the removal_indices list
    running_question_count=0
    start_counting=False # flag to start including history items in the removal_indices list
    for message in history:
        idx += 1
        message = dict(message)
        print("role: ", message.get("role"), "name: ", message.get("name"))
        if message.get("role") == "user":
            running_question_count +=1
            start_counting=True
        if start_counting and (question_count- running_question_count>= MAX_HIST):
            removal_indices.append(idx-1)
        elif question_count- running_question_count< MAX_HIST:
            break
            
    # remove items with indices in removal_indices
    # print("removal_indices", removal_indices)
    for index in removal_indices:
        del history[index]
    question_count=0
    # print("done purging history, len history now", len(history ))

    for message in history:
        idx += 1
        message = dict(message)
        if message.get("role") != "system" and message.get("role") != "tool" and message.get("name") is None and message.get("content") and len(message["content"]) > 0:
            with st.chat_message(message["role"]):
                st.write(message["content"])

else:
    query_used,history, agent_response,display_pictures = agent.run(user_input=None)
    with st.chat_message("assistant"):
        st.write(agent_response)
    user_history=[]
if user_input:
    with st.chat_message("user"):
        st.write(user_input)
    query_used, history, agent_response, display_pictures = agent.run(user_input=user_input, conversation=history)
    st.session_state['question_count'] += 1
    if display_pictures:  # Nếu display_pictures là True
        with st.chat_message("assistant"):
            chart_dir = "exports/charts"
            if os.path.exists(chart_dir):
                chart_files = [f for f in os.listdir(chart_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
                if chart_files:
                    first_chart = os.path.join(chart_dir, chart_files[0])
                    st.image(first_chart, caption="Chart from exports/charts")
                else:
                    st.write("No charts found in exports/charts.")
            else:
                st.write("Directory exports/charts does not exist.")
    else:
        with st.chat_message("assistant"):
            st.write(agent_response)

st.session_state['history'] = history