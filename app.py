import os, dotenv, base64
import streamlit as st
import asyncio
import edge_tts
from streamlit_mic_recorder import speech_to_text
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.vectorstores import FAISS
from langchain_cohere.embeddings import CohereEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import tempfile

# Load environment variables
dotenv.load_dotenv()

os.environ['GROQ_API_KEY'] = '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'
os.environ['GOOGLE_API_KEY'] = "AIzaSyBYyvQTqe9qdQTWKPp4hyjyFYKkjfNTero"    
os.environ['COHERE_API_KEY'] = "6zXEeF9WcwbBmRFExg0b5JSc8ZgH77o1LwuOf1hs"


# Available voices for Text-to-Speech
voices = {
    "William":"en-AU-WilliamNeural",
    "James":"en-PH-JamesNeural",
    "Jenny":"en-US-JennyNeural",
    "US Guy":"en-US-GuyNeural",
    "Sawara":"hi-IN-SwaraNeural",
}

st.set_page_config(page_title="Practice Planner Chatbot", layout="wide", page_icon="./images/pencil.png")

# Title
st.markdown("""
    <h1 style='text-align: center;'>
        <span style='color: #fcfcfc;'>Practice</span> 
        <span style='color: #CEAB5C;'>Planner</span>
        <span style='color: #fcfcfc;'>bot</span>
    </h1>
""", unsafe_allow_html=True)


with st.sidebar:
    st.markdown("## Practice Planner Bot")
    st.write("This bot can help you to generate different Practice Plans for your Athelites.")
    st.divider()

# Load vectorstore only once
if "vectorstore" not in st.session_state:
    embeddings = CohereEmbeddings(model="embed-multilingual-v3.0")
    st.session_state["vectorstore"] = FAISS.load_local("faiss_db", embeddings, allow_dangerous_deserialization=True)

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state['chat_history'] = [
        {"role":"assistant", "content":"Hey there! How can I assist you today?"}
    ]

def format_docs(docs):
    return "\n\n".join(
        [f'Document {i+1}:\n{doc.page_content}\n'
         f'Source: {doc.metadata.get("source", "Unknown")}\n'
         f'Category: {doc.metadata.get("category", "Unknown")}\n'
         f'Instructor: {doc.metadata.get("instructor", "N/A")}\n-------------'
         for i, doc in enumerate(docs)]
    )

# Reset conversation
def reset_conversation():
    st.session_state.pop('chat_history')
    st.session_state['chat_history'] = [
        {"role":"assistant", "content":"Hey there! How can I assist you today about sillkup academy?"}
    ]

def rag_qa_chain(question, retriever, chat_history):

    llm = ChatGroq(model="llama-3.1-70b-versatile")
    output_parser = StrOutputParser()

    contextualize_q_system_prompt = """Given a chat history and the latest user question which might reference context in the chat history,
    formulate a standalone question which can be understood without the chat history. If the original question is in any other language then englist,
    then ask the user polightly that please give me question or query in english. Do NOT answer the question if it is in enlgish, just reformulate and ask the user it if needed and otherwise return it as is."""

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{question}"),
            ]
        )
    contextualize_q_chain = contextualize_q_prompt | llm | output_parser


    qa_system_prompt = (
    """You are a helpful and insightful coach designed to create **customized practice plans** for coaches by utilizing the markdown files of drills and the provided `Practice-Plan-Template.md`.

### Behavior:
- **Purpose**: Assist users in generating practice plans based on their input, such as the number of athletes, practice intensity, and type of session (e.g., off-season, pre-season).
- **Content Source**: Use the Markdown files categorized as follows:
  - Blocked Skills Training
  - CNS + Visual Drills
  - Competitive Game-Like
  - Competitive Simulations
  - Frameworks
  - Strength and Conditioning
  - Practice Plan Templates
- **Formatting**: Ensure the generated practice plan matches the formatting of `Practice-Plan-Template.md`. And if there are any discriptions about any drill please provide it when you will give the details about that drill. And if you don't have then its ok you don't have to mention it. Please don't provide any wrong description about anything. Don't give any Day time and year.



### Guidelines:

1. When the user requests a practice plan:
   - Select drills from the appropriate Markdown files. Create continuity throughout practice by suggesting situational variations of the game-like drills based on the focuses of the blocked training drills. 
   - Follow the structure provided in the `Practice-Plan-Template.md`, adding details on set up and variations, when available.
   - Ensure that the intensity and drill selection align with the practice type (e.g., low-intensity for pre-tournament)

2. When the user asks irrelevant questions:
   - Respond politely, redirecting the conversation back to practice planning. Avoid profanity or inappropriate language.

3. If information is unavailable:
   - Politely apologize and inform the user that the requested information is not available.

### Input Example:
"Make me a 6-person, off-season practice plan"  

### Expected Behavior:
- Use the files of drills and the template to generate the requested practice plan.
- Maintain continuity between skill development and gameplay drills to ensure a progressive practice flow.
- Provide output in the structure and tone of the `Practice-Plan-Template.md`.

Retrieved Documents (Context):
------------
{context}
------------

"""
)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ])

    final_llm = ChatGoogleGenerativeAI(model='gemini-1.5-pro', temperature=0.5)

    rag_chain = (
        RunnablePassthrough.assign(
            context=contextualize_q_chain | retriever | format_docs
        )
        | prompt
        | final_llm
        | output_parser
    )

    return rag_chain.stream({"question": question, "chat_history": chat_history})



# Generate the speech from text
async def generate_speech(text, voice):
    communicate = edge_tts.Communicate(text, voice)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
        await communicate.save(temp_file.name)
        temp_file_path = temp_file.name
    return temp_file_path

# Get audio player
def get_audio_player(file_path):
    with open(file_path, "rb") as f:
        data = f.read()
        b64 = base64.b64encode(data).decode()
        return f'<audio autoplay="true" src="data:audio/mp3;base64,{b64}">'


#  Text-to-Speech function which automatically plays the audio
def generate_voice(text, voice):
    text_to_speak = (text).translate(str.maketrans('', '', '#-*_😊👋😄😁🥳👍🤩😂😎')) # Removing special chars and emojis
    with st.spinner("Generating voice response..."):
        temp_file_path = asyncio.run(generate_speech(text_to_speak, voice)) 
        audio_player_html = get_audio_player(temp_file_path)  # Create an audio player
        st.markdown(audio_player_html, unsafe_allow_html=True)
        os.unlink(temp_file_path)
    
if st.sidebar.toggle("Enable Voice Response"):
    voice_option = st.sidebar.selectbox("Choose a voice for response:", options=list(voices.keys()), key="voice_response")

# Dividing the main interface into two parts
col1, col2 = st.columns([1, 5])

# Displaying chat history
for message in st.session_state.chat_history:
    avatar = "./images/athelete.png" if message["role"] == "user" else "./images/coach.png"
    with col2:
        st.chat_message(message["role"], avatar=avatar).write(message["content"])

# Handle voice or text input
with col1:
    st.button("Reset", use_container_width=True, on_click=reset_conversation)

    with st.spinner("Converting speech to text..."):
        text = speech_to_text(language="eng", just_once=True, key="STT", use_container_width=True)


query = st.chat_input("Type your question")

# Generate the response
if text or query:
    col2.chat_message("user", avatar="./images/athelete.png").write(text if text else query)
    
    st.session_state.chat_history.append({"role": "user", "content": text if text else query})

    # Generate response
    with col2.chat_message("assistant", avatar="./images/coach.png"):
        response = st.write_stream(rag_qa_chain(question=text if text else query,
                            retriever=st.session_state["vectorstore"].as_retriever(search_kwargs={"k": 6}),
                            chat_history=st.session_state.chat_history))
    
        # Add response to chat history
        st.session_state.chat_history.append({"role": "assistant", "content": response})

    # Generate voice response if the user has enabled it
    if "voice_response" in st.session_state and st.session_state.voice_response:
        response_voice = st.session_state.voice_response
        generate_voice(response, voices[response_voice])
