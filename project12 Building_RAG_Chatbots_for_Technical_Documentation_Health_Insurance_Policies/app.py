import streamlit as st
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import UnstructuredHTMLLoader
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

# --- Streamlit UI Configuration ---
st.set_page_config(page_title="Star Health Insurance RAG Chatbot", page_icon=":hospital:", layout="centered")

# Custom CSS for UI adjustments
st.markdown("""
<style>
.main { 
    background-color: #f0f2f6; 
    background-image: linear-gradient(to right, #eceff1, #e0e0e0);
}
[data-theme='dark'] .main {
    background-color: #0e1117;
    background-image: linear-gradient(to right, #1a1e26, #2c313a);
}
.stButton>button { 
    background-color: #4CAF50; 
    color: white;
    border-radius: 8px;
}
.stTextInput>div>div>input { 
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)

# --- Sidebar --- 
with st.sidebar:
    st.image("https://www.starhealth.in/sites/all/themes/starhealth/logo.png", width=150)
    st.title("Star Health RAG Chatbot")
    st.markdown("---")
    st.header("Project Information")
    st.write("This project demonstrates a Retrieval Augmented Generation (RAG) chatbot using LangChain and OpenAI models to answer questions about Star Health Insurance plans.")
    st.markdown("---")
    st.header("About the Model")
    st.write("The chatbot uses `gpt-4o-mini` as the LLM for generating responses and `text-embedding-3-small` for creating document embeddings. ChromaDB serves as the vector store.")
    st.markdown("---")
    st.header("Facilities Available")
    st.write("- Context-aware Q&A on health insurance documents\n- Rapid information retrieval\n- Interactive chat interface")
    st.markdown("---")
    st.write("Developed with ❤️")

# --- Main Content --- 
st.header("Ask me about Star Health Insurance Plans!")

# Load OpenAI API Key
openai_api_key = os.environ.get("OPENAI_API_KEY")
if not openai_api_key:
    st.error("OpenAI API key not found. Please set the 'OPENAI_API_KEY' environment variable or add it to Streamlit Secrets.")
    st.stop()

# --- Load and Process Document ---
@st.cache_resource
def setup_rag_system(api_key):
    # Dynamically look for starhealth.html in the current execution folder
    file_name = "starhealth.html"
    
    if not os.path.exists(file_name):
        # Fallback check for standard Colab file path if running in Colab environment
        file_name = "/content/starhealth.html"
        
    try:
        loader = UnstructuredHTMLLoader(file_path=file_name)
        machine_docs = loader.load()
    except FileNotFoundError:
        st.error(f"Error: 'starhealth.html' not found. Please ensure the file is in your application directory.")
        st.stop()
        
    # Initialize Text Splitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(machine_docs)

    # Define LLM, Embeddings, and Vectorstore
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=api_key)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=api_key)
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    retriever = vectorstore.as_retriever()

    # Define RAG prompt
    prompt = ChatPromptTemplate.from_template("""You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know.
Use three sentences maximum and keep the answer concise.\nQuestion: {question} \nContext: {context} \nAnswer:""")

    # Setup the RAG chain (Appended StrOutputParser to cleanly output string responses)
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain

rag_chain = setup_rag_system(openai_api_key)

# --- Chat Interface ---

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if user_prompt := st.chat_input("What would you like to know?"):
    # Display user message
    st.chat_message("user").markdown(user_prompt)
    st.session_state.messages.append({"role": "user", "content": user_prompt})

    with st.spinner("Thinking..."):
        # Invoke the RAG chain
        response = rag_chain.invoke(user_prompt)
        
        # Display assistant response
        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})