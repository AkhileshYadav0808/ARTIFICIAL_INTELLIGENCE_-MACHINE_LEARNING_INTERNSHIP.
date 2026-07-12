import streamlit as st
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import UnstructuredHTMLLoader
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

# --- Streamlit App Configuration ---
st.set_page_config(page_title="Washing Machine Manual Chatbot", layout="centered")
st.title("🤖 Washing Machine Manual Chatbot")
st.markdown("Welcome! This chatbot uses **Retrieval Augmented Generation (RAG)** to answer questions based on your Samsung washing machine manual. ")
st.markdown("Just type your question below, and I'll do my best to provide a concise answer from the manual.")

# --- Sidebar for API Key input ---
st.sidebar.header("⚙️ Configuration")
openai_api_key = st.sidebar.text_input("Enter your OpenAI API Key", type="password")

# --- Instructions Expander ---
with st.expander("💡 How it works and what you need"):
    st.write("1. **Enter your OpenAI API Key:** This key is essential for accessing OpenAI's models (GPT-4o-mini for responses and text-embedding-3-small for embeddings). Your key is not stored.")
    st.write("2. **Manual File:** Ensure the `How to use the various modes of the washing machine _ Samsung LEVANT.html` file is in the same directory as this `app.py` script.")
    st.write("3. **Ask a Question:** Once configured, type your question about the washing machine in the text box below.")
    st.info("**Note:** If you don't have an OpenAI API key, you can get one from [platform.openai.com](https://platform.openai.com/api-keys).")

# --- Main Application Logic ---
if not openai_api_key:
    st.warning("⚠️ Please enter your OpenAI API key in the sidebar to activate the chatbot.")
else:
    os.environ["OPENAI_API_KEY"] = openai_api_key

    # Load the HTML document
    try:
        # The HTML file should be present in the same directory as app.py
        manual_path = "How to use the various modes of the washing machine _ Samsung LEVANT.html"
        if not os.path.exists(manual_path):
            st.error(f"🚫 Error: The manual file '{manual_path}' was not found. Please ensure it's in the same directory as app.py.")
            st.stop()

        loader = UnstructuredHTMLLoader(manual_path)
        machine_docs = loader.load()
    except Exception as e:
        st.error(f"❌ Failed to load HTML document: {e}. Check file path and permissions.")
        st.stop()

    # Initialize LLM and Embeddings
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=os.environ["OPENAI_API_KEY"])

    # Initialize RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    # Split the machine documents
    splits = text_splitter.split_documents(machine_docs)

    # Initialize Chroma vectorstore
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)

    # Setup vectorstore as retriever
    retriever = vectorstore.as_retriever()

    # Define RAG prompt
    prompt = ChatPromptTemplate.from_template("""You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know.
Use three sentences maximum and keep the answer concise.\nQuestion: {question} \nContext: {context} \nAnswer:""")

    # Setup the RAG chain
    # The RunnablePassthrough allows the original question to be passed through to the prompt
    # along with the retrieved context.
    rag_chain = (
        {"context": retriever , "question": RunnablePassthrough()}
        | prompt
        | llm
    )

    st.subheader("❓ Ask your question:")
    # User input for query
    user_query = st.text_input("Type your question here...")

    if user_query:
        with st.spinner("Thinking..."):
            # Invoke the RAG chain and get the answer
            # We can also get the context if we structure the chain differently, but for now just the answer.
            response = rag_chain.invoke(user_query)
            answer = response.content

            st.success("✅ Here is the answer:")
            st.write(answer)
            
            # Optionally, display the retrieved context for debugging/transparency
            # This requires modifying the chain to return context as well, or running retriever separately.
            # For simplicity, we're just showing the final answer for now.

    st.markdown("--- Felt this was helpful? --- Give it a 🌟 on GitHub!")
