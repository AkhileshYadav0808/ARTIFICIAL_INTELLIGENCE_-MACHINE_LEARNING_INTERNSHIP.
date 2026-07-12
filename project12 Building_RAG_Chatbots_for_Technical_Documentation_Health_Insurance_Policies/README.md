# Project 12: Building RAG Chatbots for Technical Documentation & Health Insurance Policies

## Overview

This project focuses on building a **Retrieval-Augmented Generation (RAG) chatbot** capable of answering user queries from large collections of technical documents and health insurance policy documents.

The chatbot combines:
- **Document retrieval** to find relevant information from a knowledge base.
- **Large Language Models (LLMs)** to generate accurate and context-aware responses.
- **Vector databases** for efficient semantic search.

The system is designed to reduce manual document searching and provide users with quick, reliable answers from complex doc

            User Query
                |
                v
      +-------------------+
      |   Chat Interface  |
      +-------------------+
                |
                v
      +-------------------+
      | Query Processing  |
      +-------------------+
                |
                v
      +-------------------+
      | Vector Database   |
      | Semantic Search   |
      +-------------------+
                |
                v
      +-------------------+
      | Retrieved Context |
      +-------------------+
                |
                v
      +-------------------+
      |      LLM          |
      | Response Generator|
      +-------------------+
                |
                v
         Final Answer

         

streamlit_link=https://5xl6kzmdsscsmrjruc7glb.streamlit.app/
