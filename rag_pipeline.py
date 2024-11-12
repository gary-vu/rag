from typing import List, Union, Generator, Iterator
import os
from pydantic import BaseModel

# LangChain specific imports
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

class Pipeline:

    class Valves(BaseModel):


    def __init__(self):
        self.documents = None
        self.index = None

    async def on_startup(self):
        # Initialize LangChain Components for document retrieval, embedding, and LLM interaction
        from langchain_community.document_loaders import WebBaseLoader
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from langchain_chroma import Chroma
        from langchain_ollama import OllamaEmbeddings

        # Load documents for LangChain
        self.documents = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/").load()

        # Text splitting
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
        splits = text_splitter.split_documents(self.documents)

        # Create embeddings and vector store
        embedding = OllamaEmbeddings(model=self.valves.LLAMAINDEX_EMBEDDING_MODEL_NAME)
        self.index = Chroma.from_documents(documents=splits, embedding=embedding, persist_directory="./sql_chroma_db")

    async def on_shutdown(self):
        # Handle shutdown if necessary
        pass

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        """
        This is where we integrate LangChain RAG processing into OpenWebUI's pipeline.
        """
        # Initialize LangChain retriever and query engine
        retriever = self.index.as_retriever(search_type="similarity_score_threshold", search_kwargs={"k": 3, "score_threshold": 0.5})

        # Prepare the prompt template
        prompt = PromptTemplate.from_template(
            """
            <s> [Instructions] You are a friendly assistant. Answer the question based only on the following context.
            If the context does not contain enough information to answer the question, politely reply with "I'm not sure" or "No Context". [/Instructions] </s>
            [Instructions] Question: {input}
            Context: {context}
            Answer: [/Instructions]
            """
        )

        # Initialize the model
        model = ChatOllama(model=self.valves.LLAMAINDEX_MODEL_NAME, base_url=self.valves.LLAMAINDEX_OLLAMA_BASE_URL)

        # Create the document chain using LangChain
        document_chain = create_stuff_documents_chain(model, prompt)
        chain = create_retrieval_chain(retriever, document_chain)

        # Process the user message through LangChain pipeline
        result = chain.invoke({"input": user_message})

        # Return the generated response and the context (relevant documents)
        return result["answer"], result.get("context", [])

