from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

document_text = """
Retrieval-Augmented Generation (RAG) grounds LLM responses
using retrieved external context.
Chunking is necessary due to context length limits.
Good chunking improves retrieval quality.
Poor chunking causes missing or noisy answers.
"""

splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50,
)

docs = splitter.create_documents([document_text])

embeddings = HuggingFaceEmbeddings(
    model_name = "all-MiniLM-L6-v2"
)

vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    collection_name="rag_example"
)

retriever = vectorstore.as_retriever(search_kwargs = {"k":2})

prompt = ChatPromptTemplate.from_template("""
Answer the question using ONLY the context below.
If the answer is not present, say "Not found in the context."

Context:
{context}

Question:
{question}
                                          """)

llm = Ollama(model = "llama3")

rag_chain = (
    {
        "context": retriever,
        "question": lambda x: x
    }
    | prompt
    | llm
    | StrOutputParser()
)

query = "Why is chunking important in RAG systems?"
result = rag_chain.invoke(query)

print("\nAnswer:")
print(result)



