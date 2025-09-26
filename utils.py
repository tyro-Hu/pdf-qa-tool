from langchain_deepseek import ChatDeepSeek
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain


def qa_agent(deepseek_api_key, memory, uploaded_file, question):
    # 定义模型
    model = ChatDeepSeek(model="deepseek-chat",api_key=deepseek_api_key)

    # 读取上传的文件，并写入临时文件
    file_content = uploaded_file.read()
    temp_file_path = "tenp.pdf"
    with open(temp_file_path, "wb") as temp_file:
        temp_file.write(file_content)

    # 加载文档
    loader = PyPDFLoader(temp_file_path)
    docs = loader.load()

    # 分割文档
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=50,
        separators=["\n\n", "\n", "。", "!", "？", "，", ""]
    )
    texts = text_splitter.split_documents(docs)

    # embedding模型
    embeddings_model = HuggingFaceBgeEmbeddings(
        model_name="BAAI/bge-m3"
    )

    # 向量数据库,检索器
    db = FAISS.from_documents(texts, embeddings_model)
    retriever = db.as_retriever()

    # 对话检索链
    qa = ConversationalRetrievalChain.from_llm(
        llm = model,
        retriever = retriever,
        memory = memory
    )

    # 问答
    response = qa.invoke({"chat_history":memory, "question": question})

    return response


