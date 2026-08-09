import os
import shutil
import streamlit as st
from dotenv import load_dotenv
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from rag_engine import get_retriever, add_pdf_to_knowledge_base, clear_database_safe

load_dotenv()

# --- Page Config ---
st.set_page_config(page_title="AI Document Assistant", page_icon="🤖")
st.title("🤖 AI Document Assistant")
st.markdown("Upload PDFs and ask questions about them.")

# --- Initialize LLM ---
if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.1)

prompt = ChatPromptTemplate.from_template(
    """Answer using the context. Mention the source filename.
    <context>{context}</context>
    Question: {input}"""
)

document_prompt = PromptTemplate(
    input_variables=["page_content", "source"],
    template="[Source: {source}] Content: {page_content}"
)

combine_docs_chain = create_stuff_documents_chain(llm, prompt, document_prompt=document_prompt)

# --- Sidebar: Upload ---
with st.sidebar:
    st.header("📁 Step 1: Upload")
    uploaded_files = st.file_uploader("Select PDFs", type="pdf", accept_multiple_files=True)
    if st.button("Build Knowledge Base"):
        if uploaded_files:
            clear_database_safe()
            with st.spinner("Processing files..."):
                for uploaded_file in uploaded_files:
                    save_path = os.path.join("data", uploaded_file.name)
                    with open(save_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    add_pdf_to_knowledge_base(save_path)
            st.success("✅ Knowledge Base Ready!")
        else:
            st.warning("⚠️ Please select files first.")

# --- Main Area: Chat ---
st.header("💬 Step 2: Ask the AI")
query = st.text_input("Your Question", placeholder="What is in the documents?")

if st.button("Ask AI"):
    if query:
        retriever = get_retriever()
        if retriever:
            with st.spinner("Thinking..."):
                try:
                    chain = create_retrieval_chain(retriever, combine_docs_chain)
                    response = chain.invoke({"input": query})
                    st.markdown("### AI Answer:")
                    st.write(response["answer"])
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        else:
            st.warning("⚠️ Build the database in the sidebar first!")
