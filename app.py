import os
import streamlit as st
from langchain_core.messages import HumanMessage
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from deepagents import create_deep_agent

# ---------------------------------------------------------
# 1. Streamlit Setup & Konfiguration
# ---------------------------------------------------------
st.set_page_config(page_title="Deep Agent RAG MVP", page_icon="🤖")
st.title("🤖 Deep Agent + RAG MVP")
st.write(
    "Ein zustandsbehafteter Agent, der selbstständig entscheidet, wann er in seiner RAG-Wissensdatenbank suchen muss."
)

api_key = os.environ.get("OPENROUTER_API_KEY")

if not api_key:
    st.error("OPENROUTER_API_KEY Umgebungsvariable ist nicht gesetzt.")
    st.stop()

st.sidebar.write("✅ OpenRouter API Key geladen")


# ---------------------------------------------------------
# 2. RAG Setup (Wissensdatenbank)
# ---------------------------------------------------------
@st.cache_resource
def setup_rag_tool(_api_key):
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=_api_key,
        base_url="https://openrouter.ai/api/v1",
    )

    # Dummy-Dokumente für das MVP
    docs = [
        Document(
            page_content="LangChain ist ein Open-Source-Framework zur Entwicklung von LLM-Anwendungen."
        ),
        Document(
            page_content="LangGraph ist eine Bibliothek, um komplexe, zustandsbehaftete Multi-Akteur-Agenten zu bauen."
        ),
        Document(
            page_content="Deep Agents nutzen LangGraph, um langfristige Gedächtnisse, Sandboxes und Tools zu orchestrieren."
        ),
        Document(
            page_content="Streamlit ist ein fantastisches Framework, um schnelle Web-Apps für Python und KI zu bauen."
        ),
    ]

    # Vektordatenbank erstellen
    vectorstore = FAISS.from_documents(docs, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    @tool
    def rag_tool(query: str) -> str:
        """Suche in der Wissensdatenbank nach Informationen zu LangChain, LangGraph, Deep Agents oder Streamlit."""
        docs = retriever.invoke(query)
        return "\n\n".join([doc.page_content for doc in docs])

    return rag_tool


# ---------------------------------------------------------
# 3. Agent Setup (Deep Agents)
# ---------------------------------------------------------
@st.cache_resource
def setup_agent(_api_key):
    rag_tool = setup_rag_tool(_api_key)

    agent = create_deep_agent(
        model="openrouter:openai/gpt-oss-20b:free",
        tools=[rag_tool],
        system_prompt=(
            "Du bist ein hilfreicher Assistent. Antworte auf Deutsch, halte Antworten knapp "
            "und nutze `rag_tool`, wenn du Wissen aus der eingebetteten Wissensdatenbank brauchst."
        ),
        checkpointer=MemorySaver(),
    )
    return agent


agent = setup_agent(api_key)

# ---------------------------------------------------------
# 4. Streamlit Chat Interface
# ---------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = "streamlit-session"

# Bisherigen Chatverlauf anzeigen
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


if prompt := st.chat_input("Frag mich etwas, z.B. 'Was sind Deep Agents?'"):
    # User-Nachricht zum UI und Session State hinzufügen
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Agent überlegt und durchsucht ggf. die Datenbank..."):
            config = {"configurable": {"thread_id": st.session_state.thread_id}}

            result = agent.invoke(
                {"messages": [HumanMessage(content=prompt)]},
                config=config,
            )

            final_response = result["messages"][-1].content
            st.markdown(final_response)

    st.session_state.messages.append({"role": "assistant", "content": final_response})
