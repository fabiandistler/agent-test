import os
import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain.tools import tool
from langgraph.prebuilt import create_react_agent

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
        openai_api_key=_api_key, base_url="https://openrouter.ai/v1"
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

    # Als Tool für den Agenten verpacken
    @tool
    def rag_tool(query: str) -> str:
        """Suche in der Wissensdatenbank nach Informationen zu LangChain, LangGraph, Deep Agents oder Streamlit."""
        docs = retriever.invoke(query)
        return "\n\n".join([doc.page_content for doc in docs])

    return rag_tool


# ---------------------------------------------------------
# 3. Agent Setup (LangGraph)
# ---------------------------------------------------------
@st.cache_resource
def setup_agent(_api_key):
    # LLM initialisieren (GPT-4o-mini ist schnell und günstig fürs MVP)
    llm = ChatOpenAI(
        model="minimax/minimax-m2.5:free",
        openai_api_key=_api_key,
        base_url="https://openrouter.ai/v1",
        temperature=0,
    )

    # RAG Tool laden
    rag_tool = setup_rag_tool(_api_key)

    # LangGraph ReAct Agent erstellen
    agent = create_react_agent(llm, tools=[rag_tool])
    return agent


agent = setup_agent(api_key)

# ---------------------------------------------------------
# 4. Streamlit Chat Interface
# ---------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# Bisherigen Chatverlauf anzeigen
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User Input
if prompt := st.chat_input("Frag mich etwas, z.B. 'Was sind Deep Agents?'"):
    # User-Nachricht zum UI und Session State hinzufügen
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Agenten-Antwort generieren
    with st.chat_message("assistant"):
        with st.spinner("Agent überlegt und durchsucht ggf. die Datenbank..."):
            # Die komplette Message-Historie an den Agenten übergeben (für den Kontext)
            # LangGraph erwartet ein Dict mit einem "messages" Array
            inputs = {
                "messages": [
                    (msg["role"], msg["content"]) for msg in st.session_state.messages
                ]
            }

            # Agent aufrufen
            result = agent.invoke(inputs)

            # Die letzte Nachricht ist die finale Antwort des LLMs
            final_response = result["messages"][-1].content
            st.markdown(final_response)

    # Assistant-Nachricht zum Session State hinzufügen
    st.session_state.messages.append({"role": "assistant", "content": final_response})
