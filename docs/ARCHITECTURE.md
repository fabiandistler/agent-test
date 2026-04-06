# Architektur

## Übersicht

```
┌─────────────────────────────────────────────────────────────┐
│                      Streamlit UI                            │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  Chat Interface (st.chat_input / st.chat_message)   │    │
│  └─────────────────────────────────────────────────────┘    │
│                         │                                    │
│                    Session State                             │
│              (messages, thread_id)                           │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                   Deep Agent (LangGraph)                     │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  create_deep_agent()                                 │    │
│  │  - Model: OpenRouter (gpt-oss-20b:free)             │    │
│  │  - Tools: [rag_tool]                                │    │
│  │  - Checkpointer: MemorySaver                         │    │
│  └─────────────────────────────────────────────────────┘    │
│                         │                                    │
│              ┌──────────┴──────────┐                         │
│              ▼                     ▼                         │
│      Tool Decision          Direct Response                   │
│   (braucht RAG-Wissen?)                                       │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                      RAG Pipeline                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐      │
│  │  Documents  │───▶│  Embeddings │───▶│   FAISS     │      │
│  │  (4 Dummy)  │    │ (OpenRouter)│    │ Vectorstore │      │
│  └─────────────┘    └─────────────┘    └─────────────┘      │
│                                                │              │
│                                                ▼              │
│                                        retriever.invoke()     │
└─────────────────────────────────────────────────────────────┘
```

## Komponenten

### 1. Streamlit UI

- **Chat Interface**: `st.chat_input()` für User-Input, `st.chat_message()` für Darstellung
- **Session State**: Speichert `messages` (Chatverlauf) und `thread_id` (Konversations-ID)
- **Cached Resources**: `@st.cache_resource` für einmalige Initialisierung von RAG und Agent

### 2. LangGraph Agent

- **create_deep_agent()**: Erzeugt einen Agenten mit LangGraph-Orchestration
- **MemorySaver**: Checkpointer für persistente Konversationszustände
- **Tool Orchestration**: Agent entscheidet autonom, wann `rag_tool` genutzt wird

### 3. RAG Pipeline

- **Documents**: 4 Dummy-Dokumente zu LangChain, LangGraph, Deep Agents, Streamlit
- **Embeddings**: OpenAI `text-embedding-3-small` via OpenRouter
- **FAISS**: In-Memory Vektordatenbank für Ähnlichkeitssuche
- **Retriever**: Gibt Top-2 relevante Dokumente zurück

## Datenfluss

1. User sendet Nachricht via Streamlit
2. Nachricht wird an Agent mit `thread_id` übergeben
3. Agent prüft: RAG-Tool nötig?
   - Ja: RAG durchsucht Vektordatenbank, liefert Kontext
   - Nein: Direkte Antwort
4. Antwort wird im Chat angezeigt und im Session State gespeichert
5. Checkpointer persistiert Konversationszustand für Folge-Anfragen
