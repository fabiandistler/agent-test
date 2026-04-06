# Deep Agent RAG MVP

Demo-Projekt, das die Integration von **LangGraph** und **Streamlit** für einen zustandsbehafteten KI-Agenten mit RAG zeigt.

## Features

- **LangGraph State Management**: Zustandsbehafteter Agent mit Memory Checkpointing
- **RAG Integration**: Vektorsuche mit FAISS für kontextbezogene Antworten
- **Streamlit UI**: Einfache Chat-Oberfläche mit Session-Management
- **OpenRouter API**: Flexibler Zugriff auf verschiedene LLM-Modelle

## Installation

```bash
# Repository klonen
git clone <repo-url>
cd agent-test

# Abhängigkeiten installieren (mit uv)
uv sync
```

## Ausführung

```bash
# API Key setzen
export OPENROUTER_API_KEY="dein-api-key"

# App starten
uv run streamlit run app.py
```

## Projektstruktur

```
├── app.py              # Hauptanwendung (Streamlit + LangGraph)
├── pyproject.toml      # Projekt-Dependencies
└── docs/
    └── ARCHITECTURE.md # Architektur-Dokumentation
```

## Technologien

| Library | Zweck |
|---------|-------|
| LangGraph | Zustandsbehaftete Agenten-Orchestration |
| Streamlit | Web-Oberfläche |
| FAISS | Vektorsuche für RAG |
| Deep Agents | High-Level Agent-Abstraktion |
