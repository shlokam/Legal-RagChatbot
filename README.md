# 🏛️ Indian Income Tax RAG System

> Production-grade Retrieval-Augmented Generation (RAG) system specialized in Indian Income Tax Law, powered by AWS S3 Vectors and Groq LLM inference.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![AWS](https://img.shields.io/badge/AWS-S3%20Vectors-orange.svg)](https://aws.amazon.com/s3/)
[![LLM](https://img.shields.io/badge/LLM-Llama--4%20Maverick-purple.svg)](https://groq.com/)

## 📋 Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Features](#features)
- [Technology Stack](#technology-stack)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Architecture Deep Dive](#architecture-deep-dive)
- [Design Patterns](#design-patterns)
- [Performance](#performance)
- [Deployment](#deployment)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This system provides an intelligent legal assistant for Indian Income Tax Law queries by combining:
- **Semantic Search** via AWS S3 Vectors for case law retrieval
- **LLM-powered Reasoning** using Groq's Llama-4 Maverick (17B parameters)
- **Intelligent Routing** between direct answers and RAG-based responses
- **Citation-rich Responses** with source document traceability

### Key Capabilities

- Query classification (general vs. case law requirements)
- Dense semantic search across legal documents
- Context-aware answer generation with judicial citations
- Conversational follow-up questions
- Sub-second vector search latency

## 🏗️ System Architecture
```mermaid
flowchart TB
    A[FastAPI REST API<br/>app.py] -->|POST /decide| B[Decision Engine<br/>main.py]
    B -->|Tool Selection| G[Groq LLM<br/>Llama-4 Maverick]
    B -->|RAG Path| C[RAG Pipeline<br/>ragPipeline.py]
    B -->|Direct Answer| G
    
    C --> D[RAG Retriever<br/>ragRetreiver.py]
    D --> E[Embedding Manager<br/>embedding.py]
    D --> F[AWS S3 Vectors<br/>vectorStore_AWS.py]
    
    E -->|Generate 384-dim<br/>Embeddings| F
    C -->|Generate Answer| G
    
    H[Configuration<br/>configuration.py] -.->|API Keys| G
    H -.->|AWS Config| F
    
    style A fill:#e1f5ff
    style B fill:#fff4e1
    style C fill:#fff4e1
    style D fill:#f0e1ff
    style E fill:#f0e1ff
    style F fill:#e1ffe1
    style G fill:#ffe1e1
    style H fill:#f5f5f5
```

## ✨ Features

### Intelligent Query Routing
- **Automatic Tool Selection**: LLM classifies queries as general (direct answer) vs. specific (RAG required)
- **Query Optimization**: Transforms user questions into dense semantic search queries with statutory context

### Advanced RAG Pipeline
- **Top-K Retrieval**: Configurable document retrieval (default: 10 documents)
- **Score Thresholding**: Minimum similarity filtering (default: 0.3)
- **Metadata Extraction**: Automatic PDF path construction from case metadata
- **Citation Management**: Tracks and lists all referred legal documents

### Conversational AI
- **Context Preservation**: Summary-based conversation history
- **Follow-up Generation**: Intelligent next-step question suggestions
- **Professional Tone**: Legal assistant persona with junior associate-friendly explanations

## 🛠️ Technology Stack

| Component | Technology | Version/Model | Purpose |
|-----------|-----------|---------------|---------|
| **Web Framework** | FastAPI | 0.104+ | REST API endpoints |
| **LLM Provider** | Groq (ChatGroq) | Llama-4 Maverick 17B | Fast inference (128K context) |
| **Vector Database** | AWS S3 Vectors | Boto3 SDK | Distributed vector storage |
| **Embedding Model** | SentenceTransformer | all-MiniLM-L6-v2 | 384-dimensional embeddings |
| **Orchestration** | LangChain | Latest | LLM workflow management |
| **Language** | Python | 3.8+ | Core implementation |






