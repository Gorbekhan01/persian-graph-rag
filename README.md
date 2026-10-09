# Persian RAG with Graph-Enhanced Retrieval


## Abstract
Retrieval-Augmented Generation (RAG) is an advanced technique in AI used to retrieve the most relevant answers to a user's query based on a given context. However, implementing a RAG system to achieve exact results typically requires embedding and generator models with a high parameter count, demanding high-performance hardware. Furthermore, when the context language is a low-resource language such as Persian, issues like hallucinations and out-of-context answers tend to increase. In this project, taking inspiration from [MOGG (Mix-of-Granularity-Graph)]("https://arxiv.org/abs/2406.00456"), I present a lightweight approach capable of running on low-performance systems while improving upon three core factors: faithfulness, answer relevance, and context relevance.

---

## Introduction
A RAG system consists of three main steps: Retrieval, Augmentation, and Generation. Retrieval methods are generally categorized into sparse and dense. Since dense retrieval accounts for semantic similarity, its results are usually more accurate, though more computationally expensive. Initially, the context is split into multiple chunks, with neighboring chunks sharing some overlap to prevent truncated sentences. In the next step, every chunk is mapped to a vector of fixed dimensions using embedding models and stored in a vector database. Subsequently, by converting the query into the same vector space, the system retrieves the most similar vector using cosine similarity and passes it to an LLM to generate the final response. 

While this standard path is used in simple FAISS-based RAG systems, it lacks high accuracy. The proposed RAG method improves retrieval performance by incorporating reranking and a simplified MOGG approach.

---

## Code Architecture & Implementation
The project is implemented modularly in Python using LangChain and NetworkX. The core codebase comprises the following modules:

1. **`src/textـprocessor.py`**: Handles text cleaning and Persian-specific normalization using the `Parsivar` library (removing invisible Unicode marks, fixing spacing, and preserving paragraph structures).
2. **`src/retriever.py`**: 
   - Implements `ProposedRetriever` inheriting from LangChain's `BaseRetriever`.
   - **Graph Construction (`_build_semantic_context_graph`)**: Builds a NetworkX graph where nodes represent fine-grained text chunks (`chunk_size=200`, `chunk_overlap=100`). It establishes two types of edges:
     - *Sequential Edges*: Connects adjacent text chunks to preserve local context.
     - *Semantic Edges*: Connects non-adjacent chunks across documents based on cosine similarity exceeding a threshold ($\ge 0.65$).
   - **Dynamic Hopping Router (`_predict_expansion_hops`)**: Predicts the required graph expansion depth ($N$-hop) based on the word count of the query (e.g., $\le 5$ words: 0-hop; $\le 12$ words: 1-hop; $>12$ words: 2-hop).
3. **`src/pipeline.py`**: Constructs the LCEL (LangChain Expression Language) RAG pipeline integrating `ChatOllama` (`gemma3:4b`) with a strictly tailored Persian prompt template that eliminates external hallucinations and unsupported reasoning.
4. **`src/evaluate.py`**: Runs comparative evaluation using an LLM-as-a-Judge (`openai/gpt-oss-120b` via Groq) across three rubrics: Faithfulness, Answer Relevance, and Context Relevance.

---

## Workflow
To improve retrieval in our RAG pipeline, after receiving the context, the proposed method builds a similarity graph using the vectors stored in the vector database. In this graph, chunks with high semantic similarity share a mutual edge (if their cosine similarity exceeds a specific threshold). 

Upon receiving a query, a simple routing logic determines the number of hops (by analyzing the word count of the query). The hop count indicates how many neighboring nodes should be added to the final context. For instance, if the hop count is 2, for every chunk retrieved in the initial search, the model adds all neighboring chunks within a maximum distance of 2. This expands the final context and increases the likelihood of including the exact answer.

---

## Experiment & Evaluation
This model runs entirely locally. The models used in this experiment are listed in the table below:

| Component | Model |
| :--- | :--- |
| **Embedding Model** | `BAAI/bge-m3` |
| **Reranking Model** | `BAAI/bge-reranker-v2-m3` |
| **Generator Model** | `gemma3:4b` (Local via Ollama) |
| **Evaluation Judge** | `openai/gpt-oss-120b` (via Groq) |

To evaluate the proposed model, Iran's 11th-grade chemistry textbook was used as the dataset. This book presents a complex context containing dispersed definitions (ideal for multi-hop questions), mathematical equations, numbers, dates, and other structured data. DeepSeek V3 was used to extract the text from the book and convert it into a Markdown file (`dataset/md-book.md`). Additionally, Claude generated the `Q&A.json` file from the context. 

The Q&A dataset contains multiple question types to assess the model across different scenarios:
- **Simple:** 17 questions (Single-hop questions explicitly answered within the context)
- **Multi-hop:** 18 questions (Complex questions whose answers are located in different parts of the book)
- **Conceptual:** 10 questions (Questions requiring deeper comprehension)
- **Trap:** 5 questions (Questions whose answers are *not* present in the context)

### Experimental Results

| Model | Avg. Faithfulness | Avg. Answer Relevance | Avg. Context Relevance | Total Execution Time (sec) | Avg. Latency per Question (sec) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **PROPOSED** | 4.78 | 3.54 | 3.48 | 1422.2 | 28.44 |
| **FAISS** | 4.56 | 3.02 | 3.20 | 466.36 | 9.33 |

Despite running fully on limited hardware (M2) with a 4B-parameter model, the proposed method outperformed FAISS across all three core metrics. This demonstrates that smart graph design and dynamic routing can meaningfully improve Persian RAG quality without requiring heavy resources.


> Detailed results are available in the `results/` directory (including `overall_results.csv`, `category_matrix.csv`, and individual evaluation reports).