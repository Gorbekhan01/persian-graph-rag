import os
from functools import lru_cache
from typing import List, Dict, Set, Optional
import networkx as nx
import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import Field
from sentence_transformers import CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity
from text_processor import clean_persian_text


@lru_cache(maxsize=2)
def get_base_documents(md_file_path: str) -> List[Document]:
    # Loads and processes a Markdown file into fine-grained document chunks.
    
   
    CHUNK_SIZE = 200
    CHUNK_OVERLAP = 100

    # Load raw text from the specified markdown file
    with open(md_file_path, "r", encoding="utf-8") as file:
        raw_text = file.read()

    # Preprocess and clean Persian text
    cleaned_text = clean_persian_text(raw_text)

    # Split text into small, fine-grained chunks suitable for MoGG graph construction
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "؟", ".", "!", " "],
    )
    chunks = text_splitter.split_text(cleaned_text)

    # Wrap chunks into LangChain Document objects with unique metadata IDs
    return [
        Document(page_content=chunk, metadata={"chunk_id": i})
        for i, chunk in enumerate(chunks)
    ]


@lru_cache(maxsize=2)
def get_embeddings():

    os.environ["HF_HUB_OFFLINE"] = "1"
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        encode_kwargs={"normalize_embeddings": True},
    )


def build_faiss_retriever(md_file_path: str, k: int = 5):
    # constructs a dense FAISS vector store retriever using MMR for diversity.
    docs = get_base_documents(md_file_path)
    embeddings = get_embeddings()
    vector_db = FAISS.from_documents(docs, embeddings)
    return vector_db.as_retriever(
        search_type="mmr", search_kwargs={"k": k, "lambda_mult": 0.7}
    )


class ProposedRetriever(BaseRetriever):
    """
    Advanced Retriever Pipeline with Graph-based Neighborhood Expansion and Reranking.
    
    Inherits from LangChain's BaseRetriever to support native LCEL composition.
    
    Pipeline Steps:
    1. FAISS Retrieval: Retrieves initial candidate seed nodes (0-hop).
    2. Context Graph Construction: Connects fine-grained nodes via sequential context 
       and cross-document semantic edges.
    3. Dynamic Hopping Router: Predicts the required graph expansion depth based on query complexity.
    4. Multi-hop Neighborhood Expansion: Traverses the graph to collect related contexts.
    5. Cross-Encoder Reranking: Reranks expanded candidate chunks and eliminates noise.
    """

    md_path: str
    top_k_retrieval: int = 15
    top_k_rerank: int = 5
    sim_threshold: float = 0.65

    # Internal components excluded from Pydantic direct instantiation
    faiss_retriever: Optional[BaseRetriever] = Field(default=None, exclude=True)
    reranker: Optional[CrossEncoder] = Field(default=None, exclude=True)
    docs: Optional[List[Document]] = Field(default=None, exclude=True)
    doc_map: Optional[Dict[int, Document]] = Field(default=None, exclude=True)
    doc_graph: Optional[nx.Graph] = Field(default=None, exclude=True)

    def __init__(
        self, 
        md_path: str, 
        top_k_retrieval: int = 15,
        top_k_rerank: int = 5,
        sim_threshold: float = 0.65,
        **kwargs
    ):
        super().__init__(
            md_path=md_path,
            top_k_retrieval=top_k_retrieval,
            top_k_rerank=top_k_rerank,
            sim_threshold=sim_threshold,
            **kwargs
        )

        self.faiss_retriever = build_faiss_retriever(md_path, k=top_k_retrieval)

        # Step 2: Load Cross-Encoder reranking model
        self.reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")

        # Step 3: Build the relational document graph using NetworkX
        self.docs = get_base_documents(md_path)
        self.doc_map = {
            doc.metadata["chunk_id"]: doc for doc in self.docs
        }
        self.doc_graph = self._build_semantic_context_graph()

    def _build_semantic_context_graph(self) -> nx.Graph:
        """
        Builds the graph containing two types of relationships:
        1. Sequential Edges: Connects adjacent text chunks to preserve local context.
        2. Semantic Edges: Connects distantly situated chunks across documents based on cosine similarity.
        """
        graph = nx.Graph()
        
        if not self.docs:
            return graph

        # Add all document chunks as nodes in the graph
        for doc in self.docs:
            chunk_id = doc.metadata["chunk_id"]
            graph.add_node(chunk_id, content=doc.page_content)

        # 1. Add sequential edges between neighboring chunks
        for i in range(len(self.docs) - 1):
            id_a = self.docs[i].metadata["chunk_id"]
            id_b = self.docs[i + 1].metadata["chunk_id"]
            graph.add_edge(id_a, id_b, weight=1.0, relation="sequential")

        # 2. Add semantic edges between non-adjacent chunks above similarity threshold
        embeddings_model = get_embeddings()
        texts = [doc.page_content for doc in self.docs]
        embeddings = embeddings_model.embed_documents(texts)
        emb_matrix = np.array(embeddings)

        # Compute cosine similarity matrix
        sim_matrix = cosine_similarity(emb_matrix)
        num_chunks = len(self.docs)

        for i in range(num_chunks):
            for j in range(i + 2, num_chunks):  # Skip adjacent chunks (already connected sequentially)
                sim = sim_matrix[i][j]
                if sim >= self.sim_threshold:
                    id_a = self.docs[i].metadata["chunk_id"]
                    id_b = self.docs[j].metadata["chunk_id"]
                    graph.add_edge(id_a, id_b, weight=float(sim), relation="semantic")

        return graph

    def _predict_expansion_hops(self, query: str) -> int:
        # Simple heuristic router based on query length.
        words = len(query.split())
        if words <= 5:
            return 0  # Simple queries: Retrieve seed nodes only (0-hop)
        elif words <= 12:
            return 1  # Moderate queries: 1-hop neighborhood expansion
        else:
            return 2  # Complex/multi-step queries: 2-hop graph expansion

    def _faiss_retrieve(self, query: str) -> List[Document]:
        return self.faiss_retriever.invoke(query)

    def _graph_neighborhood_expand(self, primary_docs: List[Document], hops: int) -> List[Document]:
        """
        Traverses the graph up to the predicted 'hops' distance 
        from the seed nodes to pull in enriched contextual chunks.
        """
        if hops == 0 or not primary_docs:
            return primary_docs

        retrieved_ids: Set[int] = set()

        # Traverse graph for each retrieved seed node
        for doc in primary_docs:
            chunk_id = doc.metadata.get("chunk_id")
            if chunk_id is not None and chunk_id in self.doc_graph:
                # Find all neighbor node IDs within the specified cutoff radius
                sub_graph_nodes = nx.single_source_shortest_path_length(
                    self.doc_graph, source=chunk_id, cutoff=hops
                )
                retrieved_ids.update(sub_graph_nodes.keys())

        # Map retrieved node IDs back to Document objects
        expanded_docs = [
            self.doc_map[c_id] for c_id in retrieved_ids if c_id in self.doc_map
        ]
        return expanded_docs

    def _cross_encoder_rerank(self, query: str, docs: List[Document]) -> List[Document]:
        # Reranks retrieved candidates using a Cross-Encoder 
        if not docs:
            return []

        pairs = [[query, doc.page_content] for doc in docs]
        scores = self.reranker.predict(pairs)

        # Sort documents by Cross-Encoder relevance score in descending order
        doc_score_pairs = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, score in doc_score_pairs[: self.top_k_rerank]]

    def _get_relevant_documents(
        self, query: str, *, run_manager: Optional[CallbackManagerForRetrieverRun] = None
    ) -> List[Document]:
        """
        Required LangChain BaseRetriever abstract method implementation.
        Executes the end-to-end retrieval process when invoked inside LCEL chains.
        """
        # Step 1: Perform initial FAISS retrieval to locate candidate seed nodes (0-hop)
        initial_docs = self._faiss_retrieve(query)

        # Step 2: Dynamically predict the graph hopping depth
        predicted_hops = self._predict_expansion_hops(query)

        # Step 3: Expand context via semantic & sequential graph traversal
        graph_expanded_docs = self._graph_neighborhood_expand(initial_docs, hops=predicted_hops)

        # Step 4: Apply final Cross-Encoder noise filtering and reranking
        final_context_docs = self._cross_encoder_rerank(query, graph_expanded_docs)

        return final_context_docs


def build_proposed_retriever(md_file_path: str):
    return ProposedRetriever(md_path=md_file_path)



def build_retriever_by_type(md_file_path: str, retriever_type: str = "faiss"):
    # Instantiate the requested retriever by name.
    retriever_type = retriever_type.lower()
    if retriever_type == "faiss":
        return build_faiss_retriever(md_file_path)
    elif retriever_type == "proposed":
        return build_proposed_retriever(md_file_path)
    else:
        raise ValueError(f"Unknown retriever type: {retriever_type}")
