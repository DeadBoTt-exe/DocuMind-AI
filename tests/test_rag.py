"""
RAG integration test.

Validates that the RAG engine can be instantiated and can successfully
answer a question using retrieved documentation context.
This test verifies the end-to-end retrieval and generation pipeline.
"""

from app.rag import RAGEngine

if __name__ == "__main__":
    rag = RAGEngine()
    result = rag.ask("Where is GST calculated?")

    print("\nANSWER:")
    print(result["answer"])

    print("\nSOURCES:")
    for src in result["sources"]:
        print("-", src)
