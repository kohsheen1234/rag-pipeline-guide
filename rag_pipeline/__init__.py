"""A RAG pipeline built one step at a time.

One module per pipeline stage, in data-flow order:

    ingestion  ->  raw files on disk become document records

Chunking, embedding, indexing, retrieval, generation, and evaluation are added
as the guide progresses.
"""
