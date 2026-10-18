from qdrant_client import QdrantClient, models


def collection_name(kb_id: str) -> str:
    return f"kb_{kb_id}"


def _doc_filter(doc_id: str) -> models.Filter:
    return models.Filter(
        must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))]
    )


def create_collection(client: QdrantClient, kb_id: str, dim: int) -> None:
    name = collection_name(kb_id)
    client.create_collection(
        name,
        vectors_config={"dense": models.VectorParams(size=dim, distance=models.Distance.COSINE)},
        sparse_vectors_config={"sparse": models.SparseVectorParams()},
    )
    client.create_payload_index(name, "doc_id", models.PayloadSchemaType.KEYWORD)


def drop_collection(client: QdrantClient, kb_id: str) -> None:
    client.delete_collection(collection_name(kb_id))


def upsert(client: QdrantClient, kb_id: str, points: list[models.PointStruct]) -> None:
    client.upsert(collection_name(kb_id), points=points, wait=True)


def delete_document(client: QdrantClient, kb_id: str, doc_id: str) -> None:
    client.delete(
        collection_name(kb_id),
        points_selector=models.FilterSelector(filter=_doc_filter(doc_id)),
        wait=True,
    )


def count_document(client: QdrantClient, kb_id: str, doc_id: str) -> int:
    return client.count(collection_name(kb_id), count_filter=_doc_filter(doc_id)).count
