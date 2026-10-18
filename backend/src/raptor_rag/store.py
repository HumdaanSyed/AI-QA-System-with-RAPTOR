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


def search(
    client: QdrantClient,
    kb_id: str,
    dense: list[float],
    sparse: models.SparseVector | None,
    doc_ids: list[str],
    limit: int,
    passages_only: bool = False,
) -> list[dict]:
    """Search the given docs; with a sparse vector, fuse dense + sparse results (RRF).

    passages_only skips RAPTOR summaries (used to evaluate retrieval without them).
    """
    must: list[models.Condition] = [
        models.FieldCondition(key="doc_id", match=models.MatchAny(any=doc_ids))
    ]
    if passages_only:
        must.append(models.FieldCondition(key="level", match=models.MatchValue(value=0)))
    only = models.Filter(must=must)
    if sparse is None:
        result = client.query_points(
            collection_name(kb_id), query=dense, using="dense", query_filter=only, limit=limit
        )
    else:
        result = client.query_points(
            collection_name(kb_id),
            prefetch=[
                models.Prefetch(query=dense, using="dense", filter=only, limit=limit),
                models.Prefetch(query=sparse, using="sparse", filter=only, limit=limit),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit,
        )
    return [p.payload or {} for p in result.points]


def count_document(client: QdrantClient, kb_id: str, doc_id: str) -> int:
    return client.count(collection_name(kb_id), count_filter=_doc_filter(doc_id)).count
