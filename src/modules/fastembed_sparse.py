from fastembed import SparseTextEmbedding

from common.abs_embedding import AbstractEmbed

class FastembedSparse(AbstractEmbed):
    def __init__(
            self,
            model_name: str,
            model_path: str,
            max_length=8192,
            batch_size=12,
            **build_kwargs,
    ) -> None:
        self.model = SparseTextEmbedding(
            model_name=model_name,
            cache_dir=model_path,
            **build_kwargs
        )

        self.max_length = max_length
        self.batch_size = batch_size

    def encode(self, texts: list[str]) -> list[tuple[None, dict[int, float]]]:
        denses = [None for _ in texts]

        sparse_nps = self.model.embed(texts, batch_size=self.batch_size)
        sparses = [{int(k): float(v) for k, v in zip(i.indices, i.values)} for i in sparse_nps]
        assert len(sparses) == len(texts)

        return list(zip(denses, sparses))

    def can_encode(self, return_dense: bool, return_sparse: bool) -> str | None:
        if return_dense == True:
            return 'dense is unsupported in this model'
        return None
