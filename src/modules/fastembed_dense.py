from fastembed import TextEmbedding

from common.abs_embedding import AbstractEmbed

class FastembedDense(AbstractEmbed):
    def __init__(
            self,
            model_name: str,
            model_path: str,
            max_length=8192,
            batch_size=12,
            **build_kwargs,
    ) -> None:
        self.model = TextEmbedding(
            model_name=model_name,
            cache_dir=model_path,
            **build_kwargs
        )

        self.max_length = max_length
        self.batch_size = batch_size

    def encode(self, texts: list[str]) -> list[tuple[None, dict[int, float]]]:
        dense_nps = self.model.embed(texts, batch_size=self.batch_size)
        denses = [i.tolist() for i in dense_nps]
        assert len(denses) == len(texts)

        sparses = [None for _ in texts]

        return list(zip(denses, sparses))

    def can_encode(self, return_dense: bool, return_sparse: bool) -> str | None:
        if return_sparse == True:
            return 'sparse is unsupported in this model'
        return None
