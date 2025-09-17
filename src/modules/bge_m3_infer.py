from FlagEmbedding import BGEM3FlagModel

from common.abs_embedding import AbstractEmbed

class BgeM3Infer(AbstractEmbed):
    def __init__(
            self,
            model_path: str,
            padding='longest',
            max_length=8192,
            batch_size=12,
            device='cuda',
            **build_kwargs,
    ) -> None:
        self.bge_m3_model = BGEM3FlagModel(
            model_name_or_path=model_path,
            device=device,
            **build_kwargs,
        )
        self.padding = padding
        self.max_length = max_length
        self.batch_size = batch_size

    def encode(self, texts: list[str]) -> list[tuple[list[float], dict[str, float]]]:
        result_dict = self.bge_m3_model.encode(
            texts,
            batch_size=self.batch_size,
            max_length=self.max_length,
            return_dense=True,
            return_sparse=True,
        )
        denses = result_dict['dense_vecs'].tolist()
        sparses = [{int(k): float(v) for k, v in i.items()} for i in result_dict['lexical_weights']]
        assert len(denses) == len(texts)
        assert len(sparses) == len(texts)
        return list(zip(denses, sparses))

    def can_encode(self, return_dense: bool, return_sparse: bool) -> str | None:
        return None
