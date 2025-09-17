from BCEmbedding import RerankerModel

from common.abs_reranker import AbstractRerank

class BceRerankerInfer(AbstractRerank):
    def __init__(
            self,
            model_path: str,
            padding='longest',
            max_length=512,
            batch_size=256,
            device='cuda',
            **build_kwargs,
    ) -> None:
        self.rerank_model = RerankerModel(
            model_name_or_path=model_path,
            device=device,
            **build_kwargs,
        )
        self.padding = padding
        self.max_length = max_length
        self.batch_size = batch_size

    def rerank(self, texts: list[tuple[str, str]]) -> list[float]:
        scores = self.rerank_model.compute_score(
            texts,
            batch_size=self.batch_size,
            max_length=self.max_length,
        )
        if isinstance(scores, float):
            scores = [scores]
        return scores
