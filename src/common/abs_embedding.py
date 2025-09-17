class AbstractEmbed:
    ''' 使用模型将文本表征为向量
    '''

    def can_encode(self, return_dense: bool, return_sparse: bool) -> str | None:
        raise NotImplementedError()

    def encode(self, texts: list[str]) -> list[tuple[list[float] | None, dict[int | str, float] | None]]:
        raise NotImplementedError()

    def run_gather(self, inputs):
        return self.encode(inputs)
