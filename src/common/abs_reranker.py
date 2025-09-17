class AbstractRerank:
    ''' 使用模型对文本对进行相关度打分
    '''

    def rerank(self, texts: list[tuple[str, str]]) -> list[float]:
        raise NotImplementedError()

    def run_gather(self, inputs):
        return self.rerank(inputs)
