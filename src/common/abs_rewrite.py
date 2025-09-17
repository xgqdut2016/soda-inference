class AbstractReWrite:
    ''' 拿到改写的结果
    '''

    def rewrite(self, texts: list[tuple[str, str, str]]) -> list[str]:
        raise NotImplementedError()

    def run_gather(self, inputs):
        return self.rewrite(inputs)
