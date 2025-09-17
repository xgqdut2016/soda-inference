import ctranslate2
from transformers import T5Tokenizer

from common.abs_rewrite import AbstractReWrite

# class AbstractReWrite:
#     ''' 拿到改写的结果
#     '''

#     def rerank(self, texts: list[tuple[str, str, str]]) -> list[str]:
#         raise NotImplementedError()

#     def run_gather(self, inputs):
#         return self.rerank(inputs)


class MT5RewriteInfer(AbstractReWrite):
    def __init__(
            self,
            model_path: str,
            padding='longest',
            max_length=512,
            batch_size=256,
            device='cuda',
            **build_kwargs,
    ) -> None:
        self.tokenizer = T5Tokenizer.from_pretrained(model_path)
        self.model = ctranslate2.Translator(model_path,  device=device)
        self.padding = padding
        self.max_length = max_length
        self.batch_size = batch_size

    def rewrite(self, texts: list[list[str]]) -> list[str]:
        input_text = ["[SEP]".join(item) for item in texts]
        input_idx = [self.tokenizer.convert_ids_to_tokens(self.tokenizer.encode(item)) for item in input_text]
        results = self.model.translate_batch(input_idx, beam_size=4, max_decoding_length=64)

        rewrite_list = []
        for item in results:
            ret = self.tokenizer.decode(self.tokenizer.convert_tokens_to_ids(item.hypotheses[0]), skip_special_tokens=True)
            rewrite_list.append(ret)
        return rewrite_list

if __name__ == "__main__":
    model = MT5RewriteInfer(model_path="/home/guodewen/models/mt5", )
    res = model.rerank([["青花瓷是谁的", "周杰伦唱的歌，他还有很多歌曲", "他还有什么作品"], ["青花瓷是谁的", "周杰伦唱的歌，他还有很多歌曲", "他还有什么作品"]])
    print(res)