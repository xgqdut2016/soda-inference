import sys
import pickle
import asyncio
import argparse

from datasets import load_dataset
import re

# ===== 定义 HTML 清理函数 =====
def clean_html(text):
    text = re.sub(r"<[^>]+>", "", text)  # 去除所有 HTML 标签
    text = text.replace("\n", "").strip()
    return text

sys.path.append('.')
sys.path.append('..')

import json, aiohttp
from loguru import logger
from typing import Any, List, Dict
import traceback

HEADERS = {"Content-Type": "application/json", "Accept": "application/json"}

class BGEM3Client:
    def __init__(self, url, **kargs) -> None:
        # super().__init__()
        self.url = url

    async def query_encode(self, query: str) -> List[float]:
        '''
            input args:
                embedding_url: str, e.g: "http://127.0.0.1:10992/embed"
                segments: str, e.g "北京天气怎么样"
            output args:
                response: dict, e.g {"dense_embed": [0.1, 0.5, ... , 0.8], "sparse_embed": {1: 0.5, 2: 0.62}}
        '''
        if not isinstance(query, str):
            logger.error(f"evoragError input text {query} is not str")
            raise ValueError("bge embedding input error")

        post_parm = dict(
            model = 'default',
            text = query,
            return_dense = True,
            return_sparse = True,
        )

        async with aiohttp.ClientSession() as session:
            async with session.post(self.url, json=post_parm, headers=HEADERS, timeout=5) as response:
                if response.status != 200:
                    raise ConnectionError("embedding api failed to visit")
                r = await response.text()
                result = json.loads(r)

        result['sparse_embed'] = {int(k): float(v) for k, v in result['sparse_embed'].items()}
        return result

    def document_encode(self, document: List[str]):
        raise NotImplementedError("document_encode is not yet implemented!")

    def encode(self, text: List[str]):
        return super().encode(text)

HEADERS = {"Content-Type": "application/json", "Accept": "application/json"}

class SodaRerankClient:
    def __init__(self, url: str, rerank_batch: int=5) -> None:
        # super().__init__()
        self.url = url
        self.rerank_batch = rerank_batch
        logger.info(f"{self.__class__.__name__} use url in {url}")

    async def scoring(self, query: str, contents: List[str]) -> List[float]:
        async def scoring_request(query: str, contents: List[str]):
            async with aiohttp.ClientSession() as session:
                async with session.post(self.url, json={"query": query, "texts": contents}, headers=HEADERS, timeout=5) as response:
                    if response.status != 200:
                        raise ValueError(f"response.status is {response.status}")
                    r = await response.text()
                    scores: list[float] = json.loads(r)
                    if not isinstance(scores, list):
                        logger.error(f"evoragError scores result is not list: {scores}")
                    if len(scores) != len(contents):
                        logger.error(f"evoragError scores lens is not equal contents: {scores} and {contents}")
                    return scores

        try:
            # 分发给多台机器计算，增加并行度，降低延迟
            BATCH = self.rerank_batch
            tasks = [scoring_request(query, contents=contents[idx: idx + BATCH]) for idx in range(0, len(contents), BATCH)]
            results = await asyncio.gather(*tasks)
            return sum(results, [])
        except Exception as e:
            logger.error(f'evoragError reranker raise {traceback.format_exc()}')
            return [1.0] * len(contents)
        return scores

query = [
    '我最低选多少课',
    '离东南门最近的健身房在哪？',
    '军训期间的4学分算在大一本科生26学分的范围内吗',
    'GPA计算原则',
    '限制选择的26学分包括军训的4学分吗？',
    '这26学分是一个学期限制的吗？',
    '军事技能考核总分多少分能得到两学分',
    '军训时的4学分会算在大一学期的26学分限制中吗？即第一学期还可以选多少学分的课？',
]

documents = [
    '最低选课数目是4门',
    '东南门最近的健身房是xx健身房',
    '军训期间的4学分算在大一本科生26学分的范围内吗是xx',
    'GPA计算原则是xx',
    '限制选择的26学分包括军训的4学分',
    '26学分是一个学期限制的',
    '军事技能考核总分多少分能得到两学分是xx',
    '军训时的4学分会算在大一学期的26学分限制中吗？即第一学期还可以选多少学分的课？是xx',
]

embedding_client = BGEM3Client(url='http://127.0.0.1:10991/embed')
reranker_client = SodaRerankClient(url='http://127.0.0.1:10993/rerank')

async def main():
    parser = argparse.ArgumentParser(description='test mteb data')
    parser.add_argument('--dataset_path', type=str, help='dataset path')
    parser.add_argument('--output_path', type=str, help='dataset path')

    args = parser.parse_args()
    dataset_path = args.dataset_path
    output_path = args.output_path

    if dataset_path is None:
        print("使用预定义的 query 和 documents")
        queries = query
        corpus = documents
    else:
        dataset = load_dataset(dataset_path)
        # ===== 提取 queries 和 corpus 文本 =====
        queries = [item["text"].strip() for item in dataset["queries"]]
        corpus = [clean_html(item["text"]) for item in dataset["corpus"]]
        num_samples = 16
        queries = queries[:num_samples]
        corpus = corpus[:num_samples]

    embedding_list = []
    reranker_list = []
    for q, d in zip(query, documents):
        embedding = await embedding_client.query_encode(q)
        reranker = await reranker_client.scoring(q, [d])
        embedding_list.append(embedding)
        reranker_list.append(reranker)

    with open(output_path + "embedding.pkl", "wb") as f:
        pickle.dump(embedding_list, f)
    with open(output_path + "reranker.pkl", "wb") as f:
        pickle.dump(reranker_list, f)

if __name__ == '__main__':
    asyncio.run(main())
