import requests


def get_rerank(
        query,
        texts,
        model = 'default',
        url = "http://127.0.0.1:18800/rerank",
    ):
    resp = requests.post(url, json=dict(
        model = model,
        query = query,
        texts = texts,
    ))
    assert resp.status_code == 200, resp.content
    resp_json = resp.json()
    return resp_json

if __name__ == "__main__":
    # print(get_rerank("haha", ["hehe", "sd"]))
    # res_torch = get_rerank("haha", ["hehe", "hoho"], url="http://127.0.0.1:10993/rerank")
    res_onnx = get_rerank("haha", ["hehe", "hoho"], url="http://127.0.0.1:10994/rerank")
    # res_torch = get_rerank("haha", ["hehe", "hoho"], url="http://127.0.0.1:10995/rerank")
    # res_onnx = get_rerank("haha", ["hehe", "hoho"], url="http://127.0.0.1:10996/rerank")
    # print(res_torch)
    print(res_onnx)
