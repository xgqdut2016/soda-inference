import requests


def get_emb(
        text,
        model = 'default',
        return_dense = True,
        return_sparse = True,
        url = "http://127.0.0.1:10991/embed",
    ):
    resp = requests.post(url, json=dict(
        model = model,
        text = text,
        return_dense = return_dense,
        return_sparse = return_sparse,
    ))
    assert resp.status_code == 200, resp.content
    resp_json = resp.json()
    return resp_json

if __name__ == "__main__":
    # print(get_emb("haha"))
    print(get_emb(["haha", "hehe"]))
    # res_torch = get_emb(["haha", "hehe"], url="http://127.0.0.1:10991/embed")
    # res_onnx = get_emb(["haha", "hehe"], url="http://127.0.0.1:10992/embed")
    # print('hoho')
