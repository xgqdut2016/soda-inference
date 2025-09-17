FROM ac2-registry.cn-hangzhou.cr.aliyuncs.com/ac2/cuda:12.1.1-cudnn8-devel-alinux3.2304

FROM alibaba-cloud-linux-3-registry.cn-hangzhou.cr.aliyuncs.com/alinux3/python:3.11.1

WORKDIR /api
COPY ./requirements.txt .
ENV PIP_INDEX_URL=${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple/}

RUN pip install -r requirements.txt
RUN pip install onnxruntime-gpu==1.18.0 --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/
COPY . .

WORKDIR /api/src

CMD ["python", "-u", "start_server.py"]
