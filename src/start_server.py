import atexit
from contextlib import AbstractContextManager
import threading
from common import consts
from common.abs_embedding import AbstractEmbed
from common.abs_reranker import AbstractRerank
from common.abs_rewrite import AbstractReWrite
from common.batch_gather import BatchGather
from flask import Flask, request
from flask import jsonify, Response
from modules import MODEL_DICT_EMBED, MODEL_DICT_RERANK, MODEL_DICT_REWRITE
from prometheus_client import Counter, Gauge, Summary, Histogram
from prometheus_client import start_http_server as start_prometheus_http_server
from validx.cy import (
    Dict as VDict,
    List as VList,
    Str as VStr,
    Float as VFloat,
    Int as VInt,
    Bool as VBool,
    OneOf,
)
from validx.exc import ValidationError

class runner_timer(AbstractContextManager):
    def __init__(self, time, function, args=None, kwargs=None):
        self.timer = threading.Timer(time, function, args, kwargs);
        self.timer.start()
    def __enter__(self):
        return self.timer
    def __exit__(self, *exc_info):
        self.timer.cancel()

model_counter = Counter('model_counter', 'model_counter', ['model_type', 'model_name', 'model_cls_str', 'status'])
model_time_elapse = Histogram('model_time_elapse', 'model_time_elapse', ['model_type', 'model_name', 'model_cls_str'])
class InferServer:
    def __init__(self) -> None:
        self.app = Flask(__name__)
        self.model_batch_exec_embed: dict[str, tuple[AbstractEmbed, BatchGather]] = {}
        self.model_batch_exec_rerank: dict[str, tuple[AbstractRerank, BatchGather]] = {}
        self.model_batch_exec_rewrite: dict[str, tuple[AbstractReWrite, BatchGather]] = {}

        self.app.add_url_rule('/embed', view_func=self.embed, methods=['get', 'post'])
        self.app.add_url_rule('/rerank', view_func=self.rerank, methods=['get', 'post'])
        self.app.add_url_rule('/rewrite', view_func=self.rewrite, methods=['get', 'post'])
        self.valid_schema_embed = VDict({
            'model': VStr(),
            'text': OneOf(VStr(), VList(VStr(),minlen=1)),
            'return_dense': VBool(),
            'return_sparse': VBool(),
        }, optional=["model", 'return_dense', 'return_sparse'],)
        self.valid_schema_rerank = VDict({
            'model': VStr(),
            'query': VStr(),
            'texts': VList(VStr(),minlen=1),
        }, optional=["model"],)
        self.valid_schema_rewrite = VDict({
            'message': VList(VList(VStr(), minlen=1), minlen=1),
        }, optional=[],)

    def add_model(
            self,
            model_type,
            model_name,
            model_cls_str,
            batch_size_limit,
            queue_size_limit,
            build_kwargs,
        ):
        if model_type == 'embed':
            self.app.logger.info('start to load embed model:: {}'.format(model_name))
            if model_cls_str not in MODEL_DICT_EMBED:
                raise ValueError(f'{model_cls_str} is not find in embed dict')
            model_cls = MODEL_DICT_EMBED[model_cls_str]
            model_batch_exec = self.model_batch_exec_embed
        elif model_type == 'rerank':
            self.app.logger.info('start to load rerank model:: {}'.format(model_name))
            if model_cls_str not in MODEL_DICT_RERANK:
                raise ValueError(f'{model_cls_str} is not find in rerank dict')
            model_cls = MODEL_DICT_RERANK[model_cls_str]
            model_batch_exec = self.model_batch_exec_rerank
        elif model_type == 'rewrite':
            self.app.logger.info('start to load rewrite model:: {}'.format(model_name))
            if model_cls_str not in MODEL_DICT_REWRITE:
                raise ValueError(f'{model_cls_str} is not find in rerank dict')
            model_cls = MODEL_DICT_REWRITE[model_cls_str]
            model_batch_exec = self.model_batch_exec_rewrite
        else:
            raise ValueError(f'{model_type} is not supported')

        model = model_cls(
            batch_size=batch_size_limit,
            **build_kwargs,
        )
        batch_gather = BatchGather(model.run_gather, batch_size=batch_size_limit, max_capacity=queue_size_limit)
        batch_gather.start()
        atexit.register(batch_gather.stop)
        model_batch_exec[model_name] = model, batch_gather
        self.app.logger.info('load model success: {}'.format(model_name))

    def embed(self):
        try:
            self.valid_schema_embed(request.json)
        except ValidationError as err:
            err.sort()
            err_msg = 'valid error: {}'.format(err.format_error())
            self.app.logger.error('embed: {}'.format(err_msg))
            return Response(err_msg, status=400)

        model_name = request.json.get('model', 'default')
        if model_name not in self.model_batch_exec_embed:
            err_msg = 'model name [{}] not found'.format(model_name)
            self.app.logger.error('embed: {}'.format(err_msg))
            return Response(err_msg, status=400)
        model, batch_gather = self.model_batch_exec_embed[model_name]

        texts = request.json['text']
        return_list = isinstance(texts, list)
        if not return_list:
            texts = [texts]

        return_dense: bool = request.json.get('return_dense', True)
        return_sparse: bool = request.json.get('return_sparse', False)
        check_ret = model.can_encode(return_dense, return_sparse)
        if check_ret is not None:
            err_msg = 'param check error: {}'.format(check_ret)
            self.app.logger.error('embed: {}'.format(err_msg))
            model_counter.labels(model_type='embed', model_name=model_name, model_cls_str=type(model).__name__, status=400).inc()
            return Response(err_msg, status=400)

        with (
                model_time_elapse.labels(model_type='embed', model_name=model_name, model_cls_str=type(model).__name__).time(), 
                runner_timer(consts.REST_TIMEOUT, self.app.logger.error, ['time out in running embed!'])
            ):
            try:
                ret_list = batch_gather.put_task(texts).get_res()
                assert len(ret_list) == len(texts)
                ret_data = []
                for dense_embed, sparse_embed in ret_list:
                    ret = {}
                    if return_dense: ret['dense_embed'] = dense_embed
                    if return_sparse: ret['sparse_embed'] = sparse_embed
                    ret_data.append(ret)
                if not return_list:
                    ret_data = ret_data[0]

                model_counter.labels(model_type='embed', model_name=model_name, model_cls_str=type(model).__name__, status=200).inc()
                return jsonify(ret_data)
            except Exception as e:
                err_msg = 'run exception: {}'.format(e)
                self.app.logger.error('embed: {}'.format(err_msg))
                model_counter.labels(model_type='embed', model_name=model_name, model_cls_str=type(model).__name__, status=500).inc()
                return Response(err_msg, status=500)

    def rerank(self):
        try:
            self.valid_schema_rerank(request.json)
        except ValidationError as err:
            err.sort()
            err_msg = 'valid error: {}'.format(err.format_error())
            self.app.logger.error('rerank: {}'.format(err_msg))
            return Response(err_msg, status=400)

        model_name = request.json.get('model', 'default')
        if model_name not in self.model_batch_exec_rerank:
            err_msg = 'model name [{}] not found'.format(model_name)
            self.app.logger.error('rerank: {}'.format(err_msg))
            return Response(err_msg, status=400)
        model, batch_gather = self.model_batch_exec_rerank[model_name]

        query = request.json['query']
        texts = request.json['texts']
        text_pairs = [(query, text) for text in texts]

        with (
                model_time_elapse.labels(model_type='rerank', model_name=model_name, model_cls_str=type(model).__name__).time(), 
                runner_timer(consts.REST_TIMEOUT, self.app.logger.error, ['time out in running rerank!'])
            ):
            try:
                ret_list = batch_gather.put_task(text_pairs).get_res()
                model_counter.labels(model_type='rerank', model_name=model_name, model_cls_str=type(model).__name__, status=200).inc()
                return jsonify(ret_list)
            except Exception as e:
                err_msg = 'run exception: {}'.format(e)
                self.app.logger.error('rerank: {}'.format(err_msg))
                model_counter.labels(model_type='rerank', model_name=model_name, model_cls_str=type(model).__name__, status=500).inc()
                return Response(err_msg, status=500)


    def rewrite(self):
        try:
            self.valid_schema_rewrite(request.json)
        except ValidationError as err:
            err.sort()
            err_msg = 'valid error: {}'.format(err.format_error())
            self.app.logger.error('rewrite: {}'.format(err_msg))
            return Response(err_msg, status=400)

        model_name = request.json.get('model', 'default')
        model, batch_gather = self.model_batch_exec_rewrite[model_name]
        message = request.json['message']
        

        with (
                model_time_elapse.labels(model_type='rewrite', model_name=model_name, model_cls_str=type(model).__name__).time(), 
                runner_timer(consts.REST_TIMEOUT, self.app.logger.error, ['time out in running rewrite!'])
            ):
            try:
                ret_list = batch_gather.put_task(message).get_res()
                model_counter.labels(model_type='rewrite', model_name=model_name, model_cls_str=type(model).__name__, status=200).inc()
                print(ret_list)
                return jsonify(ret_list)
            except Exception as e:
                err_msg = 'run exception: {}'.format(e)
                self.app.logger.error('rewrite: {}'.format(err_msg))
                model_counter.labels(model_type='rewrite', model_name=model_name, model_cls_str=type(model).__name__, status=500).inc()
                return Response(err_msg, status=500)


    def run(self, host, port):
        try:
            self.app.run(host=host, port=port)
        finally:
            for _, exec in self.model_batch_exec_embed.values():
                exec.stop()
            for _, exec in self.model_batch_exec_rerank.values():
                exec.stop()
            for _, exec in self.model_batch_exec_rewrite.values():
                exec.stop()

def serve():
    server = InferServer()
    server.add_model(
        model_type =        consts.MODEL_TYPE,
        model_name =        consts.MODEL_NAME,
        model_cls_str =     consts.MODEL_CLS,
        batch_size_limit =  consts.BATCH_SIZE,
        queue_size_limit =  consts.QUEUE_SIZE,
        build_kwargs =      consts.MODEL_KWARGS,
    )
    server.app.logger.info(f'Infer Server started')
    server.run('0.0.0.0', consts.SERVICE_PORT)

def main():
    if consts.PROMETHEUS_SERVICE_PORT:
        start_prometheus_http_server(consts.PROMETHEUS_SERVICE_PORT)
    serve()

if __name__ == '__main__':
    main()
