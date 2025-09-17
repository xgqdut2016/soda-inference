import queue
from typing import Callable
from threading import Thread, Condition

class SingleTask:
    def __init__(self, info: list) -> None:
        self.cond = Condition()
        self.info = info # input
        self.exp = None # exception
        self.res = None # output

    def put_res(self, res):
        self.res = res
        with self.cond:
            self.cond.notify()

    def put_exp(self, exp):
        self.exp = exp
        with self.cond:
            self.cond.notify()

    def get_res(self):
        with self.cond:
            self.cond.wait()
            if self.exp:
                raise self.exp
            else:
                return self.res

class BatchGather:
    def __init__(self, batch_func: Callable[[list], list], batch_size, max_capacity=0) -> None:
        self.task_queue = queue.Queue(max_capacity)
        self.batch_func = batch_func
        self.batch_size = batch_size
        self.loop_finish = True
        self.thread = None

    def put_task(self, info):
        task = SingleTask(info)
        self.task_queue.put(task)
        return task

    def start(self):
        self.loop_finish = False
        self.thread = Thread(target=self._loop)
        self.thread.start()

    def stop(self):
        if self.loop_finish == False:
            print('stop batch gather')
            self.loop_finish = True
            self.task_queue.put(SingleTask([]))
            self.thread.join()
            self.thread = None

    def _get_task_batch(self):
        self.last_task: SingleTask | None
        inputs = []
        tasks: list[SingleTask] = []
        if self.last_task is None:
            try:
                self.last_task: SingleTask = self.task_queue.get(False)
            except queue.Empty:
                # 阻塞到下一次
                self.last_task: SingleTask = self.task_queue.get(True)
        # do-while 形式，第一次的 task 一定加入列表
        while True:
            tasks.append(self.last_task)
            inputs.extend(self.last_task.info)
            try:
                self.last_task: SingleTask = self.task_queue.get(False)
            except queue.Empty:
                self.last_task = None
                break
            if len(inputs) + len(self.last_task.info) > self.batch_size:
                break
        return inputs, tasks

    def _loop(self):
        self.last_task = None
        while True:
            inputs, tasks = self._get_task_batch()

            if self.loop_finish:
                return

            try:
                result_batch = self.batch_func(inputs)
                # Convert numpy array or other iterable to list if needed
                if hasattr(result_batch, 'tolist'):
                    result_batch = result_batch.tolist()
                elif not isinstance(result_batch, list):
                    result_batch = list(result_batch)

                assert isinstance(result_batch, list), 'batch gather result must be a list, now its [{}]'.format(type(result_batch))
                assert len(result_batch) == len(inputs), 'batch gather result length must equal to input [{}], now its [{}]'.format(len(inputs), len(result_batch))
                for item in tasks:
                    item_res, result_batch = result_batch[:len(item.info)], result_batch[len(item.info):]
                    item.put_res(item_res)
            except Exception as e:
                for item in tasks:
                    item.put_exp(e)

            if self.loop_finish:
                return
