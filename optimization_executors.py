from abc import abstractmethod, ABCMeta
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from multiprocessing.pool import ApplyResult
import pickle
import uuid
from pathos.multiprocessing import ProcessingPool
import pika 

from optimization_tools.optimizers.abstract_optimizer import AbstractOPtimizer
from . import opt_tools_settings

def set_local_directions_and_run_single_optimization(optimizer: AbstractOPtimizer):
    """
    Для запуска через функции, требующие сериализации
    """
    optimizer.optimized_object.solver.set_working_dir(optimizer.optimized_object.unique_id)
    optimizer.optimized_object.solver.initialize_log(optimizer.optimized_object.unique_id)
    result = optimizer.run_optimization()
    return result

def run_single_optimization(optimizer: AbstractOPtimizer):
    result = optimizer.run_optimization()
    return result

class panel_optimization_client(object):
    def __init__(self):
        credentials = pika.PlainCredentials('user', 'password')
        parameters = pika.ConnectionParameters(opt_tools_settings.RPC_Q_IP,
                                        opt_tools_settings.RPC_Q_PORT,
                                        '/',
                                        credentials,
                                        heartbeat=600,
                                        blocked_connection_timeout=600)
        self.connection = pika.BlockingConnection(parameters)

        self.channel = self.connection.channel()

        result = self.channel.queue_declare(queue='', exclusive=True)
        self.callback_queue = result.method.queue

        self.channel.basic_consume(
            queue=self.callback_queue,
            on_message_callback=self.on_response,
            auto_ack=True)

        self.response = None
        self.corr_id = None

    def on_response(self, ch, method, props, body):
        if self.corr_id == props.correlation_id:
            self.response = body

    def call(self, _js_str: str):
        self.response = None
        self.corr_id = str(uuid.uuid4())
        self.channel.basic_publish(
            exchange='',
            routing_key='rpc_queue',
            properties=pika.BasicProperties(
                reply_to=self.callback_queue,
                correlation_id=self.corr_id,
            ),
            body=_js_str)
        while self.response is None:
            self.connection.process_data_events(time_limit=None)
        return self.response

def run_single_optimization_on_cluster(optimizer: AbstractOPtimizer):
            client = panel_optimization_client()
            task_byte_view = pickle.dumps(optimizer, pickle.HIGHEST_PROTOCOL)
            response = client.call(task_byte_view)
            ready_task = pickle.loads(response)
            return ready_task


class AbstractExecutor(metaclass=ABCMeta):
    @abstractmethod
    def __call__(self, tasks):
        pass


class ForLoopExecutor(AbstractExecutor):
    def __init__(self) -> None:
        self.function = run_single_optimization

    def __call__(self, tasks):
        return list(map(self.function, tasks))
    

# https://stackoverflow.com/questions/19984152/what-can-multiprocessing-and-dill-do-together
class MultiprocessExecutor(AbstractExecutor):
    def __init__(self, pool) -> None:
        # self.num_proc = num_proc
        self.pool: ProcessingPool = pool
        self.function = run_single_optimization
        
    def __call__(self, tasks):
        future_results: list[ApplyResult] = []
        for task in tasks:
            future_result = self.pool.apipe(self.function, task)
            future_results.append(future_result)
        calculated = []
        for result in future_results:
            calculated.append(result.get())
        return calculated


class MultiprocessExecutorCF(AbstractExecutor):
    def __init__(self, pool) -> None:
        # self.num_proc = num_proc
        self.pool: ProcessPoolExecutor = pool
        self.function = run_single_optimization
    
    def __call__(self, tasks):
        future_results: list[Future] = []
        for task in tasks:
            future_result = self.pool.submit(self.function, task)
            future_results.append(future_result)
        calculated = []
        for result in future_results:
            calculated.append(result.result())
        return calculated



class RabbitExecutor(AbstractExecutor):
    def __init__(self, pool) -> None:
        self.pool: ThreadPoolExecutor = pool
        self.function = run_single_optimization

    def __call__(self, tasks: list[AbstractOPtimizer]):
        future_results: list[Future] = []
        for task in tasks:
            future_result = self.pool.submit(run_single_optimization_on_cluster, task)
            future_results.append(future_result)
        
        calculated = []
        for result in future_results:
            calculated.append(result.result())

        return calculated
