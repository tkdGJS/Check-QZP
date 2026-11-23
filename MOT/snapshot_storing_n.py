from torch import save
from multiprocessing import Pool, Process, set_start_method, Manager, Value, Lock, Queue
import os
PID = os.getpid()
print('subprocess:', PID)
#print(mp.set_start_method)
#OUTPUT_PATH = './fio_test/covid-checkpoint/'
#os.sync()
class async_checkpointing:

    def __init__(self, model_state, op_state, epoch, PATH, shared_value=None, lock=None):
        self.model_state = model_state
        self.epoch = epoch
        self.op_state = op_state
        self.PATH = PATH
        self.shared_value = shared_value
        self.lock = lock

    def snapshot_storing(self, model_state, op_state, epoch, PATH):
        shared_value = self.shared_value
        lock = self.lock
        
        model_state = _to_cpu(self.model_state)
        op_state = _to_cpu(self.op_state)

        if lock is not None:
            with lock:
                shared_value = Value('i', 0)

#        for i, j in model_state.items():
#            model_state[i] = j.cpu()
#        for keys in model_state.keys():
#            model_state[keys] = model_state[keys].cpu()
        
        OUTPUT_PATH = PATH

        save({
            'model_state_dict': model_state,
            'optimizer_state_dict' : op_state
        }, OUTPUT_PATH)

        #PID_w = os.system("ls -lh /proc/"+str(PID)+"/fd")
        PID_w = os.system("cat /proc/"+str(PID)+"/io")


def _to_cpu(ele, snapshot=None):
    #while True:
    if snapshot is None:
        snapshot = {}
    if hasattr(ele, 'cpu'):
        snapshot = ele.cpu()
    elif isinstance(ele, dict):
        snapshot = {}
        for k,v in ele.items():
            snapshot[k] = None
            snapshot[k] = _to_cpu(v, snapshot[k])
    elif isinstance(ele, list):
        snapshot  = [None for _ in range(len(ele))]
        for idx, v in enumerate(ele):
            snapshot[idx] = _to_cpu(v, snapshot[idx])

    return snapshot     
