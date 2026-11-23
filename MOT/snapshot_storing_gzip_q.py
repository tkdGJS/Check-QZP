from torch.multiprocessing import Pool, Process, set_start_method, Manager, Value, Lock
import os
import Uniform as U
import time
import gzip
import pickle

PID = os.getpid()
print('subprocess:', PID)
#print(mp.set_start_method)
#OUTPUT_PATH = './fio_test/covid-checkpoint/'

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
        snapshot_start=time.time()

        model_state = _to_cpu(self.model_state)
        op_state = _to_cpu(self.op_state)
        snapshot_end=time.time()
#        print(len(op_state['state']))
#        assert False
#        for i, j in model_state.items():
#            model_state[i] = j.cpu()
#        for keys in model_state.keys():
#            model_state[keys] = model_state[keys].cpu()
        if lock is not None:
            with lock:
                shared_value = Value('i', 0)
        print("snapshot time : ", snapshot_end-snapshot_start)

        q_start = time.time()
        Uniform1 = U.UniformModel(model_state, 8)
        model_state = Uniform1.Quantization(model_state, 8)
        Uniform2 = U.UniformOptimizer(op_state, 8)
        optimizer_state = Uniform2.Quantization(op_state, 8)
        q_end = time.time()
        print("quantization time : ", q_end-q_start)
        save_start = time.time()
        OUTPUT_PATH = PATH
        storing = {
            'model_state_dict': model_state,
            'optimizer_state_dict' : optimizer_state
        }
        binary_s = pickle.dumps(storing)

        with gzip.open(OUTPUT_PATH,'wb', compresslevel=9) as f:
            f.write(binary_s)
        
        save_end = time.time()
        print("save time : ", save_end-save_start)
        PID_w = os.system("cat /proc/"+str(PID)+"/io")
        return 0


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
