import gzip
import os
import pickle

mem = {'key' : [1,2,3,4,5,6],
        'key2' : [1.2,1.3,1.4,1.7]}
bmem = pickle.dumps(mem)
print(mem)

with gzip.open('./file.txt.gz','wb', compresslevel=9) as f:
    f.write(bmem)


