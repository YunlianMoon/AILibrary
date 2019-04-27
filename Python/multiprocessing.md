# Multiprocessing

(1) reference
```python
import multiprocessing
```
(2) function
```python
# add process and start
added_process = multiprocessing.Process(target=, name=, args=)
added_process.start()
```
(3) join
```python
# wait process finish
added_process = multiprocessing.Process(target=, name=, args=)
added_process.start()
added_process.join()
```
(4) lock
```python
lock = multiprocessing.Lock()
lock.acquire()
lock.release()
```
(5) pool
```python
pool = multiprocessing.Pool(processes=)
pool.map()
pool.apply_async()
```
(5) shared memory
```python
multiprocessing.Value()
multiprocessing.Array()
```

