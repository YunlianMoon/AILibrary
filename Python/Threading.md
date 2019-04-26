# Threading

(1) reference
```python
import threading
```
(2) function
```python
# how many activated threads
threading.active_count()

# all threads information
threading.enumerate()

# current running threads
threading.current_thread()

# add thread and start
added_thread = threading.Thread(target=, name=, args=)
added_thread.start()
```
(3) join
```python
# wait thread finish
added_thread = threading.Thread(target=, name=, args=)
added_thread.start()
added_thread.join()
```
(4) lock
lock=threading.Lock()
lock.acquire()
lock.release()
