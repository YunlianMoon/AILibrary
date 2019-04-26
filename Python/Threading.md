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
added_thread = threading.Thread(target=, name=)
added_thread.start()
```
