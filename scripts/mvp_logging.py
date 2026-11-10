"""Bounded diagnostic output, isolated from the control and capture threads."""
from queue import Empty, Full, Queue
import threading
import time


class AsyncLog:
    def __init__(self, path, capacity=4096, opener=None):
        self.path=path
        self.queue=Queue(capacity)
        self.stop=threading.Event()
        self.dropped=0
        self.error=''
        self.closed=False
        self.opener=opener or (lambda:path.open('w',encoding='utf-8'))
        self.thread=threading.Thread(target=self._run,daemon=True)
        self.thread.start()

    def write(self, text):
        if self.closed or self.error:
            self.dropped+=1
            return 0
        try:
            self.queue.put_nowait(text)
            return len(text)
        except Full:
            self.dropped+=1
            return 0

    def flush(self):
        # The writer flushes on its own schedule; callers never wait on disk.
        pass

    def _run(self):
        try:
            with self.opener() as output:
                flushed=time.monotonic()
                while not self.stop.is_set() or not self.queue.empty():
                    try:
                        output.write(self.queue.get(timeout=.05))
                    except Empty:
                        pass
                    if time.monotonic()-flushed>=.25:
                        output.flush()
                        flushed=time.monotonic()
        except Exception as error:
            self.error=f'{type(error).__name__}: log write failed'
            self.dropped+=self.queue.qsize()

    def close(self):
        self.closed=True
        self.stop.set()
        self.thread.join(timeout=2)
        if self.thread.is_alive():
            self.error='Log writer did not finish in 2 seconds; output may be incomplete'

    def stats(self):
        return dict(dropped=self.dropped,pending=self.queue.qsize(),error=self.error)


def open_log(path, asynchronous=True):
    return AsyncLog(path) if asynchronous else path.open('w',encoding='utf-8')
