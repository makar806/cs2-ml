"""Nonblocking movement-release settling gate for standard WASD controls."""
class StopBeforeShot:
    def __init__(self, settle=.18):
        self.settle=settle
        self.until=0.
    def ready(self, now, held, release):
        if held:
            release(held)
            self.until=now+self.settle
            return False
        return now>=self.until
