
import time, math

class Timer():
    def __init__(self):
        self.start = None
        self.reset()
    def reset(self):
        self.start= time.time()
    def __call__(self, percent):
        now = time.time()
        s = now - self.start
        es = s / (percent)
        rs = es - s
        return '%s (- %s)' % (self.asMinutes(s), self.asMinutes(rs))
    def asMinutes(self, s):
        m = math.floor(s / 60)
        s -= m * 60
        return '%dm %ds' % (m, s)

if __name__ == '__main__':
    timer = Timer()

