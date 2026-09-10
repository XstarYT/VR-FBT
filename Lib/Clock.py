import ctypes
import time

winmm = ctypes.WinDLL('winmm')

class delta:
	class Clock:
		delta = 0
		def __enter__(self):
			self.d1 = time.perf_counter()
			return self

		def __exit__(self, *arg, **kwarg):
			d2 = time.perf_counter()
			self.delta = max(d2-self.d1, 0)

	class precClock:
		delta = 0
		def __enter__(self):
			winmm.timeBeginPeriod(1)
			self.d1 = time.perf_counter()
			return self

		def __exit__(self, *arg, **kwarg):
			d2 = time.perf_counter()
			winmm.timeEndPeriod(1)
			self.delta = max(d2-self.d1, 0)

def sleep(FPS, Delta):
	time.sleep(max(FPS-Delta, 0))
