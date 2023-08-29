from typing import Tuple

from giantpipe import Clock, Profiler

def start_monitor() -> Tuple[object, object]:
    profiler = Profiler()
    profiler.start()
    # Start clock
    clock = Clock()
    clock.start()
    return profiler, clock

def stop_monitor(profiler: object, clock: object):
    # Stop clock
    clock.stop()
    clock.elapsed_time()
    # Stop monitoring code and 
    profiler.stop()
    profiler.save_profile_report()