import numpy as np
import cProfile
import io
import pstats
import time

from config import Config

class Clock:
    """Responsible for timing the execution of the code."""
    
    def __init__(self) -> None:
        self.clock: bool = Config.clock
        self.start: float
        self.stop: float

    def start(self) -> None:
        """Start the clock."""
        if not self.clock:
            return
        if self.clock:
            self.start = time.time()

    def stop(self) -> None:
        """Stop the clock."""
        if not self.clock:
            return
        if self.clock:
            self.stop = time.time()
    
    def elapsed_time(self) -> None:
        """Calculate the total time taken for code execution."""
        if not self.clock:
            return
        if self.clock:
            difference = self.stop - self.start
            print(f"Elapsed time: {np.round(difference, 3)} s")

class Profiler:
    """Responsible for creating a profile containing a set of statistics 
    that describes the execution frequency and duration of various parts of the program."""
    
    def __init__(self):
        self.profiler: bool = Config.profiler

    def start(self) -> None:
        """Start the profiler."""
        if not self.profiler:
            return
        if self.profiler:
            self.profile = cProfile.Profile()
            self.profile.enable()

    def stop(self) -> None:
        """Stop the profiler."""
        if not self.profiler:
            return
        if self.profiler:
            self.profile.disable()
            return
    
    def save_profile_report(self) -> None:
        """Save report of the performance of code execution."""
        if not self.profiler:
            return
        if self.profiler:
            s = io.StringIO()
            ps = pstats.Stats(self.profile, stream=s).sort_stats('tottime')
            ps.strip_dirs()
            ps.print_stats()
            with open('./cProfiler_output.txt', 'w+') as f:
                f.write(s.getvalue())