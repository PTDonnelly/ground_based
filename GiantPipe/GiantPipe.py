
from dataclasses import dataclass
import icecream as ic

from processing import register_maps
from testing import Clock, Profiler

def main():

    # Step 1: Read FITS files, geometric registration, save output files
    register_maps()

if __name__ == '__main__':
        
    # Start monitoring code
    profiler = Profiler()
    profiler.start()
    # Start clock
    clock = Clock()
    clock.start()
    # Execute module-level code
    main()
    # Stop clock
    clock.stop()
    clock.elapsed_time()
    # Stop monitoring code and 
    profiler.stop()
    profiler.save_profile_report()
