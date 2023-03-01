
from dataclasses import dataclass
import icecream as ic

from config import Config
from processing import register_maps, binning
from testing import Clock, Profiler

def main():

    # Step 1: Read FITS files, geometric registration, save output files
    dataset =  register_maps()

    more_stuff = binning()

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
