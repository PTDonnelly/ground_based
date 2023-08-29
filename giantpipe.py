from giantpipe import Dataset
from giantpipe import scripts

def main():
    """Run the giantpipe VISIR Data Reduction Pipeline."""
    
    # Start monitoring code
    profiler, clock = scripts.start_monitor()

    # For calibration of the VISIR data
    Dataset.run_calibration()

    # For plotting the VISIR data
    Dataset.run_plotting()

    # For generating spectral files (.spx) from the VISIR data
    Dataset.run_spxing()

    completion = "\ngiantpipe is finished, have a great day."
    print(completion)

    scripts.stop_monitor(profiler, clock)

if __name__ == '__main__':
    main()