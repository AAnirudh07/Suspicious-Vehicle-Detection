# suspicious-vehicle-detection
This repository contains the source code for the suspicious vehicle detection system built for the Tamil Nadu state police. The system returns a list of suspicious vehicles based on the time taken for them for them to cross two consecutive toll gates. It consists of the following stages:

## Stage 1: Object Tracking using YOLOv5 + DeepSORT

To execute the code for stage, run the file track.py with the video path given as a command line argument. The output will be stored in the folder ./run/tracks.   


## Stage 2: Suspicious Vehicle Identification
The results of stage 1 are stored in a CSV file. It has the following headers:

        1. Frame number
        2. Vehicle ID
        3. Car type
        4. Car color

Two such CSV files are generated (one for each video). Each row in the source file is checked to see if a vehicle with the same specifications os present in the second file using a range search. If no such entry is found, then the vehicle is labeled suspicious. After all the rows of the first CSV file are checked, the system returns the list of suspicious vehicles and their pictures. 
