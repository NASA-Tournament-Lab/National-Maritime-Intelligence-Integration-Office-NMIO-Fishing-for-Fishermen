To reproduce the result:

1. Copy VesselTracks/ folder here
2. Compile each C++ script with g++ -std=gnu++11 -O3
3. Execute sequentially:
        binarize
        run train
        run rf
        run test > output.csv
