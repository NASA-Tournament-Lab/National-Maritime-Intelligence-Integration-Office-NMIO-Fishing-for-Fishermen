#!/usr/bin/env bash
mkdir input/TrackDistances
cd scripts
python gen_distances.py
python gen_speed.py
python get_predictions.py
