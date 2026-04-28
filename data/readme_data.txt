Dataset: NASA C-MAPSS Turbofan Engine Degradation Simulation Data, FD001 subset.

Files expected in this folder:
- train_FD001.txt: training trajectories run to failure
- test_FD001.txt: partial test trajectories
- RUL_FD001.txt: true remaining useful life labels for the final cycle of each test engine

Data format:
Each row contains engine_id, cycle, 3 operational settings, and 21 sensor readings. The files are whitespace-delimited with no header row.

How to obtain the data:
The FD001 files are commonly distributed as part of the NASA Prognostics Center of Excellence C-MAPSS turbofan degradation dataset. Download the C-MAPSS dataset from the NASA prognostics data repository, extract the FD001 files, and place train_FD001.txt, test_FD001.txt, and RUL_FD001.txt in this data folder.

For this submission, the three required FD001 files are included in the data folder so the code runs immediately after unzipping.
