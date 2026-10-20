# data/

`groundlink_metadata.csv`: GroundLink subject mass and height, from the public dataset. Read by `GroundLinkAdapter`
(`src/data/dataset_adapters.py`).

The raw captures are not in this repository. Point the code at them with `GROUNDLINK_ROOT` and `TAMU_DATA_ROOT`, set in
the shell (they are read at import time, so `.env` cannot set them); see the top-level `README.md`. Patient Handling
participant metadata (`Experiment Detail.xlsx`) is read from `TAMU_DATA_ROOT/01.PatientHandlingTasks/` and is not
distributed.
