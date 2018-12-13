# cs393r-2d-asami

This repository contains the source code for running the EM algorithm for 2D ASAMI. `em.py` is the main file that runs 

Requirements
* Python 3.6
* filterpy (`pip install filterpy`)

The behavior for running the data collection can be found at https://github.com/srama2512/cs393r-latest/blob/asami_project/core/python/behaviors/sample_2d.py . This will output `2d_asami_data.txt`. 

Running instructions:
```
$ python3 em.py --data 2d_asami_data.txt --n_iter <iters> --seed <seed> \
  --n_plot <plot_every_n_plot_iters> --save_path <save_dir> --display no
```
