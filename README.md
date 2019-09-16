# Damn Dams
Canal blocking computation for Winrock.

## Installation
1. Create a new environment within Conda with minimal packages:

```
conda create -n [name of environment] -c conda-forge python=2.7 numpy scipy rasterio fipy matplotlib pandas
```

Optional packages:
  - ``` spyder```: IDE for Python. Can be added to the list above.
  - ``` deap ```: For Genetic Algorithm implementation. Can be added to the list above.
  - ``` simanneal ```: For Simulated Annealing implementation. Cannot be added to the list above and must be installed separately. Instructions [here](https://github.com/perrygeo/simanneal).

2. Don't forget to activate the new environment!

3. Clone or download this repository.

## How to use?
There are 3 alternative ways to run the code:
  - ```main.py``` runs the general computation without any optimization algorithm. It still allows for more than one iteration.
  - ```ga_try_2.py``` runs the Genetic Algorithm version.
  - ```simanneal_try.py```runs the Simulated Annealing version.

The code needs the following data to run:
  - A DEM as a raster image.
  - A canal network as a raster image.
  - A soil type map as a raster image.
  - (A soil depth map as a raster image.) UNDER DEV.

The data must be stored in the path "data/" relative to the ```.py``` files.
