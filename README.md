# CC mlcourses

### Introduction.
Learning Course in CC
### Setup. 
#### System requirements
* Miniconda :  https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html   
* To see which packages are installed in your current conda environment and their version numbers, in your terminal window or an Anaconda Prompt, run conda list.
```
/phuongdo/mlcourses (miniconda3)  conda list
# packages in environment at /Users/phuongdv/miniconda3:
#
# Name                    Version                   Build  Channel
brotlipy                  0.7.0           py39h9ed2024_1003
ca-certificates           2020.12.8            hecd8cb5_0
```

#### Create python env

```bash
conda env create -f environment.yml --name devEnv 
# conda env update -f environment.yml --name devEnv ( if you want to update your env)
conda activate devEnv 

(devEnv)$

```

#### How to use Jupyterlab (Notebooks)
_Note_: not required in this demo.)
```bash
$ conda install -c conda-forge jupyterlab
```
* https://jupyterlab.readthedocs.io/en/stable/getting_started/installation.html

Add your environment to Jupyterlab. 

```bash
conda activate devEnv 

(devEnv)$ conda install -c anaconda ipykernel

(devEnv)$ python -m ipykernel install --user --name=devEnv

```
Just check your Jupyter Notebook, to see the shining devEnv.



#### How run python script in this sample sources?

```bash
(devEnv)$ source bin/setvars.sh                                                                                                                                                                                              1 ↵
(devEnv)$  python course02/src/sample/pipeline/run_pipeline.py   

```




