# unitary-nn-design
Unitary Neural Network Design

## Running
Install dependencies:
```shell
apt install graphviz-dev
pip install pygraphviz gym torch==0.4.0
```

Install the PyTorch extension for the primary network:
```shell
cd dagnn
python setup.py install
cd ..
```

You may need to run with an OpenMP flag:
```shell
# get argparse help
OMP_NUM_THREADS=1 python main.py -h
```
