
The data in this project can be (randomly) generated by running the following lines (depending on the python installation `python3` needs to be replaced by `python`). This will likely take days. By changing the `--number` flag, the number of generated samples can be adjusted. 
```
python3 -m scripts.gen_cost_matrices_benchmarking --number 256;
python3 -m scripts.hu_benchmark_qc;

python3 -m scripts.gen_cost_matrices_scaling --number 20 --n 1024 --s 16;
python3 -m scripts.hu_beta_scaling --epsilon 0.001;
python3 -m scripts.hu_epsilon_scaling;
python3 -m scripts.hu_gamma_scaling --epsilon 0.001;

python3 -m scripts.gen_cost_matrices_comparison --number 20 --n 128 --s 16;
python3 -m scripts.hu_comparison;
```

The data can by evaluated by running the following lines to determine fit functions and create figures:
```
python3 -m fig.benchmark;
python3 -m fig.beta_scaling;
python3 -m fig.epsilon_scaling;
python3 -m fig.gamma_scaling;
python3 -m fig.comparison --n 128;
```
