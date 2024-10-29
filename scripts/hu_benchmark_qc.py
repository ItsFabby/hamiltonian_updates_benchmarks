from src.numerics import compute_batch_bin

import os

if __name__ == '__main__':
    epsilon = 0.01
    hu_kwargs_sparse = {
        'b': 8,
        'lambda_c': 10,
        'lambda_d': 10,
        'beta': 0.5,
        'lambda_increase': 1.3,
    }

    os.makedirs('data/results', exist_ok=True)

    compute_batch_bin(
        load_dir='data/cost_matrices/benchmarking/sparse',
        save_path='data/results/benchmarking_sparse',
        epsilon=epsilon,
        gap=epsilon,
        hu_kwargs=hu_kwargs_sparse
    )
