from src.numerics import add_optimal_rounds
from src.numerics import binary_search_HU_round

import argparse
import numpy as np
import os

if __name__ == '__main__':
    hu_kwargs = {
        'b': 8,
        'lambda_c': 10,
        'lambda_d': 10,
        'beta': 0.45,
        'lambda_increase': 1.3,
    }

    CLI = argparse.ArgumentParser()
    CLI.add_argument(
        "--shots",
        type=int,
        default=100_000,
    )
    CLI.add_argument(
        "--num_batches",
        type=int,
        default=100,
    )
    CLI.add_argument(
        "--log_eps_min",
        type=float,
        default=2.0,
    )
    CLI.add_argument(
        "--log_eps_max",
        type=float,
        default=3.400001,
    )
    CLI.add_argument(
        "--log_eps_step",
        type=float,
        default=0.2,
    )
    CLI.add_argument(
        "--skip_optimal",
        action="store_true"
    )

    args = CLI.parse_args()
    shots = args.shots
    number_batches = args.num_batches

    log_eps_range = np.arange(args.log_eps_min, args.log_eps_max, args.log_eps_step)

    if not args.skip_optimal:
        print(f'adding optimal rounds')
        os.makedirs('data/cost_matrices/opt_rounds/sparse', exist_ok=True)
        add_optimal_rounds(
            load_dir='data/cost_matrices/scaling/sparse',
            save_dir='data/cost_matrices/opt_rounds/sparse',
            shots=shots,
            number_batches=number_batches
        )

    print('Performing HU binary search')
    os.makedirs('data/results/epsilon/sparse', exist_ok=True)
    for log_eps in log_eps_range:
        print(f'{log_eps = }')
        binary_search_HU_round(
            load_dir='data/cost_matrices/opt_rounds/sparse',
            save_path=f'data/results/epsilon/sparse/log_eps_{round(log_eps, 4)}',
            log_eps=log_eps,
            gap=None,
            shots=shots,
            number_batches=number_batches,
            hu_kwargs=hu_kwargs
        )
