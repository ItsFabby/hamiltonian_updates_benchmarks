from src.cost_matrices import sample_uniform_dim
import os
import argparse

if __name__ == '__main__':
    CLI = argparse.ArgumentParser()
    CLI.add_argument(
        "--number",
        type=int,
        default=256,
    )
    CLI.add_argument(
        "--n_min",
        type=int,
        default=512,
    )
    CLI.add_argument(
        "--n_max",
        type=int,
        default=4096,
    )
    CLI.add_argument(
        "--s",
        type=int,
        default=16,
    )
    args = CLI.parse_args()
    number = args.number
    n_min = args.n_min
    n_max = args.n_max
    s = args.s

    if n_max <= n_min:
        raise Exception(f'{n_max = } needs to be larger than {n_min = }.')

    save_dir = 'data/cost_matrices/benchmarking/sparse'
    os.makedirs(save_dir, exist_ok=True)
    print(f'Generating {number} sparse cost {"matrices" if number>1 else "matrix"} for benchmarking '
          f'with dimensions between {n_min} and {n_max}, {s = }.')

    sample_uniform_dim(
        n_min=n_min,
        n_max=n_max,
        s=s,
        number=number,
        save_dir=save_dir,
        solve_SDP=False,
        solver_eps=None,
    )
