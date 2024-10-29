import numpy as np
import os
import time

from src.hamiltonian_updates import hamiltonian_update
from src.rounding import randomized_rounding
from src.hamiltonian_updates import gibbs_state
from src.hu_binary_search import binary_search_HU


def compute_batch(load_dir, save_path, epsilon, hu_kwargs, offset=0):
    C_til_list = []
    opt_list = []
    result_list = []
    print('Loading files...')
    for filename in os.listdir(f'{load_dir}/'):
        data = np.load(f'{load_dir}/{filename}', allow_pickle=True).item()
        C_til_list.append(data['C_til'])
        opt_list.append(data['opt'])

    for i in range(len(C_til_list)):
        print(f'n={C_til_list[i].shape[0]}, compute {i + 1}/{len(C_til_list)}')
        start = time.time()
        try:
            feasible, H, t, delta_entropy, qc, logs = hamiltonian_update(
                C_til=C_til_list[i],
                gamma=opt_list[i] + offset,
                epsilon=epsilon,
                **hu_kwargs
            )
            result_list.append({'feasible': feasible, 'iters': t, 'delta_entropy': delta_entropy, 'qc': qc,
                                'logs': logs, 'time': time.time() - start})
        except Exception as E:
            with open('log.txt', 'a') as file:
                file.write(f'Exception: "{E}" encountered for {load_dir}/{os.listdir(f"{load_dir}/")[i]} '
                           f'with {epsilon = }, {offset = }, {hu_kwargs = } \n')

    context = {'epsilon': epsilon, 'hu_kwargs': hu_kwargs, 'offset': offset}
    np.save(f'{save_path}.npy', {'results': result_list, 'context': context})


def compute_batch_bin(load_dir, save_path, epsilon, hu_kwargs,
                      gap=None, weight_ratio=1, gamma_min=-1, gamma_max=1):
    if gap is None:
        gap = epsilon

    C_til_list = []
    result_list = []
    print('Loading files...')
    for filename in os.listdir(f'{load_dir}/'):
        data = np.load(f'{load_dir}/{filename}', allow_pickle=True).item()
        C_til_list.append(data['C_til'])

    for i in range(len(C_til_list)):
        print(f'n={C_til_list[i].shape[0]}, compute {i + 1}/{len(C_til_list)}')
        start = time.time()

        H, gamma, bin_counter, bin_accepted, bin_declined, iters_total = binary_search_HU(
            C_til=C_til_list[i],
            epsilon=epsilon,
            gap=gap,
            weight_ratio=weight_ratio,
            gamma_min=gamma_min,
            gamma_max=gamma_max,
            hu_kwargs=hu_kwargs
        )

        result_list.append({
            'iters': iters_total, 'bin_counter': bin_counter, 'bin_accepted': bin_accepted,
            'bin_declined': bin_declined, 'time': time.time() - start,
        })

    context = {'epsilon': epsilon, 'hu_kwargs': hu_kwargs}
    np.save(f'{save_path}.npy', {'results': result_list, 'context': context})


def add_optimal_rounds(load_dir, save_dir, shots, number_batches):
    for filename in sorted(os.listdir(load_dir)):
        data: dict = np.load(f'{load_dir}/{filename}', allow_pickle=True).item()

        opt_round_res = randomized_rounding(
            data['C_til'], data['rho_opt'], shots, number_batches=number_batches
        )

        data['optimal_rounding'] = opt_round_res
        np.save(f'{save_dir}/{filename}', np.array(data, dtype='object'))


def hu_rounds(load_dir, save_path, log_eps, shots, number_batches, hu_kwargs):
    epsilon = 10 ** (-log_eps)
    result_list = []

    for filename in sorted(os.listdir(load_dir)):
        data: dict = np.load(f'{load_dir}/{filename}', allow_pickle=True).item()
        hu_kwargs['s'] = data['context']['s']

        start = time.time()

        _, H, t, delta_entropy, qc, logs = hamiltonian_update(
            C_til=data['C_til'],
            gamma=data['opt'],
            epsilon=epsilon,
            **hu_kwargs
        )
        data['time'] = time.time() - start
        data['iters'] = t

        hu_rounding = randomized_rounding(data['C_til'], gibbs_state(H)[0], shots, number_batches=number_batches)

        result_list.append({
            'iters': t, 'delta_entropy': delta_entropy, 'qc': qc,
            'logs': logs, 'time': time.time() - start,
            'optimal_rounding': data['optimal_rounding'], 'hu_rounding': hu_rounding
        })

    context = {'epsilon': epsilon, 'hu_kwargs': hu_kwargs, 'shots': shots, 'number_batches': number_batches}
    np.save(f'{save_path}.npy', {'results': result_list, 'context': context})


def binary_search_HU_round(load_dir, save_path, log_eps, shots, number_batches,
                           gap=None, weight_ratio=1, gamma_min=-1, gamma_max=1, hu_kwargs=None):
    if hu_kwargs is None:
        hu_kwargs = dict()

    epsilon = 10 ** (-log_eps)
    if gap is None:
        gap = epsilon

    result_list = []

    for filename in sorted(os.listdir(load_dir)):
        data: dict = np.load(f'{load_dir}/{filename}', allow_pickle=True).item()

        start = time.time()

        H, gamma, bin_counter, bin_accepted, bin_declined, iters_total = binary_search_HU(
            C_til=data['C_til'],
            epsilon=epsilon,
            gap=gap,
            weight_ratio=weight_ratio,
            gamma_min=gamma_min,
            gamma_max=gamma_max,
            hu_kwargs=hu_kwargs)

        hu_rounding = randomized_rounding(data['C_til'], gibbs_state(H)[0], shots, number_batches=number_batches)
        result_list.append({
            'iters': iters_total, 'bin_counter': bin_counter, 'bin_accepted': bin_accepted,
            'bin_declined': bin_declined, 'time': time.time() - start,
            'optimal_rounding': data['optimal_rounding'], 'hu_rounding': hu_rounding
        })

    context = {'epsilon': epsilon, 'hu_kwargs': hu_kwargs, 'shots': shots, 'number_batches': number_batches,
               'gap': gap, 'weight_ratio': weight_ratio, 'gamma_min': gamma_min, 'gamma_max': gamma_max}
    np.save(f'{save_path}.npy', {'results': result_list, 'context': context})
