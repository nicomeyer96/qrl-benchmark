# If used in your project please cite this work as described in the README file.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Use cluster resampling to compute percentiles of empirical sampling complexities.
"""

import os
import numpy as np

from helper import extract
from sampling_complexity import sampling_complexity


def cluster_bootstrapping(data, epsilon, delta, samples_per_epoch, max_epochs, random_clusters, percentile):
    # number of samples in cluster
    cluster_size = data.shape[0]

    # run cluster bootstrapping to estimate percentiles
    sampling_complexities = []
    for _ in range(random_clusters):
        data_resampled = data[np.random.randint(cluster_size, size=cluster_size)]
        sampling_complexities.append(sampling_complexity(data_resampled, epsilon, delta, samples_per_epoch, max_epochs))
    lower_percentile = np.percentile(sampling_complexities, percentile)
    upper_percentile = np.percentile(sampling_complexities, 100 - percentile)
    return lower_percentile, upper_percentile


def evaluate_sampling_complexity_with_percentiles(path, filename, experiment_ids, epsilons, deltas,
                                                  samples_per_epoch=2000, max_epochs=100,
                                                  random_clusters=1000, percentile=5):
    # extract raw validation data
    data = extract(path, filename, experiment_ids)

    # compute sampling complexity for all epsilon-delta combinations
    sampling_complexities_with_percentiles = np.zeros((len(epsilons), len(deltas), 3))
    for epsilon_index, epsilon in enumerate(epsilons):
        for delta_index, delta in enumerate(deltas):
            sampling_complexities_with_percentiles[epsilon_index, delta_index, 0] \
                = sampling_complexity(data, epsilon, delta, samples_per_epoch, max_epochs)
            sampling_complexities_with_percentiles[epsilon_index, delta_index, 1:] \
                = cluster_bootstrapping(data, epsilon, delta, samples_per_epoch, max_epochs, random_clusters, percentile)
            # post-process to clean up some minor inaccuracies due to small re-sampling repetitions

    # post-process to clean up some minor inaccuracies due to small re-sampling repetitions
    np.clip(sampling_complexities_with_percentiles[:, :, 1], out=sampling_complexities_with_percentiles[:, :, 1],
            a_max=sampling_complexities_with_percentiles[:, :, 0], a_min=None)
    np.clip(sampling_complexities_with_percentiles[:, :, 2], out=sampling_complexities_with_percentiles[:, :, 2],
            a_max=None, a_min=sampling_complexities_with_percentiles[:, :, 0])

    return sampling_complexities_with_percentiles  # entries: [:, :, (sampling_complexity, lower_percentile, upper_percentile)]


if __name__ == '__main__':
    print(
        evaluate_sampling_complexity_with_percentiles(os.path.join('results', 'dqn', 'a3-main', 'deg3'), 'C[64-2]', range(100),
                                                      [0.2, 0.15, 0.1],  # fractional error
                                                      [0.5, 0.3, 0.1]  # failure probability
                                                      )
    )
