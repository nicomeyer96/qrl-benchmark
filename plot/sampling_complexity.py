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
Compute empirical sampling complexities from training data.
"""

import os
import numpy as np

from helper import extract


def sampling_complexity(data, epsilon, delta, samples_per_epoch, max_epochs):
    assert 2 == len(data.shape)
    assert data.shape[1] >= max_epochs, 'max_epochs exceeds available epochs in data'
    assert np.all(data >= 0.0) and np.all(data <= 1.0), 'data values are in invalid range'

    # mask data according to desired quality, i.e. 1-epsilon (ignore initial validation run)
    data_succeeded = data[:, :max_epochs] > 1 - epsilon
    # compute fraction of runs above required quality
    data_succeeded = np.average(data_succeeded, axis=0)

    # determine sampling complexity as number of runs that are of below required quality with probability delta
    sc_raw = np.sum(data_succeeded < delta)

    # multiply by number of samples used per epoch to get actual sampling efficiency
    sc = sc_raw * samples_per_epoch
    return sc


def evaluate_sampling_complexity(path, filename, experiment_ids, epsilons, deltas, samples_per_epoch=2000, max_epochs=100):
    # extract raw validation data
    data = extract(path, filename, experiment_ids)

    # compute sampling complexity for all epsilon-delta combinations
    sampling_complexities = np.zeros((len(epsilons), len(deltas)))
    for epsilon_index, epsilon in enumerate(epsilons):
        for delta_index, delta in enumerate(deltas):
            sampling_complexities[epsilon_index, delta_index] = sampling_complexity(data, epsilon, delta, samples_per_epoch, max_epochs)
    return sampling_complexities


if __name__ == '__main__':

    print(
        evaluate_sampling_complexity(os.path.join('results', 'dqn', 'a3-main', 'deg3'), 'C[64-2]', range(100),
                                     [0.2, 0.15, 0.1],  # fractional error
                                     [0.5, 0.7, 0.9]  # threshold probability
                                     )
    )
