# NeuroPose

Package for creating and working with a neuromorphic model for maintaining three-dimensional pose estimates via attractor dynamics. The representation is updated through neural dynamics according to tangential and angular velocity inputs.

### Project Report
This package was created as part of a project titled **Neuromorphic Implementation of a 3D Attractor Network for Maintaining and Updating Multiple Simultaneous Pose Estimates During Path Integration**. The final report is available at [report.pdf](report.pdf).

### Code Documentation
Please refer to [martinschonger.github.io/neuropose/pose](https://martinschonger.github.io/neuropose/pose.html).

### Debug Repo
Extensive debug, eval, plotting, and other supporting code can be found at [github.com/martinschonger/neuropose-debug](https://github.com/martinschonger/neuropose-debug) (private due to sensitive information).

### Example Usage
How to run the pose network on the Nengo Loihi backend, with velocity input
corresponding to straight movement. We use [Sacred](https://github.com/IDSIA/sacred) and a
MongoDB database for configuration management and organizing simulation results.

```python
from exp import ex

# Run the simulation. Specify duration, input commands, and enable the Nengo Loihi backend. All
# other parameters get default values as defined in `exp.default_config`.
ex.run(config_updates={
    'simulation_duration': 5.7,
    'input': [{  # Initialize a pose estimate, specified in reciprocal grid coordinates.
        'duration': 0.2,
        'cmds': [{'cmd': 'input_freq',
                    'bump_centers': [(4, 4, 0)]}]
    }, {  # Short period of no movement to allow the attractor network reach a stable state.
        'duration': 0.5,
        'cmds': [{'cmd': 'manual',
                    'shift_inhib': 1,
                    'pos_rot_shift_inhib': 1,
                    'neg_rot_shift_inhib': 1}]
    }, {  # Constant tangential velocity input of 0.5, angular velocity input of 0.
        'duration': 5.0,
        'cmds': [{'cmd': 'manual_vel',
                         'tangential_vel': 0.5,
                         'angular_vel': 0}]
    }],
    'use_loihi': True});
```

The data generated during simulation can be accessed and decoded as follows.
Assuming defaults for everything, you only need to plug in ``your_mongo_uri``.

The ``decoders`` can be obtained as the weights of the main recurrent connection as computed by Nengo. To this end, disable our custom direct neurons-to-neurons connection and enable the NEF-style connection in `nengo_utils`. Then get the decoders like ``decoders = sim.data[pose_network.attractor_network.rec_con].weights.copy()``. This needs to be done only once. Do not forget to undo the changes in `nengo_utils` afterwards.

Eventually, we arrive at a set of pose estimates for each time point.

```python
import incense
import nengo
import numpy as np

from exp import ex
from pose.experiments import get_cluster_centers_timeseries, reconstruct_timeseries
from pose.freq_space import get_0_coefs
from pose.hex import fspace_base, get_3d_coordinates_unwrapped_vectorized
from pose.nengo_utils import encoding_mask
from pose.plotting import plot_experiment_xy

# Load the experiment's metadata and data from the database.
loader = incense.ExperimentLoader(
    mongo_uri=your_mongo_uri,
    db_name='sacred'
)
def load_experiment(loader):
    exp = loader.find_latest()  # we want the most recent experiment
    _config = exp.to_dict()['config']
    data_raw = exp.artifacts['sim_data.pkl'].as_type(incense.artifact.PickleArtifact)
    data = data_raw.render()  # load pickle
    return exp.to_dict(), _config, data
_exp_info, _config, data = load_experiment(loader)

# Get some config values.
variance_pose = _config['variance_pose']
cov = np.eye(3) * variance_pose
fgrid_shape = _config['fgrid_shape']

# For each time point, decode the neuron activations into representational space.
decoded_sim_data = decoders.dot(data['pose_network.output'].transpose())
decoded_sim_data = decoded_sim_data.transpose()

# Reintroduce the ommitted imaginary parts of the Nyquist terms.
output_restored = np.zeros((decoded_sim_data.shape[0], np.prod(fgrid_shape)*2))
output_restored[:, encoding_mask(fgrid_shape).ravel()] = decoded_sim_data

# Undo normalization to obtain the final reciprocal space representation.
gauss0_f_cropped_flat = get_0_coefs(fgrid_shape, cov)
fact = gauss0_f_cropped_flat[0] / gauss0_f_cropped_flat[2]
output_restored[:, 0] = output_restored[:, 2] * fact

# Back-transform into real space.
pose_reconstructed = reconstruct_timeseries(output_restored, fgrid_shape)

# Compute pose estimates from their 3D-Gaussian representation.
bump_centers = get_cluster_centers_timeseries(pose_reconstructed, fgrid_shape)

# Plot the projection of the pose estimates along the third axis, color-coded by time.
plot_experiment_xy(_exp_info, _config, bump_centers, time_slice=np.s_[7:])
```

<br/>

### Note
This README file is adapted from [pose/\_\_init\_\_.py](pose/__init__.py).

<br/>

Copyright © 2023 Martin Schonger  
This software is licensed under the GPLv3.

The content in [pose/fonts](pose/fonts) is licensed under the GPLv3 with copyright belonging to
the authors listed in [pose/fonts/gnu_freefont/AUTHORS](pose/fonts/gnu_freefont/AUTHORS) and [pose/fonts/gnu_freefont/CREDITS](pose/fonts/gnu_freefont/CREDITS).
