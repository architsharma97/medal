import pickle
import numpy as np
import random
import torch
from torch.utils.data import Dataset
from backend.timestep import ExtendedTimeStep, StepType

def filterNone(step_type):
    return False

def filterFIRST(step_type):
    return step_type == StepType.FIRST

def filterLAST(step_type):
    return step_type == StepType.LAST


class SimpleReplayBuffer:
    """Simple replay buffer that takes in (s, a, r) tuples temporally."""

    def __init__(self,
                 data_specs,
                 max_size,
                 batch_size=None,
                 replay_dir=None,
                 discount=0.99,
                 filter_transitions=filterFIRST,
                 with_replacement=True):
        """Initialize a replay buffer.

        Arguments:
            data_specs
            max_size (int): Maximum size of replay buffer.
            batch_size (int): Default batch size when sampling.
            replay_dir (str): Directory for loading a saved replay buffer.
            discount (float): Discount factor.
            filter_transitions (func): Function to filter s' when sampling an
                (s, a, r, s') pair.
            with_replacement (bool): Whether to sample with replacement when
                returning a random batch.
        """
        self._data_specs = data_specs
        self._max_size = int(max_size)  # assume max_size >= total transitions
        self._batch_size = batch_size
        self._discount = discount
        self._replay_dir = replay_dir
        self._filter_transitions = filter_transitions
        self._with_replacement = with_replacement
        self.reset()

    def reset(self):
        """Reset replay buffer by removing all entries."""
        self._replay_buffer = {}
        for spec in self._data_specs:
            self._replay_buffer[spec.name] = np.empty(
                (self._max_size, *spec.shape), dtype=spec.dtype)
        self._replay_buffer['step_type'] = np.empty((self._max_size, 1),
                                                    dtype=np.int32)
        self._num_transitions = 0
        self._next_idx = 0

    def add_offline_data(self, demos, default_action=None):
        """Add a batch of offline data, each is a (s, a, r, s') tuple.
        Arguments:
            demos (dict):
                'observations' (np.array): (bsz, dim_o)
                'actions' (np.array): (bsz, dim_a)
                'rewards' (np.array): (bsz, )
                'terminals' (np.array): (bsz, ): Whether s' in the (s, a, r, s')
                    tuple is LAST. This is usually all True.
                'next_o' (np.array): (bsz, dim_o)
            default_action (np.array): (dim_a, ): Default action to use for
                (s', a, r) when s' is FIRST. This is a dummy action that is not
                used.
        """
        obs = demos['observations']
        acts = demos['actions']
        rew = demos['rewards']
        term = demos['terminals']
        next_o = demos['next_observations']

        if default_action is None:
            default_action = acts[0]

        for idx in range(obs.shape[0]):
            # If previous step is LAST, add a FIRST step so that we would not
            # sample (LAST, FIRST).
            if idx == 0 or term[idx - 1][0]:
                # Previous step is termination, so this step type is FIRST
                time_step = ExtendedTimeStep(observation=obs[idx],
                                             step_type=StepType.FIRST,
                                             action=default_action,
                                             reward=0.0,
                                             discount=1.0)
                self.add(time_step)

            if term[idx][0]:
                # Current step is termination, so this step type is LAST.
                time_step = ExtendedTimeStep(observation=next_o[idx],
                                             step_type=StepType.LAST,
                                             action=acts[idx],
                                             reward=rew[idx][0],
                                             discount=1.0)
            else:
                # Step type is MID.
                time_step = ExtendedTimeStep(observation=next_o[idx],
                                             step_type=StepType.MID,
                                             action=acts[idx],
                                             reward=rew[idx][0],
                                             discount=1.0)
            self.add(time_step)

    def next(self, batch_size=None, filter_transitions=None, with_replacement=None):
        """Return a batch of transitions."""
        assert self._num_transitions > 1, "Replay buffer only has 1 time step!"
        batch_size = self._batch_size if batch_size is None else batch_size

        filter_transitions = self._filter_transitions if filter_transitions is None else filter_transitions
        with_replacement = self._with_replacement if with_replacement is None else with_replacement

        # Because we look back a step, we start at 1, not 0
        if with_replacement:
            idxs = np.random.randint(1, len(self), size=batch_size)
        else:
            # do not use np.random.choice, it gets much slower as the size increases
            idxs = np.array(random.sample(range(1, len(self)), batch_size), dtype=np.int64)

        # Shift to correct start
        idxs = (idxs + self._next_idx) % self._num_transitions

        # Filter transitions
        filtered_idxs = []
        for idx in idxs:
            step_type = self._replay_buffer['step_type'][idx]
            if not filter_transitions(step_type):
                filtered_idxs.append(idx)

        filtered_idxs = np.array(filtered_idxs)
        prev_idxs = (filtered_idxs - 1) % self._num_transitions

        return (
            self._replay_buffer['observation'][prev_idxs],
            self._replay_buffer['action'][filtered_idxs],
            self._replay_buffer['reward'][filtered_idxs],
            self._replay_buffer['discount'][filtered_idxs],
            self._replay_buffer['observation'][filtered_idxs],
            self._replay_buffer['step_type'][prev_idxs].squeeze(1),
            self._replay_buffer['step_type'][filtered_idxs].squeeze(1),
        )

    def gather_all(self, filter_transitions=None):
        """Gather all (s, a, r, s') pairs in the replay buffer."""
        assert self._num_transitions > 1, "Replay buffer only has 1 time step!"
        filter_transitions = self._filter_transitions if filter_transitions is None else filter_transitions
        return self.next(self._num_transitions - 1, filter_transitions)

    def add(self, time_step):
        """Add new time step to replay buffer. If we are at `self._max_size`,
        overwrite the oldest step according to `self._next_idx`, the index at 
        which we add the next new time step."""
        for spec in self._data_specs:
            value = time_step[spec.name]
            if spec.name == 'discount':
                value = np.expand_dims(time_step.discount * self._discount,
                                       0).astype('float32')
            if np.isscalar(value):
                value = np.full(spec.shape, value, spec.dtype)
            assert spec.shape == value.shape and spec.dtype == value.dtype
            np.copyto(self._replay_buffer[spec.name][self._next_idx], value)

        np.copyto(self._replay_buffer['step_type'][self._next_idx],
                  time_step.step_type)

        self._next_idx = (self._next_idx + 1) % self._max_size
        self._num_transitions = min(self._num_transitions + 1, self._max_size)

    def __len__(self):
        return self._num_transitions

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    @property
    def batch_size(self):
        return self._batch_size

    def save_buffer(self):
        with open(self._replay_dir / 'episodic_replay_buffer.buf', 'wb') as f:
            pickle.dump(self._current_episode, f)
        np.save(self._replay_dir / 'num_transitions.npy',
                self._num_transitions)

    def load(self):
        try:
            self._replay_buffer = pickle.load(
                open(self._replay_dir / 'episodic_replay_buffer.buf'), 'rb')
            self._num_transitions = np.save(self._replay_dir /
                                            'num_transitions.npy').tolist()
        except:
            print('no replay buffer to be restored')



class UnionBuffer:
    def __init__(self,
                 replay_buffer_1, replay_buffer_2,
                 batch_size=None,
                 filter_transitions=True,
                 with_replacement=True,):

        self._rb1 = replay_buffer_1
        self._rb2 = replay_buffer_2
        self._batch_size = batch_size
        self._filter_transitions = filter_transitions
        self._with_replacement = with_replacement

    def __iter__(self):
        return self

    def __next__(self):
        batch_size_1 = np.random.binomial(self._batch_size,
                                          len(self._rb1) / (len(self._rb1) + len(self._rb2)))
        batch_size_2 = self._batch_size - batch_size_1

        # batch size being 0 should be handled correctly
        batch_1 = self._rb1.next(batch_size=batch_size_1,
                                 filter_transitions=self._filter_transitions,
                                 with_replacement=self._with_replacement)
        batch_2 = self._rb2.next(batch_size=batch_size_2,
                                 filter_transitions=self._filter_transitions,
                                 with_replacement=self._with_replacement)
        batch = ()
        for el1, el2 in zip(batch_1, batch_2):
            batch += (np.concatenate([el1, el2], axis=0),)

        return batch
