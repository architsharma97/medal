import earl_benchmark
import numpy as np

from backend.wrappers import (
    ActionRepeatWrapper,
    ObsActionDTypeWrapper,
    ExtendedTimeStepWrapper,
    ActionScaleWrapper,
    DMEnvFromGymWrapper,
)

def make(name, frame_stack, action_repeat):
    env_loader = earl_benchmark.EARLEnvs(
        name,
        reward_type="sparse",
        reset_train_env_at_goal=False,
    )
    train_env, eval_env = env_loader.get_envs()
    reset_states = env_loader.get_initial_states()
    reset_state_shape = reset_states.shape[1:]
    goal_states = env_loader.get_goal_states()
    if env_loader.has_demos():
        forward_demos, backward_demos = env_loader.get_demonstrations()
    else:
        forward_demos, backward_demos = None, None

    # add wrappers
    train_env = DMEnvFromGymWrapper(train_env)
    train_env = ObsActionDTypeWrapper(train_env, np.float32, np.float32)
    train_env = ActionRepeatWrapper(train_env, action_repeat)
    train_env = ActionScaleWrapper(train_env, minimum=-1.0, maximum=+1.0)
    train_env = ExtendedTimeStepWrapper(train_env) 

    eval_env = DMEnvFromGymWrapper(eval_env)
    eval_env = ObsActionDTypeWrapper(eval_env, np.float32, np.float32)
    eval_env = ActionRepeatWrapper(eval_env, action_repeat)
    eval_env = ActionScaleWrapper(eval_env, minimum=-1.0, maximum=+1.0)
    eval_env = ExtendedTimeStepWrapper(eval_env)

    return train_env, eval_env, reset_states, goal_states, forward_demos, backward_demos