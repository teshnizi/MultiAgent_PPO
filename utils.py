import gymnasium as gym


def make_env(env_id, seed, idx, capture_video, run_name, **kwargs):
    def thunk():
        env = gym.make(env_id, **kwargs)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        return env

    return thunk


def strtobool(val):
    val = val.lower()
    if val in {'y', 'yes', 't', 'true', 'on', '1'}:
        return True
    elif val in {'n', 'no', 'f', 'false', 'off', '0'}:
        return False
    else:
        raise ValueError(f"invalid truth value {val}")
