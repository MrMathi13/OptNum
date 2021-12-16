import gym
import gym.envs.classic_control.pendulum
import numpy as np
import matplotlib.pyplot as plt
import functools

## In this exercise you are given the pendulum environment.
# This is based on gym.make("Pendulum-v1")
# i.e. gym.envs.classic_control.pendulum.PendulumEnv Differences: we
# directly take the state as observations (th, thdot), instead of
# (sin(th), cos(th), thdot), and we add some friction in the pendulum,
# because that makes our life easier.
# Also this one is not under-actuated (i.e. the motor has enough power)


class Pendulum(gym.envs.classic_control.pendulum.PendulumEnv):
    def __init__(self):
        super(Pendulum, self).__init__()
        self.max_torque = 30.0
        self.action_space = gym.spaces.Box(low=-self.max_torque, high=self.max_torque, shape=(1,), dtype=np.float32)
        self.goal = 0.0  # top position
        self.dt = 0.05  # 0.01
        self.state = None  # need to reset first
        self.last_u = None

    def reset(self):
        self.state = np.array([np.pi, 0.0])
        self.last_u = None
        return self.state

    def step(self, u):
        _obs, reward, done, info = super(Pendulum, self).step(u)
        th, thdot = self.state
        delta_thdot = -0.01 * thdot  # friction
        thdot = thdot - delta_thdot
        th = th + delta_thdot * self.dt
        th = (th + np.pi) % (2 * np.pi) - np.pi
        self.state = np.array([th, thdot])
        return self.state, reward, done, info


gym.envs.register(id="StatePendulum-v1", entry_point=Pendulum, max_episode_steps=200)


# This function performs a rollout using your controller function
#
# ufunc(Array[2]) -> Array[1]
#
# that maps the state to a control input.
def rollout(
    env: gym.Env,
    ufunc,
    err_reset=lambda state: np.array([0.0, 0.0, 0.0]),
    err_step=lambda err_state, state: err_state,
    render=True,
    print_states=False,
):
    """
    """
    state = env.reset()
    err_state = err_reset(state)
    done = False
    states = [state]
    err_states = [err_state]
    cost = 0.0
    while not done:
        u = ufunc(err_state)
        state, reward, done, _info = env.step(u)
        err_state = err_step(err_state, state)
        if render:
            env.render(mode="human")
        if print_states:
            print(state, err_state, u)
        states.append(state)
        err_states.append(err_state)
        cost -= reward
    return np.vstack(states), np.vstack(err_states), cost


# example u_func:
def go_right(err_state):
    del err_state
    return np.array([2.0])


if __name__ == "__main__":
    np.set_printoptions(formatter=dict(float=lambda number: "%9.3f" % (number,)))
    env = gym.make("StatePendulum-v1")
    env.reset()

    # Now you have to implement the necessary components for the PID
    # controller, that is you need the error signal, the integration
    # over the error signal and the derivative of the error signal.
    # And put that in a vector [int_error, error, dot_error]
    # where "error" is the difference between the current position of
    # the pendulum and the goal (env.goal)

    # Task 1.) (1 Point) Implement
    #
    # err_reset(state_0) -> error_state_0
    #
    # and
    #
    # err_step(error_state_k, state_k) -> error_state_{k+1}

    # in the rollout function we will use err_reset together with
    # env.reset() to initialize your error-signal tracker.  Then
    # env.step(action) is used to advance the environment state, and
    # your err_step function is used to advance the error_{k+1} based
    # on the last error_{k} and state_{k+1}.  Different to our
    # previous discrete time systems, the error signals current state
    # can be directly calculated:
    #
    # error_k+1 = theta_{k+1} - goal
    #
    # While   int_error is the integration of the error
    # and     dot_error is the derivative of the error
    #
    # So you have to do both a numerical integration (for int_error)
    # as well as a numerical derivative (for dot_error).

    def err_reset(state):
        pass # your code here
        return np.array([1.0, 2.0, 3.0])  # <- wrong

    def err_step(err_state, state):
        (th, _thdot) = state
        err_i_k, err_k, err_dot_k = err_state
        err_state_kp1 = np.array([err_i_k + 0.1, -err_dot_k, err_k * 1.01])  ## <-- very wrong
        pass # your code here
        return err_state_kp1

    # you can try your solution with
    rollout(env, go_right, err_reset=err_reset, err_step=err_step, render=True, print_states=True)

    # Task 2.) (1 Point) build a configurable controller function with configurable gains kI, kP, kD:
    # your controller function should map from err_state -> u
    # and u should be of np.array([scalar])
    #
    # Some ways of doing that would be a function closure, a lambda
    # function (to fix the parameters) or functools.partial.  ...or a
    # class but that's a bit overkill.
    def pidstep(err_state, kp=0.0, ki=0.0, kd=0.0):
        return np.array([np.sum(np.abs(err_state))])

    pass # your code here

    # Task 3.) (1 Point) Tune your controller from the previous step using the Ziegler-Nichols method!

    pid_ufunc = functools.partial(pidstep, kp=0.0, ki=0.0, kd=0.0)
    pass # your code here

    ### Plot the result!
    states, err_states, cost = rollout(env, pid_ufunc, err_reset=err_reset, err_step=err_step, render=True, print_states=True)

    plt.plot(err_states[:, 0], label="int_error")
    plt.plot(err_states[:, 1], label="error")
    plt.plot(err_states[:, 2], label="dot_error")
    plt.hlines(env.goal, 0, len(err_states), color="r", label="goal")
    plt.title(f"cost: {cost})")
    plt.legend()
    plt.show()

    # Task 4.) (1 Point) Further optimize the kp, ki, kd parameters!
    # In the previous task you used Ziegler-Nichols to tune the PID and the plot showed the cost in the title.
    # Optimize your controller further!
    # How small can you get the cost by tuning kp, ki, kd?
