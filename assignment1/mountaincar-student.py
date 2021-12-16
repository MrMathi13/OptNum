import gym
import numpy as np
from sklearn.utils.extmath import cartesian
import matplotlib.pyplot as plt
import time
import tqdm


### Utility for plotting heatmaps
class GridEval:
    def __init__(self, grid_evaluate, x1_min, x1_max, x2_min, x2_max, x1steps=10, x2steps=10):
        k1 = np.linspace(x1_min, x1_max, x1steps)
        k2 = np.linspace(x2_min, x2_max, x2steps)
        grid_points = cartesian([k1, k2])
        result = np.vstack([grid_evaluate(gp) for gp in tqdm.tqdm(grid_points)])
        grid_result = result.reshape((k1.shape[0], k2.shape[0])).T
        self.k1 = k1
        self.k2 = k2
        self.grid_result = grid_result

    def plot(self, close=True):
        k1 = self.k1
        k2 = self.k2
        if close:
            plt.close()
        plt.imshow(self.grid_result, extent=[k1[0], k1[-1], k2[0], k2[-1]], aspect="auto", origin="lower")
        plt.xlabel("k1")
        plt.ylabel("k2")
        plt.colorbar()
        plt.show()


# we can define a do-nothing controller:
def U_go_right(x):
    return [1]


def render(env, U_func, n=1):
    for i in range(n):
        episode_return = 0
        state = env.reset()
        done = False
        while not done:
            state, reward, done, info = env.step(U_func(state))
            env.render(mode="human")
            time.sleep(0.01)


if __name__ == "__main__":
    # In this exercise we take a first look at defining controllers (policies).
    # A controller/policy provides us with a control input given a state: u = f(x)

    # we first instantantiate the environment
    env = gym.make("MountainCarContinuous-v0")

    # above there is a pre-defined controller that simply tries to
    # drive up the mountain.  we can visualise the struggle:

    render(env, U_go_right)

    # 1.) (1 Point) Define the controller as a linear function of the state:
    #     u=Kx   K is a 1x2 matrix.
    # Create a new function that takes K as input, and returns a new control function.
    # K is called the control-gain matrix.
    # try some values for the control gain and render.

    def Kfunc(K):
        pass # your code here
        return lambda x: np.array([0])

    # 2.) (1 Point) write an evaluation function that uses a given control
    # function, resets the environment and does n (=5 as default)
    # rollouts with your controller. accumulating the reward
    # (=-cost).:

    # define an evaluate function
    # def evaluate(env, U_func, n=5):
    # - that loops n-times
    #  - sets the initial state
    #  - loop until the done flag is true
    #   - steps according to u(state)
    #   - accumulates the reward
    #  - take the sum_of_the_reward
    # - take the mean of the [sum_of_the_reward, ...], and return that.
    pass # your code here

    # 3.) (1 Point) Since the gain matrix is 1x2 we can visualize the
    # quality (== what the evaluate function gives us) for different
    # gain parameters as a plot.  the function should take k1, k2 as
    # input and return how good (evaluate) that controller is.  Use
    # that plot to select k1, k2 to get a good controller.

    # here is an example of such a function, calculating something completely different.
    def grid_eval_example(data):
        k1, k2 = data
        return k1 + (k2 / 10) ** 3  # obviously not the way to create
                                    # the controller and run the
                                    # evaluate...

    pass # your code here

    # and with this code we can plot a heatmap.
    grid = GridEval(grid_eval_example, -10, 10, -10, 10)
    grid.plot()

    # 4.) (1 Point) Phase-space plots: Since the state space of the
    # mountain car is two dimensional we can plot trajectories that
    # the car takes as trajectories through the phase space
    #
    #(i.e.: x = position, y = velocity)
    #
    # define a rollout function
    #
    # def rollout(env, U_func) -> array of shape [time,2]
    #
    # select 2 different K matrices based on the gripmap evaluation
    # above and plot the resulting trajectories in the phase-space.
    trajectory = np.vstack([(np.cos(x), np.sin(3 * x)) for x in np.linspace(0, 2 * np.pi, 100)])
    # shape: (time, 2)
    plt.close()
    plt.plot(*trajectory.T, label="I am the label")
    plt.legend()
    plt.show()
    pass # your code here
