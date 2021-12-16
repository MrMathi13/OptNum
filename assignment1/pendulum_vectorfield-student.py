import jax.numpy as jnp
from tensor_annotations import axes
import tensor_annotations.jax as taj
from sklearn.utils.extmath import cartesian
import jax
import matplotlib.pyplot as plt

### Definitions for Type Annotations
class State(axes.Axis):
    pass


class Action(axes.Axis):
    pass


## Plot a quiver plot for given matplotlib-axes (ax) and F_step function
def visualize_for_x(
        f_step, x_low=jnp.array([-0.1, -5]), x_high=jnp.array([2 * jnp.pi, 5]), u=jnp.zeros((1,)), steps=30, ax=None,
        title=""
):
    if ax is None:
        ax = plt.gca()
    xs = jnp.linspace(x_low, x_high, steps)
    grid_points = cartesian(xs.T)
    next_steps = jax.vmap(f_step)(grid_points, jnp.tile(u, len(grid_points)))
    delta = next_steps - grid_points
    lengths = jnp.linalg.norm(delta, axis=1)
    ndelta = delta / lengths[:, None]
    # adjusting the arrowsize size
    quiver_dict = dict(
        # scale=10., # larger number -> smaller arrow
        # scale_units="xy",
        # linewidths=0.001,
    )
    ax.quiver(grid_points[:, 0], grid_points[:, 1], ndelta[:, 0], ndelta[:, 1], lengths, **quiver_dict)
    ax.set_xlabel("theta")
    ax.set_ylabel("dot theta")
    ax.set_title(title)


# You're given the discrete-time system for the pendulum as a non-linear system:
# \[ x_{k+1} = F_{step}(x_k, u_k) \]
def F_step(x_k: taj.Array1[State], u_k: taj.Array1[Action]) -> taj.Array1[State]:
    th, thdot = jnp.split(x_k, 2, axis=-1)
    th = jnp.reshape(th, (-1,))
    thdot = jnp.reshape(thdot, (-1,))
    # --- model parameters
    g = 10.0
    m = 1.0
    l = 1.0
    dt = 0.05
    # ---
    newthdot = thdot + (3 * g / (2 * l) * jnp.sin(th) + 3.0 / (m * l ** 2) * u_k) * dt
    newth = th + newthdot * dt
    # ---
    x_kp1 = jnp.hstack([newth[:, None], newthdot[:, None]])
    return x_kp1.reshape(x_k.shape)


# helper function to visualize the vector field of one step function
# and plot specific point.
def visualize_one(ax, f_step, title, x_prime):
    visualize_for_x(f_step, ax=ax)
    ax.set_title(title)
    ax.scatter(*x_prime, marker="x", color="r")


# helper function to do a comparison between your linearization
# functions.
def plot_it_all(f_step, finite_differencer, autodiff_linearise, one_visualizer=visualize_one):
    plt.close()
    x_prime_list = [jnp.pi / 2, -1.0]
    fig_width, fig_height = plt.rcParams["figure.figsize"]
    fig = plt.figure(figsize=(fig_width, fig_height * 3))

    def onclick(event):
        if event and event.button == 1:
            x_prime_list[0] = event.xdata
            x_prime_list[1] = event.ydata
        x_prime = jnp.array(x_prime_list)
        plt.clf()
        fig = plt.gcf()
        (ax, ax2, ax3) = fig.subplots(3, 1)

        one_visualizer(ax, f_step, "Exact, non-linear", x_prime)

        plt.draw()  # redraw
        linear_F_step = make_F_linear(*finite_differencer(f_step, x_prime=x_prime))
        one_visualizer(ax2, linear_F_step, "Finite-difference", x_prime)
        plt.draw()  # redraw

        linear_F_step = make_F_linear(*autodiff_linearise(f_step, x_prime=x_prime))
        one_visualizer(ax3, linear_F_step, "Auto differentiation", x_prime)
        plt.draw()  # redraw

    onclick(None)
    fig.canvas.mpl_connect("button_press_event", onclick)
    plt.show()
    plt.draw()


if __name__ == "__main__":
    # Alright now for the actual exercise:
    # you can visualize the transition function using:
    visualize_for_x(F_step, title="Vectorfield of the Pendulum")
    plt.show()

    ## 1.) (0.5 Points) Create a linearized step function for given G, H, c
    #  (the step function calculates x_{k+1} based on x_k and u)
    def make_F_linear(G, H, c):
        def linear_F_step(x, u):
            print(f"G {G.shape}\nH: {H.shape}\n, c:{c.shape}\nX:{x.shape}\nu:{u.shape}")
            pass # your code here
            x_kp1 = -x + jnp.sum(G) # <-- this is obviously wrong.
            return x_kp1

        return linear_F_step

    # 2.) (1.5 Points) linearise this system around a given x' and u'
    #     using the finite difference method, to retrieve a system:
    #     x_{k+1} = G x + H u + c
    #     i.e. this function takes
    #     F_step -> G, H, c
    def finite_difference(f_step, x_prime=jnp.zeros((2,)), u_prime=jnp.zeros((1,)), delta=0.0001):
        pass # your code here
        G = jnp.array([[x_prime[0], x_prime[1]], [0.0, 0.0]])
        H = jnp.array([[0.0], [0.0]])
        c = jnp.array([0.0, 0.0])
        return G, H, c

    # 3.) (1.5 Points) use automatic differentiation provided by the jax library to
    #     linearise the system around a given x' and u' to retrieve a
    #     system:
    #     x_{k+1} = G x + H u + c
    # https://jax.readthedocs.io

    def linearise_autodiff(f_step, x_prime=jnp.zeros((2,)), u_prime=jnp.zeros((1,))):
        pass # your code here
        G = jnp.array([[0.0, 1.0], x_prime])
        H = jnp.array([[0.0], [0.0]])
        c = jnp.array([0.1, 0.2])
        return G, H, c

    plot_it_all(F_step, finite_difference, linearise_autodiff)

    # 4.) (0.5 Points) rollout: Implement a rollout using the F_step function
    # starting from x_prime, and using constant actions u_prime.
    def rollout(f_step, x_prime, u_prime=jnp.zeros(1), n=50):
        trajectory = [x_prime]
        for i in range(n):
            trajectory.append(x_prime * (i * 0.01))
        pass # your code here
        return jnp.vstack(trajectory)

    def visualize_one_with_rollout(ax, f_step, title, x_prime):
        ax.set_title(title)
        visualize_for_x(f_step, ax=ax)
        ax.scatter(*x_prime, marker="x", color="r")
        trajectory = rollout(f_step, x_prime)
        ylim = ax.get_ylim()
        xlim = ax.get_xlim()
        ax.plot(*trajectory.T)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)

    plot_it_all(F_step, finite_difference, linearise_autodiff, one_visualizer=visualize_one_with_rollout)
