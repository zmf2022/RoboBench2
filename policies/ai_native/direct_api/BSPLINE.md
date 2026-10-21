# B-spline joint-position formulation

`bspline_jointpos` asks the model for six future joint-space control points and a trajectory length. The client anchors those points at the robot's current state and decodes them into absolute joint-position actions.

## Model output

Each control point is one absolute 8-D joint/gripper target:

$$
\mathbf{P}_i = [q_{i,1},\ldots,q_{i,7},g_i] \in \mathbb{R}^8,
\qquad g_i \in [0,1].
$$

The seven $q$ values are Franka joint positions. The scalar $g$ is a continuous gripper command. The model returns

- six future points $\mathbf{P}_1,\ldots,\mathbf{P}_6$;
- a sample count $K$, bounded by `bspline_min_steps` and maximum horizon $H=15$.
  The default minimum is 6; `--bspline-min-steps` can override it.

The model does not return knots or sampled actions.

## Anchoring and knots

The client prepends the measured joint/gripper state $\mathbf{P}_0$. The curve therefore has seven control points:

$$
\mathbf{P} = [\mathbf{P}_0,\mathbf{P}_1,\ldots,\mathbf{P}_6].
$$

With the default cubic degree $p=3$, the clamped open-uniform knot vector is

$$
\mathbf{t} =
\left[0,0,0,0,\frac{1}{4},\frac{1}{2},\frac{3}{4},1,1,1,1\right].
$$

The four repeated knots at each endpoint clamp the curve. The three interior knots are uniformly spaced. More generally, for $L$ control points, the decoder uses

$$
p_{\mathrm{eff}} = \min(p, L-1),
\qquad
r = L-p_{\mathrm{eff}}-1,
$$

where $r$ is the number of interior knots. The corresponding knot vector is

$$
\mathbf{t} = \left[
\underbrace{0,\ldots,0}_{p_{\mathrm{eff}}+1},
\frac{1}{r+1},\ldots,\frac{r}{r+1},
\underbrace{1,\ldots,1}_{p_{\mathrm{eff}}+1}
\right].
$$

When $r=0$, the interior-knot sequence is empty.

## Curve definition

The vector-valued B-spline uses one shared basis for all eight coordinates:

$$
\mathbf{C}(u) = \sum_{i=0}^{6} N_{i,3}(u)\,\mathbf{P}_i,
\qquad 0 \leq u \leq 1.
$$

The basis functions follow the Cox–de Boor recursion:

$$
N_{i,0}(u) =
\begin{cases}
1, & t_i \leq u < t_{i+1}, \\
0, & \text{otherwise},
\end{cases}
$$

$$
N_{i,p}(u) =
\frac{u-t_i}{t_{i+p}-t_i}N_{i,p-1}(u)
+
\frac{t_{i+p+1}-u}{t_{i+p+1}-t_{i+1}}N_{i+1,p-1}(u).
$$

A term with a zero denominator is defined as zero. The decoder evaluates this recursion with de Boor's algorithm. Clamping gives the endpoint conditions $\mathbf{C}(0)=\mathbf{P}_0$ and $\mathbf{C}(1)=\mathbf{P}_6$; the decoder also assigns these endpoint samples explicitly to avoid floating-point ambiguity.

This vector curve is equivalent to eight scalar splines sharing the same knots, basis functions, and sample times. It is not eight independently timed joint trajectories.

## Trajectory sampling

For a model-selected horizon $K$, the decoder evaluates $K+1$ evenly spaced parameters:

$$
u_j = \frac{j}{K}, \qquad j=0,\ldots,K.
$$

The first sample is the measured-state anchor and is removed. The returned trajectory is

$$
\mathbf{A} =
\left[\mathbf{C}(u_1),\ldots,\mathbf{C}(u_K)\right]
\in \mathbb{R}^{K \times 8}.
$$

Thus the first simulator input is already ahead of the measured state, and the last decoded row equals the model's final control point $\mathbf{P}_6$. The default minimum $K=6$ supplies at least as many sampled rows as future control points. `--bspline-min-steps` controls this bound independently; it does not change the number of control points or the spline degree.

## Safety and execution

The B-spline itself does not enforce joint limits, collision avoidance, or global kinematic feasibility. After sampling, the client:

1. limits each joint change to 0.20 rad relative to the measured state or previous decoded row;
2. clamps the seven arm values to the Franka joint limits;
3. thresholds the interpolated gripper coordinate into an open/closed command;
4. executes one absolute joint-position row per simulator control step.

The gripper is interpolated continuously inside the spline and binarized only after decoding. No additional interpolation occurs between sampled rows.

The implementation is in [`trajectory.py`](trajectory.py).
