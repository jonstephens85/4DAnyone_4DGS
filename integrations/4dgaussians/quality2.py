# As quality.py, but with opacity reset restored. Disabling it (upstream's
# choice for static-rig captures) left diverged Gaussians on the fast-moving
# forearm with no mechanism to clear them: the green channel collapsed to zero.
_base_ = './quality.py'
OptimizationParams = dict(opacity_reset_interval=3000)
