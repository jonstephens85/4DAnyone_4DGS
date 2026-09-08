# As quality.py, but allowing per-timestamp opacity and colour deformation.
# With no_dshs/no_do left at their True defaults each Gaussian carries a single
# colour for all 121 frames; the larger deformation grid moves primitives far
# enough that one must serve skin at one instant and shirt or denim at another,
# and the compromise colour drove the forearm's green channel to zero.
# Upstream's own real-capture config sets both to False.
_base_ = './quality.py'
ModelHiddenParams = dict(no_dshs=False, no_do=False)
