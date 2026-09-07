_base_ = './pilot.py'
OptimizationParams = dict(iterations=20000, coarse_iterations=3000,
    densify_until_iter=15000, pruning_from_iter=500, opacity_reset_interval=3000)
