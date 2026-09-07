# Maximum-quality configuration for the 47-view full-resolution export.
# Deliberately does not inherit pilot.py: that config downgrades the upstream
# deformation-grid defaults, which limits achievable sharpness.
ModelParams = dict(white_background=False)
ModelHiddenParams = dict(
    # multires and output_coordinate_dim keep the upstream defaults
    # ([1,2,4,8] and 32); only the temporal axis is matched to this clip.
    net_width=128,
    kplanes_config=dict(grid_dimensions=2, input_coordinate_dim=4,
        output_coordinate_dim=32, resolution=[64, 64, 64, 60]))
OptimizationParams = dict(
    dataloader=True, batch_size=2,
    iterations=30000, coarse_iterations=3000,
    densify_from_iter=500, densify_until_iter=20000, pruning_from_iter=500,
    # Half the default gradient threshold, to actually grow primitives.
    densify_grad_threshold_coarse=0.0001,
    densify_grad_threshold_fine_init=0.0001,
    densify_grad_threshold_after=0.0001,
    # Upstream disables opacity reset for real captures; resets erase detail.
    opacity_reset_interval=60000,
    position_lr_max_steps=30000,
    lambda_dssim=0.2)
