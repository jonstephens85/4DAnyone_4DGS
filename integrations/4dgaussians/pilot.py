ModelParams = dict(white_background=False)
OptimizationParams = dict(iterations=2000, coarse_iterations=300, batch_size=1,
    dataloader=True, densify_until_iter=1500, densify_from_iter=500,
    pruning_from_iter=500, opacity_reset_interval=3000)
ModelHiddenParams = dict(multires=[1,2],
    kplanes_config=dict(grid_dimensions=2,input_coordinate_dim=4,
        output_coordinate_dim=16,resolution=[32,32,32,32]))
