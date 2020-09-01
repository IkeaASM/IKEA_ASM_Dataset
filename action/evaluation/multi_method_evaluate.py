import os
methods_paths = [
                 # '../pose_based/log/ft/HCN_128/results/',
                 # '../clip_based/i3d/log/dev3/results/',
                 # '../clip_based/i3d/log/dev1/results/',
                 # '../clip_based/i3d/log/dev2/results/',
                 # '../clip_based/i3d/log/depth/results/',
                 # '../clip_based/i3d/log/combined/combined_w_depth/results/',
                 '../clip_based/i3d/log/combined/combined_w_depth_pose/results/',
                 '../clip_based/i3d/log/combined/combined_w_pose/results/',
                 # '../clip_based/c3d_and_p3d/log/c3d/results/',
                 # '../clip_based/c3d_and_p3d/log/p3d/results/',
                 # '../frame_based/log/resnet18/results/',
                 # '../frame_based/log/resnet50/results/',
                ]

for method_results in methods_paths:
    output_path = method_results
    print('evaluating ' + output_path + ' .......')
    os.system('python3 ../evaluation/evaluate.py --results_path {} --mode img'.format(output_path))