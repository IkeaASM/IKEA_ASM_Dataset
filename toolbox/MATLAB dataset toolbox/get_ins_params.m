
function [rgb_ins_params, depth_ins_params] = get_ins_params(path)
% Load the camera intrinsics 
depth_ins_params_filname = fullfile(path, 'DepthIns.txt');
rgb_ins_params_filname = fullfile(path, 'ColorIns.txt');

depth_ins_params_mat = dlmread(depth_ins_params_filname);
depth_ins_params.fx = depth_ins_params_mat(1);
depth_ins_params.fy = depth_ins_params_mat(2);
depth_ins_params.cx = depth_ins_params_mat(3);
depth_ins_params.cy = depth_ins_params_mat(4);
depth_ins_params.k1 = depth_ins_params_mat(5);
depth_ins_params.k2 = depth_ins_params_mat(6);
depth_ins_params.k3 = depth_ins_params_mat(7);
depth_ins_params.p1 = depth_ins_params_mat(8);
depth_ins_params.p2 = depth_ins_params_mat(9);

rgb_ins_params_mat = dlmread(rgb_ins_params_filname);
rgb_ins_params.fx = rgb_ins_params_mat(1);
rgb_ins_params.fy = rgb_ins_params_mat(2);
rgb_ins_params.cx = rgb_ins_params_mat(3);
rgb_ins_params.cy = rgb_ins_params_mat(4);
rgb_ins_params.shift_d = rgb_ins_params_mat(5);
rgb_ins_params.shift_m = rgb_ins_params_mat(6);
rgb_ins_params.mx_x3y0 = rgb_ins_params_mat(7);
rgb_ins_params.mx_x0y3 = rgb_ins_params_mat(8);
rgb_ins_params.mx_x2y1 = rgb_ins_params_mat(9);
rgb_ins_params.mx_x1y2 = rgb_ins_params_mat(10);
rgb_ins_params.mx_x2y0 = rgb_ins_params_mat(11);
rgb_ins_params.mx_x0y2 = rgb_ins_params_mat(12);
rgb_ins_params.mx_x1y1 = rgb_ins_params_mat(13);
rgb_ins_params.mx_x1y0 = rgb_ins_params_mat(14);
rgb_ins_params.mx_x0y1 = rgb_ins_params_mat(15);
rgb_ins_params.mx_x0y0 = rgb_ins_params_mat(16);

rgb_ins_params.my_x3y0 = rgb_ins_params_mat(17);
rgb_ins_params.my_x0y3 = rgb_ins_params_mat(18);
rgb_ins_params.my_x2y1 = rgb_ins_params_mat(19);
rgb_ins_params.my_x1y2 = rgb_ins_params_mat(20);
rgb_ins_params.my_x2y0 = rgb_ins_params_mat(21);
rgb_ins_params.my_x0y2 = rgb_ins_params_mat(22);
rgb_ins_params.my_x1y1 = rgb_ins_params_mat(23);
rgb_ins_params.my_x1y0 = rgb_ins_params_mat(24);
rgb_ins_params.my_x0y1 = rgb_ins_params_mat(25);
rgb_ins_params.my_x0y0 = rgb_ins_params_mat(26);
end