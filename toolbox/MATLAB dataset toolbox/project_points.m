function [r, c] = project_points(x, y, z, depth_ins)
% Project the 3D points XYZ back to the image plane and get the pixel row
% and column
c = floor((x(z>0).* depth_ins.fx./ z(z>0)) + depth_ins.cx) + 1;
r = floor((y(z>0).* depth_ins.fy./ z(z>0)) + depth_ins.cy) + 1;
end