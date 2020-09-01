function [points] = getPointsXYZ(undistorted, r, c, depth_ins)
%{
Extract xyz point coordinates from pixel index and depth
INPUT:
undistorted: undistorted depth image
r: row index
c: column index
depth_ins: depth camera intrinsics
OUTPUT:
point: xyz coordinates corresponding to pixel [r,c]
%}
% undistorted = fliplr(undistorted);
depth_val = undistorted(r,c);
h = size(r,2);
w = size(c,2);
x = NaN(h, w);
y = NaN(h, w);
z = NaN(h, w);
[c, r] = meshgrid(c, r);
idx = ~or(isnan(depth_val), depth_val <= 0.001);
x(idx) = (c(idx) - 0.5 - depth_ins.cx).* depth_val(idx)./ depth_ins.fx;
y(idx) = (r(idx) - 0.5 - depth_ins.cy).* depth_val(idx)./ depth_ins.fy;
z(idx) = depth_val(idx);


points = cat(3, x, y, z );
end