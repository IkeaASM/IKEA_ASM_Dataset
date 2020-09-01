function [registered_rgb] = get_registered_image(depth_img, rgb_img, ptCloud,  depth_ins)
registered_rgb = uint8(zeros(424, 512, 3));
x = ptCloud.Location(:, 1);
y = ptCloud.Location(:, 2);
z = ptCloud.Location(:, 3);
[r, c] = project_points(x, y, z, depth_ins);
indices = sub2ind([424, 512, 3], [r, r, r], [c, c, c], [ones(size(c)), 2*ones(size(c)), 3*ones(size(c)) ]);
registered_rgb(indices) = ptCloud.Color;
end