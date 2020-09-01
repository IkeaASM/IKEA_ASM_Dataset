clear all
close all
clc


path_to_dataset = '~/Datasets/ANU_ikea_dataset/';
processed_dataset_path = '~/Datasets/ANU_ikea_dataset_processed/';
device_dir= 'dev3';
scan_name = '0017_black_table_01_01_2019_08_15_16_36'; % '0001_oak_floor_01_01_2019_08_12_16_18'; % 
category_name = 'Lack SIde Table';
frame_number = '000000';
CAD_file_type = '.ply';
CAD_path = fullfile(path_to_dataset, category_name, 'CAD');
scan_path = fullfile(path_to_dataset, category_name,  scan_name, device_dir);
processed_scan_path = fullfile(processed_dataset_path, category_name,  scan_name, device_dir);
annotation_file = fullfile(fullfile(processed_scan_path, 'annotations'),[frame_number, '_annotations.txt'] );
point_cloud_file = fullfile(fullfile(processed_scan_path, 'point_clouds'), [frame_number, '.ply']);
[color_ins, depth_ins] = get_ins_params(scan_path);
depth_path = fullfile(scan_path, 'depth');
depth_img = fliplr(double(imread(fullfile(depth_path, [frame_number, '.png']))));
rgb_path = fullfile(scan_path, 'images');
rgb_img = fliplr(imread(fullfile(rgb_path, [frame_number, '.jpg'])));

xyzPoints = getPointsXYZ(depth_img, 1:424, 1:512, depth_ins);
xyzPoints_no_crop = getPointsXYZ(depth_img, 1:424, 1:512, depth_ins);
ptCloud = pcread(point_cloud_file);
point_color = get_registered_image(depth_img, rgb_img, ptCloud,  depth_ins); % should improve to load from the rgb image


% visualization
fig_x0 = 0; fig_y0 = 0; fig_w = 0.5; fig_h = 1; 
figure('numbertitle','off','name', '3D part poses','units','normalized','outerposition',[fig_x0, fig_y0, fig_w, fig_h],'color','k');
ax_pc_poses = axes('color','k');

annotation = ReadAnnotationFile(annotation_file);
part_list = get_cad_part_list(fullfile(path_to_dataset, category_name));
for i = 1:length(part_list)
    CAD_file_name = fullfile(CAD_path, [part_list{i}, CAD_file_type]);
    if strcmp(CAD_file_type, '.ply')
        FV = my_plyread(CAD_file_name);
    else
        FV = stlread(CAD_file_name);
    end
    [FV.vertices, FV.faces]=patchslim(FV.vertices, FV.faces);
    COG = mean(FV.vertices);
    FV.vertices = FV.vertices - repmat(COG, [size(FV.vertices,1), 1]); %/ 1000; % from milietersm to meters

    T = makehgtform('translate', annotation{i}.T);
    Rx = makehgtform('xrotate', deg2rad(annotation{i}.R(1)));
    Ry = makehgtform('yrotate', deg2rad(annotation{i}.R(2)));
    Rz = makehgtform('zrotate', deg2rad(annotation{i}.R(3)));
    M = T*Rx*Ry*Rz;
    
    hgtransform_h = hgtransform('parent', gca, 'matrix',M);
    part_handle = patch(FV,'facecolor','g','facealpha',0.8, 'parent', hgtransform_h);
    

    cad_points_in_scene = M * [FV.vertices'; ones(1, size(FV.vertices,1))];
    [r, c] = project_points(cad_points_in_scene(1, :)', cad_points_in_scene(2, :)', cad_points_in_scene(3, :)', depth_ins);
    [Y, X] = meshgrid(1:512, 1:424);
    for polygon = FV.faces.'
        mask = inpolygon(X, Y, r(polygon), c(polygon));
        idx = find(mask);
        xyzPoints(idx) = NaN;
    end
    
end

% pcshow(ptCloud);
ptCloud_cropped = pointCloud(xyzPoints);
ptCloud_cropped.Color = point_color;

hold all;
scatter3(reshape(ptCloud_cropped.Location(:,:,1),1,[]), reshape(ptCloud_cropped.Location(:,:,2),1,[]), ...
        reshape(ptCloud_cropped.Location(:,:,3),1,[]), 5, '.', 'cdata', reshape(ptCloud_cropped.Color,[] , 3));
view([0, 0, -1])
daspect([1,1,1]);
axis off

ptCloud = pointCloud(xyzPoints_no_crop);
ptCloud.Color = point_color;
figure('numbertitle','off','name', '3D point cloud','units','normalized','outerposition',[fig_x0+fig_w, fig_y0, fig_w, fig_h],'color','k');
% pcshow(ptCloud);
ax_pc_origin = axes('color','k');
hold all;
scatter3(reshape(ptCloud.Location(:,:,1),1,[]), reshape(ptCloud.Location(:,:,2),1,[]), ...
        reshape(ptCloud.Location(:,:,3),1,[]), 5, '.', 'cdata', reshape(ptCloud.Color,[] , 3));
view([0, 0, -1])
daspect([1,1,1]);
axis off
hlink = linkprop([ax_pc_poses, ax_pc_origin],{'CameraPosition','CameraUpVector'}); 
FigureRotator(ax_pc_poses);
FigureRotator(ax_pc_origin);
