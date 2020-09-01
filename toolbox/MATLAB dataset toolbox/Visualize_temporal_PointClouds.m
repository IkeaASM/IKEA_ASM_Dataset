%Visualize and save point cloud visualization with tilting camera effect
clear all
close all
clc

path_to_processed_dataset = '~/Datasets/ANU_ikea_dataset_processed';
% get the folder contents
categories = get_subdirs(path_to_processed_dataset);
point_cloud_path_list = [];
for i = 1:length(categories)
    scans = get_subdirs(fullfile(categories(i).folder, categories(i).name));
    for j = 1:length(scans)
        devices = get_subdirs(fullfile(scans(j).folder, scans(j).name));
        for k = 1:length(devices)
            point_cloud_paths = fullfile(fullfile(devices(k).folder, devices(k).name), 'point_clouds');
            if exist(point_cloud_paths, 'dir')
                if isempty(point_cloud_path_list)
                    point_cloud_path_list = {point_cloud_paths};
                else
                    point_cloud_path_list=[point_cloud_path_list; {point_cloud_paths}];
                end
            end
        end
    end 

end

for point_cloud_path = point_cloud_path_list.'
    figure('visible','off');
    point_cloud_path  = point_cloud_path{1};
    phi = 0;
    theta = 270;

    video_writer = VideoWriter(fullfile(point_cloud_path,'point_cloud_video.avi'));
    open(video_writer);
    file_list = dir(fullfile(point_cloud_path, '*.ply'));
    i=1;
    for point_cloud_file = file_list.'
        ptCloud = pcread(fullfile(point_cloud_file.folder, point_cloud_file.name));
        if i == 1
            cog = mean(ptCloud.Location);
        end

        ptCloud = pctransform(ptCloud,repmat(-cog, [size(ptCloud.Location,1), 1]));
        ax_pc = pcshow(ptCloud);
        phi = 15*sin((i-1)*pi/120)  - 10;

        r = 5000;
        CamX = r*sind(theta)*sind(phi); CamY=r*cosd(theta); CamZ=r*sind(theta)*cosd(phi);
        UpX=-cosd(theta)*sind(phi);UpY=sind(theta);UpZ=-cosd(theta)*cosd(phi);
        set(gca,'CameraPosition',[CamX,CamY,CamZ],'CameraTarget',[0 0 0],...
        'CameraUpVector',[UpX,UpY,UpZ],'CameraViewAngle',40);
        set(gcf,'color','k');
        daspect([1,1,1]);
        axis off
        %add code to capture the frame
        F = getframe(gcf);
        writeVideo(video_writer,F);
        i = i+1;
    end 
    close(video_writer);
    close(gcf);

end
