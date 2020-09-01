function [part_list] = get_cad_part_list(file_list_path)
% read the part list line by line and return it in a cell
% array
fid = fopen(fullfile(file_list_path,'part_list.txt'));
next_line = fgetl(fid);
i=1;
while ischar(next_line)
    part_list{i} = next_line;
    next_line = fgetl(fid);
    i = i + 1;
end
fclose(fid);
end