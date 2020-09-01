function annotation = ReadAnnotationFile(file_name)
annotation = [];
fid = fopen(file_name);
i=1;
while ischar(fgetl(fid)) % reas part line 
    fgetl(fid); %read translation line
    annotation{i}.T = str2double(strsplit(fgetl(fid), ','));
    annotation{i}.T = annotation{i}.T(~isnan(annotation{i}.T))';
    fgetl(fid); %read rotation line
    annotation{i}.R = str2double(strsplit(fgetl(fid), ','));
    annotation{i}.R = annotation{i}.R(~isnan(annotation{i}.R))';
    i = i+1;
end
end