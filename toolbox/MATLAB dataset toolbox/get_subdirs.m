function subdirs = get_subdirs(path)
subdirs = dir(path);
subdirs = subdirs([subdirs(:).isdir]==1);
subdirs = subdirs(~ismember({subdirs(:).name},{'.','..'}));
end