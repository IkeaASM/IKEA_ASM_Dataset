function FV = my_plyread(file_name)
    FV_data_structure = plyread(file_name);
    FV.faces = cell2mat(FV_data_structure.face.vertex_indices) + 1 ;
    FV.vertices = [FV_data_structure.vertex.x, FV_data_structure.vertex.y, FV_data_structure.vertex.z];
end