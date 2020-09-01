rclone copy --progress --transfers 8 CloudStor:"/Shared/ANU IKEA DataSet/IKEA Dataset videos compressed" ~/path_to_local_dir/
cd  ~/path_to_local_dir/
cat part* > ikea_dataset_videos.tar.gz
tar -xvf ikea_dataset_videos.tar.gz



