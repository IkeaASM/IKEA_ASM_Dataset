source /home/djcam/Documents/code/human_pose/VIBE/vibe-env/bin/activate
python3 run_inference.py --run_vibe
deactivate

source /home/djcam/Documents/code/human_pose/human_dynamics/venv_hmmr/bin/activate
python3 run_inference.py --run_hmmr
deactivate

python3 run_inference.py --run_openpose --camera_id dev1 --device 0
python3 run_inference.py --run_openpose_staf --camera_id dev3 --device 0
