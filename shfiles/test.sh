# Terminal 1
source devel/setup.bash
roslaunch map_generator sim_test.launch

# Terminal 2
source devel/setup.bash
roslaunch lidar scanner.launch

# Terminal 3
conda activate orp
cd scripts
python infer.py