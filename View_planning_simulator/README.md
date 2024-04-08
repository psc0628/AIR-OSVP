# View_planning_simulator
This is the view planning simulation system for object reconstruction.  

## Installion
These libraries need to be installed: opencv 4.4.0, PCL 1.9.1, Eigen 3.3.9, OctoMap 1.9.6, Gurobi 10.0.0.  
Note that Gurobi is free for academic use.  
Our codes can be compiled by Visual Studio 2022 with c++ 14 and run on Windows 11.  
For other system, please check the file read/write or multithreading functions in the codes.  

## Prepare
Make sure "model_path" in DefaultConfiguration.yaml contains these processed 3D models.  
You may find our pre-porcessed 3D models at [Kaggle](https://www.kaggle.com/datasets/sicongpan/ma-scvp-dataset).  
Or use [this sampling method](https://github.com/PointCloudLibrary/pcl/blob/master/tools/mesh_sampling.cpp) to process your own 3D object model from *.obj or *.ply to *.pcd.  
The "pre_path" in DefaultConfiguration.yaml is the results saving path.  
Follow [MA-SCVP](https://github.com/psc0628/MA-SCVP) to setup baselines of nbvnet, pcnbv, mascvp. The "nbv_net_path", "pcnbv_path", "sc_net_path" should be the downloaded network paths.  

## Main Usage
1. The mode of the system should be input in the Console.  
2. Input object names that you want to test and -1.

## View Planning Comparsion
1. Run mode 1 first to obatin number of visible points.  
2. Run mode 0 to test proposed method and baselines. Note run python scripts in poco_pointr folder simultaneously.  
3. Run mode 2 to get pointclouds for POCO meshing.  
4. Run "python mesh_gen_all.py" to generate mesh.  

### Change in methods and setup
Change 1352-1364 lines in main.cpp for different methods with rotate_states and init_views (default run our method).




