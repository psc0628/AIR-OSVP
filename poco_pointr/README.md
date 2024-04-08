# poco_pointr_scripts

This folder contains scripts to interact with POCO and Pointr.  

## Installion

Follow ["POCO"](https://github.com/valeoai/POCO) to install the environment and [checkpoint](https://github.com/valeoai/POCO/releases/download/v0.0.0/ShapeNet_3k.zip).  The checkpoint should be put to "POCO/checkpoint.pth".  
Follow ["PoinTr"](https://github.com/yuxumin/PoinTr) to install the environment. Our dataset and checkpoint can be asscess at [Kaggle](https://www.kaggle.com/datasets/sicongpan/air-osvp-dataset).  The checkpoint should be put to "pointrVP/trained_network/best_ckpt.pth".

## Usage

1. Move all files in our POCO folder to installed POCO.  
2. Run the script under POCO environment while running the simulator.   
```
python run_test_all_poco.py
```
3. Move all files in our pointrVP folder to installed PoinTr and rename the folder to pointrVP.  
4. Run the script under PoinTr environment while running the simulator.  
```
python run_test_all_pointrvp.py
```

### Note

Change object_name.txt for different test objects.  
Change rotate_ids and first_view_ids in scripts for different tests.