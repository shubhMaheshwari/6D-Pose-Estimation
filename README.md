# 6D-Pose-Estimation 
    6D-pose estimation of an object in a RGB image refers to the task of estimating the six degrees of freedom in 3D space. 
    This involves determining the 3D rotation and translation to orient the 3D object into the same location in the image. The report outlines the algorithm, compares baselines and, provides a custom solution for object pose estimation.    


## Alogrithm 
    - Load image 
    - Get/Predict Segmentation Mask 
    - Get point cloud using the depth image 
    - Load 3D model
    - Estimate 6D parameters for each object using either
        - Chamfer + ICP
        - Pointnet 
        - Pointnet + ICP 
        - Custom 
    - Render Result   



# A. Installation

1. Download the codebase.
    ```
        git clone --recursive https://github.com/shubhMaheshwari/6D-Pose-Estimation.git 
    ```

2. Download the [dataset](https://drive.google.com/drive/folders/196tuNaIivzsfsKOdrNy9tMrENJpMeFBW?usp=drive_link).

2. Install python packages. 
    ```
    pip install torch torchvision tqdm
    ```

3. Optional: Download [blender](https://www.blender.org/download/).

*Note- Raise an issue if you are facing trouble installing any of the above packages.*



## B. Inference   
1. Update dataset path in src/utils.py

2. Get prediction
    ```
    python3 src/pose_estimation.py # For complete test dataset to get metrics
    ```
    Or 
    ```
    python3 src/pose_estimation.py \
        <sample-filepath> # Specific file 
        --force or -f # Rewrite results 
        --image # Location to save images/plots 
    ```



## C. References
- Pointnet
    ```
    @article{qi2016pointnet,
    title={PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation},
    author={Qi, Charles R and Su, Hao and Mo, Kaichun and Guibas, Leonidas J},
    journal={arXiv preprint arXiv:1612.00593},
    year={2016}
    }
    ```
