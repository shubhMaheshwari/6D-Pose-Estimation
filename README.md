# Robust 6D-Pose-Estimation using Pointnet  
    6D-pose estimation of an object in a RGB image refers to the task of estimating the six degrees of freedom in 3D space. 
    This involves determining the 3D rotation and translation to orient the 3D object into the same location in the image. The report outlines a simple statergy to solve this problem. We provide a novel algorithm, compare baselines on a custom synthetic RGBD dataset. The project was completed for the CSE275 - Deep learning for 3D data course.    



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

## Contributions 
    - End-to-end diffrentiable pipeline for ICP 
    - Mesh laplacian for consistency (can also used a ED Warpfield)
    - Use point net to predict a binary mask on the point cloud and mesh samples vertices. Only vertices greater than a theshold can be used to update ICP this is similar  
        - P(z_p={0,1} | Point cloud, mesh )
        - We do not require the occluded surface of the complete object for ICP , can also remove outliers if trained correcly. 
        - For now additional constraint is that P(z_p=1 | Pcd, mesh ) = 1 for point cloud since we have synethetic data. 
        - Note there lies exist a particular 3D to 2D projection matrix that would give the desired result. 

## ICP Input: 
1. XYZ, RGB, L2 skeleton radius (psuedo medial axis)


# A. Installation

1. Download the codebase.
    ```
        git clone --recursive https://github.com/shubhMaheshwari/6D-Pose-Estimation.git 
    ```

2. Download the [dataset](https://drive.google.com/drive/folders/196tuNaIivzsfsKOdrNy9tMrENJpMeFBW?usp=drive_link).

2. Install python packages. 
    ```
    pip install torch torchvision tqdm pycollada
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


## C. Debugging 
1. Visualize correspondence in icp.py (uncomment polyscope rendering code)
2. Use VSCode


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
