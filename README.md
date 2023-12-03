# 6D-Pose-Estimation 
    6D pose estimation of an object in a RGB image refers to the task of estimating the six degrees of freedom in 3D space. 
    This envolves determing the 3D rotation and translation to orient the 3D object to the image.   

## A. Installation

1. Download the codebase
    ```
        git clone --recursive https://github.com/shubhMaheshwari/6D-Pose-Estimation.git 
    ```
2. Python packages 
    ```
    pip install torch torchvision
    ```

3. Optional: Download [blender](https://www.blender.org/download/)

*Note- Raise an issue if you are having trouble installing any of the above packages*



## B. Inference   
1. Update dataset path in src/utils.py

2. Get prediction
    ```
    python3 src/pose_estimation.py # For complete test dataset
    ```
    Or 
    ```
    python3 src/pose_estimation.py \
        <sample-filepath> # Specific file 
        --force or -f # Rewrite results 
        --image-dir # Location to save images/plots 
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
