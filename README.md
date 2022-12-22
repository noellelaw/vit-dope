# ViTDope: An exploration of vision transformers for deep object pose estimation
![Alt text](https://github.com/noellelaw/vit-dope/blob/main/figures/model.png?raw=true "ViTDope Model Architecture")
ViTDope is a vision-transformer based model for detection and 6-DoF pose estimation of **known objects** from an RGB camera. The network has been trained on the cracker box YCB objects. 

Vision transformers are a powerful tool for a multitude of vision recognition tasks. This work explores the vision transformer in context of 3D object pose estimation for robotic manipulation tasks. Due to limitations in real-world data, a domain-randomized synthetic dataset for this task is created through leveraging NIVIDIA's Isaac Simulator. The [dataset](https://www.kaggle.com/datasets/noellelaw/dome-mesh-ycb) is composed of ~55k RGB images and ground-truth labels for two separate benchmark objects for robotic manipulation tasks, namely the mustard bottle and the cracker box. A vanilla transformer backbone and classic decoder model is then trained using a mixture of photorealistic and domain randomize synthetic data to identify the 2D bounding cuboid of an object of interest, when provided an RGB image.  This exploration found that while ViTDope achieved execution speed on par with current state-of-the-art models, it could not converge within a reasonable amount of time to achieve comparable accuracy. Visualization of model results at epoch 62 for cracker box, in addition to how the model learns can be seen below.

The model can be trained using the ViTDope_training.ipynb, and inference can be run through the ViTDope_inference.ipynb notebook. The repo is set up as folows:
* scripts/ contains files for processing data and retrieving ground truth belief and affinity maps from provided images.
* models/ contains the backbones and heads explored.
* utils/ provides code for loading weights.
* core/ provides code for postprocessing tasks.

The dataset can be found [here](https://www.kaggle.com/datasets/noellelaw/dome-mesh-ycb).

Weights for cracker box at epoch 62 can be found [here](https://drive.google.com/file/d/10O-LluXiJHJHAKuDOy6ssuE21Q33jKXn/view?usp=sharing).

Predicted 3D object pose estimation projected on original input image: 
![Alt text](https://github.com/noellelaw/vit-dope/blob/main/figures/viz.png?raw=true "Predicted 3D object pose estimation projected on original input image.")


ViTDope Model visualization of belief map vertex outputs throughout training:
![Alt text](https://github.com/noellelaw/vit-dope/blob/main/figures/progress.png?raw=true "ViTDope Model visualization of belief map vertex outputs throughout training.")

