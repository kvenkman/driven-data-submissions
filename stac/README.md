The best scoring model comprised of the following : 
- A Unet with a resnet101 backbone, initialized with imagenet weights
- A 4 channel input, comprising of vv, vh, dem and ose, where:
- dem is a [0, 1] normalized image of the nasadem image, masked for values below 0.5, and
- ose is weighted combination of occurrence (0.25), seasonality (0.25) and extent (0.5), each normalized to [0, 1], and the combination masked for values greater than 0.5
- The model uses a 50/50 split of dice and cross-entropy loss, and does not weight the classes
- The data was split into a 80/20 train/validation split, where the split was conducted on the unique chip ids present in the provided dataset
