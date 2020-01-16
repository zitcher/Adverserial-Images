
## Team: The Four Corners of Eugene Charniak's Bowtie

# Team members:
Zachary Hoffman (Top Left)
Jonathan Lister (Top Right)
Samantha Cohen (Bottom Left)
Ruiqi Mao (Bottom Right)

# What is your project idea?

We wish to generate adversarial examples for pre-trained semantic image segmentation networks. We want to be able to take arbitrary segmentation regions and apply them to images.

# What data will you use?

As we are using a pre-trained model, we do not need training data. We will use our own images to feed to the model and modify.

# Which software/hardware will you use?

Software:
The pre-trained model that we are planning on using is the Google DeepLab model: https://github.com/tensorflow/models/tree/master/research/deeplab, which is written in Keras. As such, we will be working in Python with Tensorflow for this project.

Pretrained models can be found at:
https://github.com/tensorflow/models/blob/master/research/deeplab/g3doc/model_zoo.md

Hardware:
Our laptops and GCP depending on the length of time it takes to run backpropagation on these image.

# Skills of team members? Who will do what?

Three of our team members (Samantha, Zachary, and Ruiqi) took CSCI 1470, Jon, Ruiqi, and Zach have taken CS1420. Zach and Ruiqi took 1410.

# Potential Jobs:
Cost functions
Data pipeline
Get the network working

Zach: Network debugging/get the network working.

Ruiqi: Build a data pipeline for capturing images and feeding them to the network.

Jon: Generate interesting adversarial examples for networks of interest.

Sam: Refine process for generating adversarial images (working out cost function).
