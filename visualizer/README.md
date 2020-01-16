# Visualizer

## About
The visualizer takes in a camera feed and runs an interactive program to show the capabilities of the project.

## How to Run
To run the visualizer, simply run `main.py` with Python 3:

```
python3 main.py
```

## Controls
To toggle live segmentation, press `L`. **WARNING**: Running this will likely slow your computer down to a halt without a powerful GPU.

To capture an image and segment it, press `C`.

To open a file and segment it, press `O`.

To export the segmented image, press `E`.

To fill the canvas with a selected legend color, press `F`.

To draw on the canvas with a selected legend color, click and drag.

To generate an adversarial image using the modified canvas, press `P`.

To save the generated adversarial image, press `S`.

To run the generated adversarial image in the segmenter, press `T`.

To quit, press `Q`.
