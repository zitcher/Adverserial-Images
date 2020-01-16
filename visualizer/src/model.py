import os
import tarfile

from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import sys
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

"""
Credit to https://colab.research.google.com/github/tensorflow/models/blob/master/research/deeplab/deeplab_demo.ipynb
for code on how to load deeplab models.
"""


class DeepLabModel(object):
    """Class to load deeplab model and run inference."""
    def __init__(self, tarball_path):
        """Creates and loads pretrained deeplab model."""
        # If you want to see the names of other layers we can grab, view the graph
        # in tensorboard
        self.FROZEN_GRAPH_NAME = 'frozen_inference_graph'
        self.graph = tf.Graph()
        self.model_path = tarball_path

        graph_def = None

        # Extract frozen graph from tar archive.
        tar_file = tarfile.open(tarball_path)
        for tar_info in tar_file.getmembers():
            if self.FROZEN_GRAPH_NAME in os.path.basename(tar_info.name):
                file_handle = tar_file.extractfile(tar_info)
                graph_def = tf.GraphDef.FromString(file_handle.read())
                break

        tar_file.close()

        if graph_def is None:
            raise RuntimeError('Cannot find inference graph in tar archive.')

        with self.graph.as_default():
            tf.import_graph_def(graph_def, name='')

        self.sess = tf.Session(config=tf.ConfigProto(use_per_session_threads=True),
                               graph=self.graph)

        # get necessary parts of graph
        with self.graph.as_default():
            self.input = self.graph.get_tensor_by_name('ImageTensor:0')
            self.resized_image = self.graph.get_tensor_by_name('ResizeBilinear:0')
            self.output = self.graph.get_tensor_by_name('SemanticPredictions:0')
            self.logits = self.graph.get_tensor_by_name("logits/semantic/BiasAdd:0")
            self.target = tf.placeholder(tf.float32, shape=(1, 129, 257, 19), name='AdvTarget')
            self.adversarial_loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.target, name='AdvLoss')
            self.gradient = tf.gradients(self.adversarial_loss, self.resized_image)

            init = tf.global_variables_initializer()
            self.sess.run(init)

    def run(self, image):
        """Runs inference on a single image.

        Args:
          image: A PIL.Image object, raw input image.

        Returns:
          resized_image: RGB image resized from original input image.
          seg_map: Segmentation map of `resized_image`.
        """
        batch_seg_map = self.sess.run(
            self.output,
            feed_dict={self.input: [np.asarray(image)]})
        seg_map = batch_seg_map[0]
        return image, seg_map

    def run_loss(self, image, target):
        seg_map, gradient, adv_loss = self.sess.run(
            [self.output, self.gradient, self.adversarial_loss],
            feed_dict={
                self.input: [np.asarray(image)],
                self.target: target})
        return gradient, image, seg_map[0], adv_loss
