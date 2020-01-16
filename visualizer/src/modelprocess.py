import multiprocessing as mp

import numpy as np

from constants import *
from model import DeepLabModel

class Processor:

    def __init__(self, model_path):
        self.model = DeepLabModel(model_path)

    def segment(self, image):
        return self.model.run(image)[1]

    def process(self, image, segment_map, iterations):
        # Resize the image and segmentation map.
        image = cv2.resize(image, (RES_WIDTH, RES_HEIGHT), interpolation=cv2.INTER_LINEAR)
        seg_width = 120
        seg_height = int(RES_HEIGHT / RES_WIDTH * seg_width)
        segment_map = cv2.resize(segment_map.astype(np.uint8), (seg_width, seg_height),
                                 interpolation=cv2.INTER_NEAREST)

        # Create the delta mask.
        deltamask = np.zeros(image.shape, dtype=np.float32)

        # Construct the target.
        target = np.zeros((1, 129, 257, 19), dtype=np.float32)
        target[:,:,:,0] = 1
        target[:,:seg_height,:seg_width,:].fill(0)
        for i in range(seg_height):
            for j in range(seg_width):
                label = segment_map[i,j]
                target[0,i,j,label] = 1

        # Backpropagate the model.
        prev_gradient = None
        for iteration in range(iterations):
            modified = image + deltamask
            modified = np.clip(modified, 0, 255)

            # Get the gradient.
            gradient, result, _, loss = self.model.run_loss(modified, target)

            # Step size.
            grad_max = np.max(gradient)
            if grad_max < STEP_EPSILON:
                grad_max = STEP_EPSILON
            step_size = STEP_SIZE * (iterations - iteration) / iterations
            gradient = np.asarray(gradient) * step_size / grad_max

            # Momentum.
            if MODE == 'avg':
                if prev_gradient is not None:
                    gradient = ((1 - AVG_WEIGHT) * gradient) + (AVG_WEIGHT * prev_gradient)

            elif MODE == 'momentum':
                if prev_gradient is not None:
                    gradient = gradient + (MOMENTUM_GAMMA * prev_gradient)
                prev_gradient = gradient

            # Adjust the delta mask.
            deltamask -= np.reshape(np.asarray(gradient), deltamask.shape)
            deltamask = np.clip(deltamask, -DELTA_CLIP, DELTA_CLIP)

            # Yield a result.
            yield np.clip(result, 0, 255).astype(np.uint8), loss

class ModelProcess:

    def __init__(self, model_path):
        mp.set_start_method('spawn')

        self.inq = mp.Queue()
        self.outq = mp.Queue()
        self.process = mp.Process(target=ModelProcess.run, args=(self.outq,self.inq,model_path))
        self.process.start()

        self.event_callbacks = {}

    def stop(self):
        self.submit(COMMAND_STOP, None)

        # Flush the in queue.
        while True:
            message = self.inq.get()
            if message is None:
                break

        self.process.join()

    def tick(self):
        while not self.inq.empty():
            event, data = self.inq.get()
            if event not in self.event_callbacks:
                continue

            # Run callbacks for the event.
            for callback in self.event_callbacks[event]:
                callback(data)

    # Submit a command.
    def submit(self, command, data):
        self.outq.put((command, data))

    # Subscribe to an event.
    def subscribe(self, event, callback):
        if event not in self.event_callbacks:
            self.event_callbacks[event] = []

        self.event_callbacks[event].append(callback)

    @staticmethod
    def run(inq, outq, model_path):
        # Create a processor.
        processor = Processor(model_path)

        target_map = None
        segment_map = None

        while True:
            # Wait for the next command.
            command, data = inq.get(block=True)

            if command == COMMAND_STOP:
                # Flush the in queue.
                while not inq.empty():
                    inq.get()

                # Signal that we are done flushing.
                outq.put(None)

                # Stop the process.
                return

            if command == COMMAND_SEGMENT:
                response, image = data

                # Segment the image.
                segmented = processor.segment(image)

                outq.put((response, segmented))

            if command == COMMAND_PROCESS:
                image, segment_map = data

                # Process the image.
                for i, result in enumerate(processor.process(image, segment_map, ITERATIONS)):
                    outq.put((EVENT_RESULT, (i, result[0], result[1])))
                outq.put((EVENT_RESULT, None))

            if command == COMMAND_ACCURACY:
                target_segmentation = data[0]
                adversarial_segmentation = processor.segment(data[1])
                model_loss = data[2]

                num_correct = np.sum(target_segmentation == adversarial_segmentation)
                total = target_segmentation.shape[0] * target_segmentation.shape[1]

                accuracy_str = ''
                accuracy_str += 'Pixel accuracy: %f%%\n' % (num_correct * 100.0 / total)
                accuracy_str += 'Total cross-entropy loss: %f\n' % (np.sum(model_loss))
                outq.put((EVENT_ACCURACY, accuracy_str))

