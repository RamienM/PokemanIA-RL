import uuid

class SharedInferencer:
    """
    SharedInferencer is a lightweight client interface that communicates
    with a separate inference server process via multiprocessing queues.

    It sends image data along with a unique ID to the server and waits
    for a response on a shared response queue.
    """

    def __init__(self, request_queue, response_queue):
        """
        Initialize the shared inferencer.

        Args:
            request_queue (mp.Queue): Queue to send inference requests.
            response_queue (mp.Queue): Shared queue to receive responses.
        """
        self.request_queue = request_queue
        self.response_queue = response_queue

    def predict(self, np_image):
        """
        Send an image to the inference process and wait for the prediction result.

        Args:
            np_image (np.ndarray): Image to segment (NumPy array format).

        Returns:
            output: Segmentation result returned by the inference process.
        """
        # Generate a unique ID for this request
        uid = str(uuid.uuid4())

        # Send the request: (uid, image, response_queue)
        self.request_queue.put((uid, np_image, self.response_queue))

        # Wait for the result associated with this UID
        _, output = self.response_queue.get()

        return output
