from models.segmentation.STELLE_Pokemon_Segmentation.inference.inferencer import STELLEInferencer

def inference_process(request_queue):
    """
    Inference server process.

    This function is intended to be run in a separate process.
    It loads the STELLEInferencer model once and listens for image inference requests
    from a shared multiprocessing queue.

    Each request should be a tuple: (uid, np_image, response_queue)
        - uid: Unique identifier of the request.
        - np_image: Numpy image to perform inference on.
        - response_queue: Queue to send the result back to the client.

    The server continues to run until it receives a request with uid=None,
    which is used as a signal to stop the process.
    """
    print("[Inference] Loading model...")
    inferencer = STELLEInferencer()
    print("[Inference] Model loaded.")

    while True:
        uid, image, resp_q = request_queue.get()

        # Stop signal: uid is None
        if uid is None:
            break

        # Run inference on the received image
        result = inferencer.predict(image)

        # Return the result to the client via the provided response queue
        resp_q.put((uid, result))
