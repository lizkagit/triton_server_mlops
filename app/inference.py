import tritonclient.grpc as grpcclient
from config import TRITON_URL

class TritonInference:
    def __init__(self):
        self.client = grpcclient.InferenceServerClient(url=TRITON_URL)

    def run_inference(self, model_name, batch_input, output_names=["output0"]):
        inputs = [grpcclient.InferInput("images", batch_input.shape, "FP32")]
        inputs[0].set_data_from_numpy(batch_input)
        
        outputs = [grpcclient.InferRequestedOutput(name) for name in output_names]
        
        response = self.client.infer(model_name, inputs=inputs, outputs=outputs)
        return {name: response.as_numpy(name) for name in output_names}

triton_service = TritonInference()