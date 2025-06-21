import numpy as np
import onnxruntime

def read_input():
    with open("input-1.txt", "rt", encoding="utf-8") as f:
        lines = f.readlines()
        float_values = [np.float32(float(l)) for l in lines]

    return np.array(float_values).reshape(1, 1, 384, 384)

if __name__ == '__main__':
    model_file = r"d:\models\PP-FormulaNet_plus-S.onnx"

    options = onnxruntime.SessionOptions()
    providers = ["CUDAExecutionProvider"]
    onnx_session = onnxruntime.InferenceSession(model_file, options, providers=providers)

    print("")
    for index in range(5):
        processed_input = read_input()
        print("iteration: ", index)
        print("input.shape: ", processed_input.shape)
        output = onnx_session.run(output_names=["fetch_name_0"], input_feed={"x": processed_input})
        print("output.shape: ", output[0].shape)
        print("*"*31)
