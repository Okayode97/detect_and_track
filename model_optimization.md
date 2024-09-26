# Model Optimization Notes
Note made, while trying to optimize detection model performance on a raspberry pi.

Methods to benchmark models are
- FPS, defines the number of frames processed per second, ideally we want this to be higher.
- model time, defines the time taken for the model to process incoming frame, ideally we would want to reduce the time taken to process frames.sS

## Qnnpack with Pytorch

## Model Quantization

Quantization is a `memory optimization techniques that reduces memory space albeit at an accuracy cost`. It is an essential technique needed for model training, finetuning and inference stages.   
In deeplearning `quantization is used to convert high precision floating-point numbers into low precision numbers`, representing high precision numbers with fewer bits. Quantization has the benefits of
- Reducing model size & memory consumption during inference
- Reduced memory consumption during inference, increases inference speed, while maintaining roughly the same accuracy as the original model.

### Pytorch quantization
According to pytorch's quantization recipe, There are three approaches to quantize a model
- `post training dynamic/static quantization`
- `quantization aware training`
Alongside quantizing custom models, pytorch provides a collection of quantized models which can be accessed through `torchvision.models.quantization..`

Post training dynamic quantization   
This approach `quantizes the model weights before inference` and `dynamically quantizes it's activations during inference`. This approach `requires no calibration dataset`. On pytorch this approach is `only supported for linear and LSTM layers`. The range in which to quantize the model's activation is determined by the data seen during inference.

Post training static quantization
This approach `quantizes both the model weights and activation`, it's typically done offline and before running inference. This approach can `require calibrating with representative dataset to determine the dynamic range of values for the activation`.

Quantization aware training (QAT)
This approach `applies quantization effects into the training process for a model`, allowing it to `adjust and learn under lower precision constraint (quantized conditions)`. So the result is minimized accuracy loss during quantization.

QAT workflow during training.
- in forward pass of model training, quantization effects are applied by simulating conversion of weights and activation to INT8 while keeping gradient updates at FP32 during back propagation.
- The final model is quantized and used for inference in lower precision.

Benefits
- Model retains more accuracy after quantization, as it's learned to operate under quantized conditions during training.
- QAT is an approach to ensure final model performs well in some cases where PTQ might struggle.

Issues/Gotchas
- QAT can be used for models that are sensitive to precision loss, such that any loss in accuracy after quantization is unacceptable.
- QAT requires additional training time & increases the complexity of the training process.



## Just in time compilation (JIT)

## Resources
- [Model Quantization 1: Basic Concepts](https://medium.com/@florian_algo/model-quantization-1-basic-concepts-860547ec6aa9#:~:text=Quantization%20of%20deep%20learning%20models,training%2C%20finetuning%20and%20inference%20stages.)
- [Pytorch's Quantization Recipe](https://pytorch.org/tutorials/recipes/quantization.html)