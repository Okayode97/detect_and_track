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
- Reduced memory consumption during inference, increases inference speed.
-  we can reduce the size of the model and it's memory consumption during inference.

## Just in time compilation (JIT)

## Resources
- [Model Quantization 1: Basic Concepts](https://medium.com/@florian_algo/model-quantization-1-basic-concepts-860547ec6aa9#:~:text=Quantization%20of%20deep%20learning%20models,training%2C%20finetuning%20and%20inference%20stages.)
