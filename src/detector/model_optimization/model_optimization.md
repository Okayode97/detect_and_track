# Model Optimization Notes
Note made, while trying to optimize detection model performance on a raspberry pi.

Methods to benchmark models are
- FPS, defines the number of frames processed per second, ideally we want this to be higher.
- model time, defines the time taken for the model to process incoming frame, ideally we would want to reduce the time taken to process frames.sS


## Model inference Optimization Checklist
Summary of suggestions when diagnosing model inference performance issues, taken from [here](https://pytorch.org/serve/performance_checklist.html), the suggestion provided there might be outdated but still useful to know.

Address **low handing fruits** before trying to apply model inference optimization.
- Ensure using latest and compatible libraries.
- Log system level activities to understand resource utilization, helps to know how the model inference pipeline is using the system resource. From collected logs, look to target areas with high impact on current performance.
- Quantify and mitigate the influence of slow I/O (input/output operations). Suggested to look into techniques that use async, concurrency, pipelining to hide the cost of I/O.
- For NLP models, ensure that the tokenizer is not-overpadding it's input if trained with padding set to a constant length. As the model would run un-necessarily slow on shorter sequences.

**Model inference optimization**   
- For custom model use `@torch.inference_mode()` context before calling forward pass on your model's inference method. This is more specific to pytorch but it improves inference performance.
- Explore model quantization techniques and tools that offer more sophisticated quantization methods. For inference on GPU (why only GPU, why not try using it on CPU? also why does it not suffer from accuracy loss?) it's suggested to try using fp16 precision, as it rarely suffers from accuracy loss. *Worth noting* quantization typically comes with some loss in accuracy and might not always offer significant speedup on some hardware.
- Explore optimized inference engines such as onnxruntime, tensorRT, lightseq, ctranslate-2, etc
- Explore model distillation, requires additional training data.
- For inference on CPU, try core pinning.
- Balance throughput and latency with smart batching, while meeting latency SLA try larger batch sizes to increase the throughput.
- For NLP models, sequence bucketing can improve throughput if processing batches with different lengths. *what is sequence bucketing?*


## Understanding Latency and Throughput

Notes taken from [latency vs Throughput](https://medium.com/@apurvaagrawal_95485/latency-vs-throughput-c6c1c902dbfa), [Latency vs Throughput vs Bandwith](https://www.kentik.com/kentipedia/latency-vs-throughput-vs-bandwidth/)  

- Both are fundamental performance metrics in software system, measuring distinct aspect of the systems operations.
- *Latency*, is the time taken for data to travel from source to destination. A system with a lower latency would have little delay when responding to a request. This metric is particularly important for system which require real or near real interactions such as online gaming, video conferencing, data transfer, high frequency trading. In the context of my application, latency would be the time taken from sending the put request to the model server to receving a responce back.

- *Throughput*, represent the volume of data that can be processed by a system within a given time frame. It's measurred in data units per time. *"A higher throughput denotes greater data processing capability"*. Throughput is critical in system that manages substantial data volumes. Throughput depends on several factors, such as the networks physical infrastructure, the number of concurrent users and type of data being transferred.

- Understanding the key difference between latency and throughput. Throughput asks the question of, how much data can we process in a given time and latency focus more on overal speed and delays. One thing to note is that a high throughput does not necessarily mean a low latency, for example we can have systems with
    - **Low throughput with low latency**. Where we can have highly responsive systems that process only small amounts of data. An example would be a lighweight IoT device which responds instantly to commands but can only process a limited amount of data per second.
    - **Low throughput with high latency**. Systems with these metrics are highly overloaded and have been poorly optimized. They would be very unresponsive and can only process limited amounts of data.
    - **High throughput with low latency**. This is the ideal state we generally want how systems to perform at. With a high throughput we can process a large volumes of data quickly, with minial delay in transmitting the data from source to destination. 
    - **High throughgput with high latency**. Where we process large volumes of data over time, but still have each individual data take a long time to travel from/to source/destination. An example would be a satellite communication like where data can be transferred at a high rate but the signal travels time to/from the satellite introduces significant delay.

**Improving Latency** Focuses on reducing the time taken for system to respond to request or perform an operation. While **Improving Throughput**, focues on increasing rate at which the system can process data.
- Use multi-threading, asynchronous programming or parallel processing to execute task concurrently and make use of available resources more effectively.
- Cache frequently accessed data or computation to reduce repeated calculations or database queries.
- Distribute incoming request evenly across multiple servers to prevent overloading and reducing response times.
- Vertical and Horizontal Scaling. (Vertical scaling) Upgrade hardware resources such as CPU, memory and storage, or (Horizontal scaling) add more instances of servers to distribute loads.
- Optimize Algorithms and Data structures, to minimize computation time and memory usage.
- Use Hardware accelerations to offload computational task and improve performance.
- Reduce reliance on external services or other dependencies that may introduce latency.
- Use batch processing to reduce overhead in processing each data.
- Apply compression before transferring data over network to reduce bandwidth usage and improve transfer speed.
- Use faster communication protocols. Suggested that protocol such as HTTP/2 can reduce latency through features like multiplexing and header compression.


W.R.T latency and throughput for my application, actionable tasks i can take to improve throughput and latency
- Scale up horizontally, possibly using kubernetes to deploy muitple instances of the model servers. Using this approach would also require looking into use of multi-threading, asynchronous programming and communciation, to determine how to handle receving multiple responses from different sources.
- Look into optimizing code to process incoming data.


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



## Optimized inference engines 

## Model distillations

## Core pinning

## Qnnpack with Pytorch




## Resources
- [Model Quantization 1: Basic Concepts](https://medium.com/@florian_algo/model-quantization-1-basic-concepts-860547ec6aa9#:~:text=Quantization%20of%20deep%20learning%20models,training%2C%20finetuning%20and%20inference%20stages.)
- [Pytorch's Quantization Recipe](https://pytorch.org/tutorials/recipes/quantization.html)