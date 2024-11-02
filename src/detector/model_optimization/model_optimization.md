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


## TorchScripting
P.s I've reduced my full note on torchscripting to cover what i think is essentiall

### What is Torchscript and why is it needed
At a glance, `TorchScript is an intermediate representation of a PyTorch model that can be run in an high performance environment` such as C++. It is particularly useful in getting a pytorch model to run in a production environment.
We can convert our model to a torchscript using the `torch.jit.trace` and `torch.jit.script` function.


### Why convert a model into torchscript version of it's self?
- TorchScript code can be `invoked in it's own interpreter`, which `does not acquire the python Global interpreter Lock`, so many `requests can be processed on the same instance simultaneously`.
- The TorchScript format,  `allows us to save the whole model to disk` and `load it into another environment`, so we `don't need to re-write the model architecture` and then re-load it's weight we can just load up the entire model.
- TorchScript gives us a `intermediate representation` in which `we can do complier optimization on` the code to provide `more efficient execution`.
- TorchScript allows us to `interface with many backend/device runtime` that require a broader view of the program.

### **Tracing**
- `torch.jit.trace` can be used to convert a function/module into a Torch script version of the function/model.
- Using trace, you provide the model and sample input as argument.
- The input would be fed through the model as in regular inference and the executed operation would be traced and recorded into Torchscript.
- The logical structure of the resulting torch script model would be frozen into the path taken during the sample execution.

**When would you use tracing over scripting**
- Tracing is `useful if you are unable to modify the model code` for example if you don't have access or ownership.
- Tracing is `useful if the model uses unsupported Python or Pytorch functionality which scripting does not support`
- Tracing is `useful in baking in architectural decision or specific logic.`


Worth noting that tracing can return a model whose behaviour is fixed based on the initial input and operations ran for it.
- tracing is `not useful if your model uses data-dependent control flow such as loop or conditional code`
- tracing `does not preseve language specific data structures`.


### **Script Complier**
- `torch.jit.script` can be used to convert a function/module into a Torch script version of the function.model.
- Using script, you would provide the model as an argument and the generated script would be a static inspection of the nn.Module contents.


**When would you use scripting over tracing**
- Scripting `captures both the operation and full conditional logic` of a model.
- It's typically a good starting point, particularly `useful if the model does not need any unsupported pytorch functionality` and has `logic which are restricted to the supported subset of python functions and syntax.`
- With `scripting an export would either fail for a well defined reason or succed without warning`.
- Scripting is `able to capture conditional logic and control flow which tracing is not able to capture`.
- Scripting can be useful if we are working with code which we own and can be edited, `also it provides support for a subset of the python language` so we can use functions like print statements for debugging issues.


### **Mixing use of scripting and tracing**
- Scripting captures the entire structure of a model, so it makes sense to `apply scripting only to small models or parts of a models`.
- Typically you would apply a mix of scripting and tracing to optimize which part of the model you actually want to capture and which part you don't need.
- For example, you can `use JIT script only on control flow section and trace on all other section`. With using JIT script you would need to keep the control flow section as small as possible.

First of all
- Neither scripting/nor tracing works if the model is not even a proper single-device connected graph representable in TS format.

Arguments for using `tracing as default and scripting only when necessary` for deployment of non-trival models such as detection & segmentation models.
- Using tracing would not damage the code quality
- the main limitation of tracing (lack of support for data-dependent control flow) can be addressed using scripting. 

### **Cost of using scripted version**
- Pytorch's compiler has good support for most basic syntax, but medium to no support for anything more complicated. So this restricted support limits how users can write code. So a `big consequence of scripting is in the code quality`.
- `Typically most projects would choose to stay on the safe side to make their code scriptable/compilable` by the scripting compiler. They would do this by `using only the basic python syntax`.
-  As mentioned the `result of this is reduction in code quality`, as `developer would stop making abstractions and exploring useful language features` due to concerns in scriptability
- *common hack would be to keep a version of the code that is purely for scripting, but this is still difficult to maintain*

Does that mean that you would always sacrifice code quality for scriptability
- Not necessarily, it is possible to make most models scriptable without removing any abstraction, although this is not an easy task. Solutions to this would requires custom syntax fixes to the compiler, finding creative workarounds and devloping hacky patches.

Further limitations with scripting
- resulting code can be very brittle, the code might work and compile for now but as the project grows and expand, abstractions would become more necessary
- ugly code can start to accumlate and due to complier syntax limitation, abstractions cannot be made to clean up the ugliness.


### **Cost of traceability**
The main requirements for traceability include

- `Inputs/Outputs have to be a union of Tensors, Tuple[Tensors], Dict[str, Tensors] or their nested combination`. Similar restrictions exist in scripting as well. But `with tracing these constraints do not apply to submodules`, meaning `submodules can use any input/output format that python support`. Only the top-level modules are required to use the constraint format.
- So to address this requirements with the input/output you would typically write a wrapper around the input/output. A wrapper can be written around the model input and another wrapper can be written around the model's output.
- Expressions like `tensor.size(0), tensor.size()[1], tensor.shape[2] return intergers in eager mode (yes i've checked) but Tensors in tracing mode`, these `difference are only necessary so that during tracing, the shape computation can be captured as symbolic operations in the graph`. Effectively what this means is that, function associated with tensors would return tensors and not intergers, thic an cause a `model to be untraceable if it assumes returned values are integers and not tensors`. Although this can be easily fixed. 
- Those are `all the requirements for traceability`, `most importantly, any Python syntax is allowed in the model implementation`, because tracing does not care about syntax at all.

so we should all be using tracing then??
- Well no... tracing has it's faults too.
- `Tracing is not generalizeable to other input`, it freezes to the logic seen during trace.
- Any `intermediate computation of a non Tensor type would be captured as constants`, `using the value observed during tracing`. Even if you wrap the intermediate computation to return a tensor type it would still be represented as a constant tensor. This `issue occurs in any code that converts torch types (Tensors) to numpy or python primitives`.
- lastly any operators that accept a `device argument will remember the device used during tracing`, so tracing would `not generalize to inputs on different devices`, these types of generalization is almost never needed. Because deployments would usually have a target device. `Solution for these problems would be to trace on the expected target device`

Different ways to mitigate issues with tracing is to
- `pay close attention to Tracer Warnings`
- `Use unit tests to verify that the exported model produces the same output` as the original eager-model. If generalization across different input shape is needed in your unit tests make sure the inputs to the two models have different shapes.
- `Avoid un-necessary special conditions` within the model code which are data dependent.
- `Use symbolic shapes` which can be captured in the generated graph, for any custom class implement a `.size` method or use a `.__len__()` method instead of `len()`. And be wary of the fact that intermediate computation of a non tensor type would be captured as constants.

The best solution, as stated before is to `apply tracing to the majority of the code` and `use scripting only when necessary`.
- We can apply mix of scripting and tracing by using the decorator `torch.jit.script_if_tracing` which would script the desired part of the code if you are tracing the entirity of the model code.
- Any function decorated by `torch.jit.script_if_tracing` would have to be a pure function that does not contain modules
- For any submodule in our code that can not be traced, we can script that specific sub module before tracing. it's recommended to use the `torch.jit.script_if_tracing` inside the submodule where it's needed

Another distinction between scripted and traced models is
- A `traced modules would only support forward()` but a `scripted module can have multiple methods`

### **Properties of TorchScripted Modules & Other Gotchas**
- We can view the generated code and graph representation of the resulting model, using properties of the torchscripted module.
- `scripted_model.graph` displays the optimized graph representation of the model.
- `scripted_model.code` returns the python-syntax interpretation of the optimized torchscript code.

**Issues with jit.trace**

Device pinning, Performance and Portability.
- device pinning, `torch.jit.trace records and freeze conditional logic`, it `will also trace and make constant values resulting from the logic` this includes device constants. So without specifying the device, it would default to CPU as the inserted device.
- What this means is `we can have costly memory transfers and synchronization` if the `traced tensor is pinned on a different device` compared with the rest of the model.

## Optimized inference engines 

## Model distillations

## Core pinning

## Qnnpack with Pytorch



## Resources
- [Model Quantization 1: Basic Concepts](https://medium.com/@florian_algo/model-quantization-1-basic-concepts-860547ec6aa9#:~:text=Quantization%20of%20deep%20learning%20models,training%2C%20finetuning%20and%20inference%20stages.)
- [Pytorch's Quantization Recipe](https://pytorch.org/tutorials/recipes/quantization.html)
- [Introduction to TorchScript](https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html#mixing-scripting-and-tracing)
- [YouTube: TorchScript and PyTorch JIT | Deep Dive](https://www.youtube.com/watch?v=2awmrMRf0dA)
- [YouTube: PyTorch - torch.jit Optimize your model](https://www.youtube.com/watch?v=HLzPOhjZ4wc)
- [Mastering TorchScript: Tracing vs Scripting, Device Pinning, Direct Graph Modification](https://paulbridger.com/posts/mastering-torchscript/)
- [Apple Model Tracing](https://apple.github.io/coremltools/docs-guides/source/model-tracing.html)
- [TorchScript: Tracing vs Scripting](https://ppwwyyxx.com/blog/2022/TorchScript-Tracing-vs-Scripting/)
- [Loading a TorchScript model in C++](https://pytorch.org/tutorials/advanced/cpp_export.html)
- [Torchscript Language Reference](https://pytorch.org/docs/stable/jit_language_reference.html)
- [TorchScript](https://pytorch.org/docs/stable/jit.html#mixing-tracing-and-scripting)
- [TorchScript Unsupported PyTorch Constructs](https://pytorch.org/docs/stable/jit_unsupported.html)