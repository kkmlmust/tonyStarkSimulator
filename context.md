# Project Context: Real-Time Hand Gesture Control System

I am building a personal ML/computer vision project focused on controlling a computer and potentially games using hand gestures captured by a webcam in real time.

The general idea is similar to a sci-fi / Tony Stark-style interface: the webcam watches my hand, a computer vision pipeline detects and tracks the hand, an ML model recognizes what gesture I am making, and the system converts that gesture into an action on the computer.

For example:

* Making a fist → press a keyboard key such as `D`
* Pointing with a finger → move the mouse cursor
* Moving the hand → control the mouse position
* Other predefined gestures → trigger different keyboard or mouse actions

The eventual goal is to use this system for controlling games such as Dota 2, but the project should first be developed as a general-purpose desktop hand-control system rather than being tightly coupled to one specific game.

## Main technical idea

The intended pipeline is approximately:

Webcam
→ OpenCV
→ MediaPipe hand tracking
→ Hand landmarks
→ preprocessing / feature extraction
→ my own ML model
→ gesture classification
→ keyboard/mouse controller

MediaPipe would primarily be used for detecting the hand and extracting hand landmarks. The ML part should be my own implementation/model, most likely using PyTorch.

For example:

21 hand landmarks can be extracted from the hand, with coordinates such as:

`x, y, z`

These landmarks can then become the input features for a neural network that classifies gestures such as:

* fist
* open hand
* pointing
* peace sign
* etc.

I also want to eventually experiment with dynamic gestures. Static gestures can be classified from a single frame, while dynamic gestures require looking at a sequence of hand positions over time. For this part, I may experiment with models such as LSTM/GRU or other suitable sequence models.

## Project goals

The project is intended primarily as an ML/computer vision learning and portfolio project.

I want to learn and demonstrate:

* computer vision
* real-time inference
* hand tracking
* feature extraction
* dataset creation
* data preprocessing
* neural network training
* PyTorch
* model evaluation
* gesture classification
* sequence/time-series modeling
* latency and FPS optimization
* integration of an ML model with a real application

The project should not just be "MediaPipe + keyboard input". The important ML component is collecting my own gesture dataset, training my own model, evaluating it, and integrating that model into a real-time inference pipeline.

## Planned development stages

The project will probably evolve approximately like this:

1. Capture webcam frames with OpenCV.
2. Detect and track the hand using MediaPipe.
3. Visualize the hand landmarks.
4. Collect a custom dataset of hand landmarks.
5. Preprocess and normalize the landmarks.
6. Train a simple PyTorch classifier for static gestures.
7. Run the classifier in real time.
8. Connect recognized gestures to keyboard/mouse actions.
9. Add mouse movement controlled by hand/finger position.
10. Add dynamic gesture recognition using sequences of landmarks.
11. Experiment with LSTM/GRU or other suitable approaches.
12. Measure model accuracy, FPS and inference latency.
13. Eventually create a polished demo for controlling desktop applications/games.

## Initial technology stack

The initial stack is intentionally simple:

* Python
* OpenCV
* MediaPipe
* NumPy
* PyTorch
* Git/GitHub

I am currently a beginner in this type of real-time computer vision project. I have some previous backend experience and basic ML experience, including neural networks from scratch, MNIST, logistic regression in PyTorch, and introductory CNN/RNN knowledge.

Because of that, explanations should be beginner-friendly and should explain the purpose of technologies and architectural decisions rather than assuming that I already know the standard ML engineering workflow.

The project should initially avoid unnecessary infrastructure such as Docker, databases, backend servers, Kubernetes, etc. unless there is a clear reason to introduce them later.

## Important context

This is a learning project, so I do not want everything hidden behind high-level libraries.

I want to understand what is happening in the pipeline:

camera
→ image
→ hand detection
→ landmarks
→ preprocessing
→ ML model
→ prediction
→ action

The final project should ideally demonstrate both ML knowledge and practical software engineering rather than simply being a collection of disconnected experiments.

When discussing this project, treat the above as the overall project context. Do not assume that I am asking you to implement anything unless I explicitly request implementation.
