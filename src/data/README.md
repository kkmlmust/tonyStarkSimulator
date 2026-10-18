# DataCollector

## Purpose

`DataCollector` is responsible for collecting and saving hand landmark data for a gesture recognition dataset.

It receives hand detection results from `Hand_tracker` and saves the detected landmarks together with their corresponding gesture labels to CSV files.

The class must be independent of the webcam, keyboard input, and MediaPipe initialization.

## Responsibilities

- Manage the recording state.
- Start and stop recording.
- Receive hand landmark detection results.
- Save landmark coordinates and gesture labels to CSV
- Organize recordings into separate files or sessions.
- Close files and release resources correctly.

## Initialization

When creating a `DataCollector` instance, provide the base directory where the dataset will be stored.

The same instance should be reusable for recording multiple gestures.

## Recording Workflow

### Start Recording

When recording starts, the collector receives the name of the gesture being recorded.

It must:
- Set its state to `RECORDING`.
- Create or open the appropriate output CSV file.
- Prepare the file for writing new samples.
- Associate all samples from the current recording with the selected gesture label.

### Collect Results

While recording is active, the collector receives detection results.

For every result:
- Check whether a hand was detected.
- Extract the landmark coordinates.
- Save the coordinates and gesture label as one sample.
- Write the sample to the CSV file.

If no hand is detected, skip the sample.

Data should be written progressively rather than accumulated entirely in memory until recording stops.

### Stop Recording

When recording stops, the collector must:
- Stop accepting new samples for the current recording.
- Flush pending file writes if necessary.
- Close the current CSV file.
- Return to the `IDLE` state.

The collector must not overwrite previously recorded data unintentionally.

## Data Format

For the initial version, record landmarks for one hand.

Each hand contains 21 landmarks, with three coordinates per landmark: `x`, `y`, and `z`.

Each sample therefore contains:
- 63 coordinate values.
- One gesture label.

The CSV file must have a header with clearly named columns.

The format should be consistent across all recordings so that the dataset can later be loaded into NumPy or PyTorch.

## State Management

The collector has two states:

- `IDLE` — no active recording; incoming results are not saved.
- `RECORDING` — incoming valid hand detection results are saved.

Starting or stopping a recording must update the state correctly.

The collector should handle invalid operation sequences safely, such as attempting to stop when no recording is active.

## Asynchronous Results

MediaPipe operates asynchronously, so detection results may arrive after a small delay.

The collector must ensure that samples are saved only when they belong to an active recording. Results received after recording has stopped must not be written to the completed session.

Only one component should consume results from the result queue.

## Error Handling and Resource Management

- Handle missing or invalid output directories appropriately.
- Avoid accidental overwriting of existing recordings.
- Ensure files are closed even when recording ends unexpectedly.
- Keep the collector reusable after a recording has finished.

## Out of Scope

The collector must not:
- Capture frames from the webcam.
- Initialize or configure MediaPipe.
- Handle keyboard input.
- Display images or GUI elements.
- Train or run a gesture classification model.

These responsibilities belong to other parts of the application.
