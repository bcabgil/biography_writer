# biography_writer

The `biography_writer` application processes audio files to generate a cohesive and ordered text. It transcribes audio files, processes the transcriptions, and outputs a structured biography. This tool is designed to handle audio files grouped by life events or periods to avoid overwhelming the model with too much information at once.

The pipeline follows these steps:

1. Transcribe the audio files
2. Applies gramatical and ortographical corrections.
3. Divides and orders text chunks representing similar information using UMAP and DBSCAN clustering. More methods to be added.
4. Orders and merges the texts to create an ordered and corrected representation of the initial content.  


The default LLM used in this pipeline is gpt-4o.

---

## Create the Environment

1. Create the conda environment
`conda create -n bwriter python=3.11.9`

2. Activate the environment
`conda activate bwriter`

3. Install poetry in the environment
`pip install poetry==2.1.3`

4. Install the project
`poetry install`

## Running the Pipeline
The main script for running the pipeline is main.py. It requires input audio files and optionally allows skipping the transcription step if transcriptions are already available.

### Command-Line Arguments
* `--input_directory (-i)`: Path to the directory containing audio files for processing (required).
* `--transcription_model (-tm)`: Path to the transcription model (default: \models\whisper_large_v3).
* `--skip_transcription (-st)`: Boolean flag to skip the transcription step (default: False).
Example Usage
1. Run the Full Pipeline (Including Transcription)
Ensure the transcription model is available locally at the specified path (default: \models\whisper_large_v3):

`poetry run python main.py -i path/to/audio/files -tm path/to/trancription/model`

2. Skip the Transcription Step
If transcriptions are already available, you can skip the transcription step:

`poetry run python main.py -i path/to/files -st True`

The transcribed texts should be in the `path/to/files` inside a directory named `transcriptions` so that the pipeline will find them.

## Input and Output Parameters
### Input
* Audio Files: Place your audio files in the directory specified by the --input_directory argument.
### Output
The pipeline generates the following outputs in the parent directory of the input directory:

1. Transcriptions: Transcribed text files (transcriptions/).
2. Corrected Transcriptions: Corrected text files (corrected_transcriptions/).
3. Merged Transcriptions: Merged and processed text files (merged_transcriptions/).
4. Biography: A structured biography generated from the processed transcriptions.
---

## Notes
The transcription model must be available locally at the path specified by the --transcription_model argument.
It is recommended to pre-group the files in meaningful groups to avoid overwhelming the model with too much information at once.

If the files contain sensitive content, the LLMs may discard it.

---

## Logging
The application logs its progress and outputs detailed information about each step of the pipeline. Check the logs for any errors or warnings during execution.

---

## Contributing

Send PRs to the develop branch. Before submitting any change make sure to run the pre-commit checks to ensure code cleanliness.

`poetry run pre-commit run --all-files`

Feel free to reach out if you encounter any issues or need further assistance!