# Cover-letter-Generator (Fine-Tuning)

This project demonstrates fine-tuning a pre-trained language model (Falcon-rw-1b) can also use Mistral-7B-Instruct-v0.2 to generate personalized cover letters based on a user's resume and a job description. The fine-tuned model is then used to build a web application using Gradio(Front-end), allowing users to upload their resume and paste job responsibilities to get a tailored cover letter.

## Features

*   **Fine-tuning:** Fine-tune a large language model on your custom dataset to improve its ability to generate relevant cover letters.
*   **Resume Parsing:** Extracts text content from PDF and DOCX resume files.
*   **Job Description Input:** Allows users to provide job responsibilities as text input.
*   **Cover Letter Generation:** Generates a custom cover letter by combining information from the resume and job description using the fine-tuned model.
*   **Gradio Interface:** Provides a user-friendly web interface to interact with the cover letter generator.

## Requirements

*   Google Colab environment (recommended) or a Python environment with GPU support.
*   The following Python libraries:
    * transformers: Provides thousands of pre-trained models to perform tasks on texts, images, or audio.
    * datasets: Offers a collection of publicly available datasets and tools to easily load, process, and share datasets.
    * peft: Enables Parameter-Efficient Fine-tuning (PEFT) methods for large pre-trained models.
    * trl: A transformer reinforcement learning library with implementations of state-of-the-art RL algorithms.
    * accelerate: A library that helps to easily train PyTorch models on any kind of distributed setup.
    * bitsandbytes: Provides an easy way to quantize models for reduced memory usage and faster inference.
    * huggingface_hub: A library to interact with the Hugging Face Hub, allowing you to download and upload models and datasets.
    * gradio: An open-source Python library for building machine learning web applications in minutes.
    * pdfplumber: A library for extracting text and data from PDFs.
    * docx2txt: A Python module to extract text from .docx files.

## Setup and Usage

1.  **Open in Google Colab:** The provided code is designed to be run in Google Colab. Open a new Colab notebook and paste the code into it.
2.  **Install Dependencies:** Run the first code cell to install the necessary libraries.
3.  **Prepare Training Data:** Create a JSON Lines file named `fine_tune.jsonl` with your fine-tuning data. Each line in the file should be a JSON object with 'input' and 'output' keys, representing pairs of resume and job description information and the corresponding desired cover letter.
4.  **Upload Training Data:** Run the second code cell to upload your `fine_tune.jsonl` file to the Colab environment.
5.  **Hugging Face Login:** Run the third code cell and follow the instructions to log in to your Hugging Face account. This is required to download the base model and potentially upload your fine-tuned model later.
6.  **Run Fine-tuning:** Run the main code block that contains the fine-tuning logic. This will load the base model, set up quantization and LoRA, tokenize your data, and train the model.
7.  **Install Gradio and File Parsers:** Run the next code cell to install the libraries for the Gradio interface and document parsing.
8.  **Run the Gradio App:** Run the final code cell to start the Gradio web application. A public URL will be provided, allowing you to access the cover letter generator.

## Customization

*   **Base Model:** You can change the `MODEL_NAME` variable to use a different base model from the Hugging Face Hub. Ensure the chosen model is suitable for causal language modeling.
*   **Fine-tuning Data:** The quality and quantity of your `fine_tune.jsonl` file will significantly impact the performance of the fine-tuned model.
*   **PEFT Parameters:** You can adjust the `r`, `lora_alpha`, and `lora_dropout` parameters in the `LoraConfig` to experiment with different LoRA configurations.
*   **Training Arguments:** Modify the `TrainingArguments` to change hyperparameters like the number of epochs, batch size, learning rate, etc.
*   **Gradio Interface:** Customize the Gradio interface (input/output components, title, description) to fit your needs.

## Saving and Loading the Model

The fine-tuned model and tokenizer are saved to the directory specified by `OUTPUT_DIR` (`fine_tuned_model` by default). You can download this directory to save your fine-tuned model and load it later for inference without retraining.

## Contributing

If you would like to contribute to this project, please fork the repository and submit a pull request.
