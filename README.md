# run-gpt-oss

- Created at: 2025-08-06
- Created by: `🐢 Arun Godwin Patel @ Code Creations`

## Table of contents

- [Setup](#setup)
  - [System](#system)
  - [Installation](#installation)
- [Walkthrough](#walkthrough)
  - [1. Create a virtual environment](#1-create-a-virtual-environment)
  - [2. Activate the virtual environment](#2-activate-the-virtual-environment)
  - [3. Install the required packages](#3-install-the-required-packages)
  - [4. Create the pipeline](#4-create-the-pipeline)
  - [5. Serve your model](#5-serve-your-model)

## Setup

### System

This code repository was tested on the following computers:

- Windows 11

At the time of creation, this code was built using `Python 3.11.0`

### Installation

1. Install `virtualenv`

```bash
# 1. Open a CMD terminal
# 2. Install virtualenv globally
pip install virtualenv
```

2. Create a virtual environment

```bash
python -m venv venv
```

3. Activate the virtual environment

```bash
# Windows
.\venv\Scripts\activate
# Mac
source venv/bin/activate
```

4. Install the required packages

```bash
pip install -r requirements.txt
```

5. Run the module

```bash
python main.py
```

## Walkthrough

### 1. Create a virtual environment

```bash
python -m venv venv
```

### 2. Activate the virtual environment

```bash
# Windows
.\venv\Scripts\activate
# Mac
source venv/bin/activate
```

### 3. Install the required packages

```bash
pip install -r requirements.txt
```

#### 4. Create the pipeline

Create a file named `main.py` to be our entry point module for the testing the LLM.

```python
from transformers import pipeline
import torch

model_id = "openai/gpt-oss-20b"

pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype="auto",
    device_map="auto",
)

messages = [
    {"role": "user", "content": "Explain the core concepts of string theory in 3 bullet points."},
]

outputs = pipe(
    messages,
    max_new_tokens=256,
)
print(outputs[0]["generated_text"][-1])
```

Now run the script in the terminal!

#### 5. Serve your model

Now we will cover briefly a simple way to serve your model using the `transformers` library. From the terminal, run the following command:

```bash
transformers serve
transformers chat localhost:8000 --model-name-or-path openai/gpt-oss-20b
```

This completes a quick overview of the OpenAI GPT OSS model and how to set it up for local use. You can now interact with the model via the command line or through a web interface.

## Happy coding! 🚀

```bash
🐢 Arun Godwin Patel @ Code Creations
```
