# 📝 Text Summarization Tool

This is a web application for summarizing long pieces of text using **Extractive** and **Abstractive** methods. Built with **Streamlit**, it allows users to input text and receive a summarized version efficiently.

## Features

- **Extractive Summarization**: Extracts key sentences from the original text using text processing and graph algorithms.
- **Abstractive Summarization**: Generates a concise summary using a pre-trained transformer model that may include novel words or phrases.


## Installation

### Prerequisites

- **Python 3.10**
- **Anaconda** (optional but recommended for managing environments)

### Setup Instructions

#### 1. Clone the Repository

```bash
git clone https://github.com/Sulaiman-Nedal/text-summarization-tool.git
cd text-summarization-tool
```

#### 2. Create a Virtual Environment

Using **Conda**:

```bash
conda create -n text_summarizer_env python=3.10
conda activate text_summarizer_env
```

Or using **virtualenv**:

```bash
python -m venv text_summarizer_env
# Activate the environment
# On macOS/Linux:
source text_summarizer_env/bin/activate
# On Windows:
text_summarizer_env\Scripts\activate
```

#### 3. Install Dependencies

Install the required Python packages using `pip`:

```bash
pip install -r requirements.txt
```

**Note:** Ensure that you have the correct versions of the packages to avoid compatibility issues. The key dependencies are:

- `nltk==3.8.1`
- `networkx==3.0`
- `streamlit`
- `scikit-learn`
- `transformers`
- `numpy`
- `pandas`
- `torch`

#### 4. Download NLTK Data

Download the necessary NLTK data files:

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

You can run this in a Python shell or include it at the beginning of your `app.py` script.

#### 5. Run the Application

```bash
streamlit run app.py
```

## Usage

1. **Open the Application**

   After running the command, a new browser window should open automatically. If not, navigate to `http://localhost:8501` in your web browser.

2. **Input Text**

   - Enter the text you want to summarize in the text area provided.

3. **Select Summarization Method**

   - Use the sidebar to choose between **Extractive** and **Abstractive** summarization.
   - Adjust parameters:
     - **Extractive Summarization**:
       - Choose the number of sentences for the summary.
     - **Abstractive Summarization**:
       - Set the maximum and minimum length of the summary.

4. **Generate Summary**

   - Click the **🔍 Summarize** button.
   - The summary will be displayed below.

## Dependencies

- **Python Packages**:
  - `streamlit`
  - `nltk==3.8.1`
  - `numpy`
  - `pandas`
  - `scikit-learn`
  - `networkx==3.0`
  - `transformers`
  - `torch`
  - `networkx`
  - `nltk` data packages: `punkt`, `stopwords`

- **Models**:
  - `sshleifer/distilbart-cnn-12-6` (used for abstractive summarization)


Contributions are welcome! If you have suggestions or improvements, feel free to open an issue or submit a pull request.

## Contact

For any questions or support, please contact:

- **Name**: Sulaeman Aloradi
- **Email**: snedal99@gmail.com
- **GitHub**: [Sulaiman-Nedal](https://github.com/Sulaiman-Nedal)

---

*Happy summarizing!*
