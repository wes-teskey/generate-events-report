# generate-events-report
Use AI agentic crews to generate reports on events from web searches of event venues

## Prerequisites

Before you begin, ensure you have met the following requirements:
- You have installed the latest version of Python. You can download it from [here](https://www.python.org/downloads/).
- You have a package manager for Python, such as `pip`.

## Setting Up a Virtual Environment

1. **Navigate to the project directory:**
cd path/to/your/project

2. **Create and activate virtual environment (example shown for windows):**
python -m venv venv
venv\Scripts\activate

3. **Install the required packages (shown for windows):**
pip install crewai
pip install "crewai[tools]"

## Windows Users: Resolving Microsoft Visual C++ 14.0 Build Tools Error

If you encounter an error related to chroma not being installed, 
please follow the steps below to resolve it:

### Step-by-Step Guide to Fix the Issue

#### Download the Microsoft C++ Build Tools:
1. Go to the [Microsoft C++ Build Tools download page](https://visualstudio.microsoft.com/visual-cpp-build-tools/).
2. Click on the "Download Build Tools" button to download the installer.

#### Install the Build Tools:
1. Run the installer you downloaded (`vs_buildtools.exe`).
2. In the installer, choose the "Workloads" tab.
3. Select "Desktop development with C++".
4. Make sure to check the "C++ Build Tools" option.
5. Click on the "Install" button to begin the installation process. This may take some time depending on your internet speed and system performance.

#### Verify Installation:
After the installation is complete, verify the installation

## Description of Key Directories and Files

- **`data/`**: Contains all data-related files.
- **`src/`**: Contains the source code of the project.
- **`venv/`**: Contains the virtual environment for the project.

### Creating the `data/` Directory

Before running the project, ensure that the `data/` directory is created at the same level as the `src/` directory

## Usage
Run the file events_locator.py and follow the prompts

It is recommend to write out dates to search in full, for example 
write "June 11 2024, June 12 2024, June 13 2024" instead of "June 11-13"

LLM defaults to gpt-4o, and this can be changed to any suitable LLM

The event venues default to Las Vegas venues, and can be modified by changing events_venues.py