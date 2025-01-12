# Face Segregation Tool

This is a Python-based face segregation tool with a user-friendly UI built using Streamlit. The application allows you to:
1. Add known faces by uploading images or capturing them via your webcam.
2. Upload unknown images for segregation.
3. Automatically organize images based on recognized faces or group unrecognized faces into clusters.

## Features
- Upload known faces with names.
- Capture images directly from your webcam and save them with names.
- Upload multiple unknown images at once.
- Use machine learning clustering (DBSCAN) to organize unrecognized faces into groups.
- Intuitive Streamlit UI.

## Installation

### Prerequisites
- Python 3.8 or later
- A GPU-enabled computer is recommended for faster processing.

### Steps
1. Clone this repository:
   ```bash
   git clone https://github.com/your-repo/face-segregation-tool.git
   cd face-segregation-tool
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the application:
   ```bash
   streamlit run app.py
   ```

## Usage

1. **Add Known Faces**:
   - Upload photos of people with their names, or capture their photo directly using your webcam.
   - Photos will be saved in the `known_faces` directory.

2. **Upload Unknown Faces**:
   - Upload a batch of images containing unknown faces.
   - These will be saved in the `unknown_faces` directory.

3. **Run Segregation**:
   - Click the "Start Segregation" button to organize images.
   - Segregated images will be saved in the `organized_faces` folder:
     - Recognized faces will be saved in folders named after the person.
     - Unrecognized faces will be grouped into clusters.

## Folder Structure
```
project/
│
├── app.py                 # Main application script
├── known_faces/           # Directory for known faces (auto-created)
├── unknown_faces/         # Directory for unknown faces (auto-created)
├── organized_faces/       # Directory for segregated faces (auto-created)
├── requirements.txt       # Dependencies for the project
├── .gitignore             # Files and folders to ignore in Git
└── README.md              # Project documentation
```

## Notes
- Ensure your webcam is properly set up if you plan to capture images.
- For best results, use clear and high-quality photos of faces.

## License
This project is open-source under the MIT License.
```
