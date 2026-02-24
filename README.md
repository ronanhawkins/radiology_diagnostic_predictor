# radiology_diagnostic_predictor

This project is a fully versatile, multi-modal, multi-organ and multi-disease system to
predict diagnosis of radiological scans using neural networks.

The project is written almost entirely in python using libraries such as numpy and tensorflow amongst others.  
The programs in the project process scans of different data types and modalities, then trains either a 2D or 3D Convolutional Neural Network, tests the generated model and there is also a user interface to make predictions on new, unseen scans

I designed the system to work with any dataset of medical imaging, regardless of disease, area of body or type of scan.  
There are systems for both 2 dimensional and 3-dimensional medical imaging.  
I have tested the program on 2 datasets:
- OASIS-3 Dataset - CT Scans of Brains with and without Alzheimer’s – 3D
- Dataset of Pediatric Chest X-Rays with and without Pneumonia – 2D

I worked on this project from 2023 to 2025.  
I presented this project at the BTYSE in both 2024 and 2025, aswell as BD Stem Stars 2024 and SciFest Cork 2024.  
I received several awards for it including:
- 3rd place Senior Technology at BTYSE 2025
- Boston Scientific award at SciFest
- Overall Winner & Best Research at BD Stem Stars

Included in this repository are the python files for the programs, the research paper and the poster.  
The best place to start is with the poster which provides a brief overview of the project.

Python Files (filesystem tree with links):
- [user_program.py](./user_program.py) -> user interace program to make predictions on unseen scans.
- [testing.py](./testing.py) -> program to test models
- [/alzheimers_ct](./alzheimers_ct/)
  - [main.py](./alzheimers_ct/main.py) -> Program to train a 3D CNN on 3D Scans (i.e. MRIs)
  - [preprocessing.py](./alzheimers_ct/preprocessing.py) -> Program to process 3D scans (saves as numpy arrays)
- [ /pneumonia_xray](./pneumonia_xray/)
  - [training_cnn.py](./pneumonia_xray/training_cnn.py) -> Program to train a 2D CNN on 2D Scans (i.e. X-rays)
  - [preprocessing.py](./pneumonia_xray/preprocessing.py) -> Program to process 2D scans (saves as numpy arrays)
