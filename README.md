# radiology_diagnostic_predictor
This project is a fully versatile, multi-modal, multi-organ and multi-disease system to
predict diagnosis of radiological scans using neural networks.

The project is written almost entirely in python using libraries such as numpy and tensorflow amongst others.<br>
The programs in the project process scans of different data types and modalities, then trains either a 2D or 3D Convolutional Neural Network, tests the generated model and there is also a user interface to make predictions on new, unseen scans

I designed the system to work with any dataset of medical imaging, regardless of disease, area of body or type of scan. <br>
There are systems for both 2 dimensional and 3-dimensional medical imaging. <br>
I have tested the program on 2 datasets:<br>
• OASIS-3 Dataset - CT Scans of Brains with and without Alzheimer’s – 3D<br>
• Dataset of Pediatric Chest X-Rays with and without Pneumonia – 2D

I worked on this project from 2023 to 2025.<br>
I presented this project at the BTYSE in both 2024 and 2025, aswell as BD Stem Stars 2024 and SciFest Cork 2024.<br>
I received several awards for it including:<br>
  • 3rd place Senior Technology at BTYSE 2025<br>
  • Boston Scientific award at SciFest<br>
  • Overall Winner & Best Research at BD Stem Stars<br>

Included in this repository are the python files for the programs, the research paper and the poster.<br>
The best place to start is with the poster which provides a brief overview of the project.<br>
Python Files:<br>
  user_program.py -> user interace program to make predictions on unseen scans.<br>
  testing.py -> program to test models<br>
  /alzheimers_ct<br>
    main.py -> Program to train a 3D CNN on 3D Scans (i.e. MRIs)<br>
    preprocessing.py -> Program to process 3D scans (saves as numpy arrays)
  /pneumonia_xray<br>
    train_cnn.py -> Program to train a 2D CNN on 2D Scans (i.e. X-rays)<br>
    preprocessing.py -> Program to process 2D scans (saves as numpy arrays)
  
