## Face-Mask-detection
Live camera feed showing if a person is wearing a face-mask or not.
Uses MobileNetV2 for image classification. Dataset consists of around 3800 images of both categories.
Install all dependencies using requirements.txt file by running pip install -r requirements.txt.
Add path to dataset in train_mask_detector.py
Run train_mask_detector.py to train model.
Model and training loss and accuracy graph will be saved in current directory.
Run detect_mask_video.py
