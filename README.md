## Face-Mask-detection
Live camera feed showing if a person is wearing a face-mask or not.
Uses MobileNetV2 for image classification. Dataset consists of around 3800 images of both categories.
1. Install all dependencies using requirements.txt file by running `pip install -r requirements.txt`
2. Add path to dataset in `train_mask_detector.py`
3. Run `train_mask_detector.py` to train model.
4. Model and training loss and accuracy graph will be saved in current directory.
5. Run `detect_mask_video.py`
