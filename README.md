# No_NumberPlate_will_be_left

The Model can be used to detect and identify number plates in an image of cars.

Working in brief:

1.The model first uses YOLO to localize the number plate in the image.

2.The region of number plate is then cropped and is processed with various approaches of Computer Vision.

3.The resultant image is then goes through MNIST dataset and the alphanumerics is Identified and displayed above original position of the number plate.

# How to Train:

1.Use import_dataset.py to download and collect dataset in the folders required.

2.Fine tune the train.py, config.json for better training.

3.Run train.py and the training will begin

# How to use the model:

1.Add the address of saved model in config.json

2.Add the addresses of the folders where the Test images/videos are and where to save the outputs in the predict.py

3.Run predict.py and wait for results

