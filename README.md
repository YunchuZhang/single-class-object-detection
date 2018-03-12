# single-class-object-detection

The codes include data and data.rec. Based on YoLo algorithm and 4 existed model to detect single-class objects.In the code, it builds CNN body and Loss function in MXNET.
Also, this code has the function to check tensorboard.


# data

You can find the Data.rec file model and para file in those links:

https://drive.google.com/file/d/17upbN20WpzLEGCzFVWJJ1gHlAbXrsopB/view?usp=sharing

https://drive.google.com/open?id=17upbN20WpzLEGCzFVWJJ1gHlAbXrsopB



# how to train and test

To train the code, you should use:

python run_train.py

And GPU is needed with Mxnet,maybe you need to change the path of data you load

final test could be used to test the model as below:

![image](https://github.com/YunchuZhang/single-class-object-detection/blob/master/cat%20result.jpg)


The revised one which will have the function to detect every object in one picture will be uploaded recently.

# end
