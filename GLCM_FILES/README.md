# GLCM FILES

## EXPERIMENT 01

Features generated from  [Preprocessed](https://github.com/lukkascost/database-Crosswalk/tree/master/Preprocessed)  folder.
The File as named like FEATURES_M{}_CM{}b.txt when
#### M Variable

    M is the decimation of image, between 1 and 100. When M=1 the image is original image.
    the value of M is the size of step into line and col of image.

#### CM Variable

    CM is the value of bits of CoOccurence Matrix, ex:
    When the value is 8, the image must be values between 0 and 255
    When the value is 7, the image must be values between 0 and 127
    CM value has changed between 2 and 8

## EXPERIMENT 02

Features generated from  [Original](https://github.com/lukkascost/database-Crosswalk/tree/master/Original)  folder. 
All the images has be preprocessed with Otsu's algorithm.

The File as named like FEATURES_M{}_CM{}b.txt when
#### M Variable

    M is the decimation of image, between 1 and 100. When M=1 the image is original image.
    the value of M is the size of step into line and col of image.

#### CM Variable

    CM is the value of bits of CoOccurence Matrix, ex:
    When the value is 8, the image must be values between 0 and 255
    When the value is 7, the image must be values between 0 and 127
    CM value has changed between 2 and 8


## EXPERIMENT 03

Features generated from  [Original](https://github.com/lukkascost/database-Crosswalk/tree/master/Original)  folder. 
All the images has be preprocessed with Threshold algorithm value 180.

The File as named like FEATURES_M{}_CM{}b.txt when
#### M Variable

    M is the decimation of image, between 1 and 100. When M=1 the image is original image.
    the value of M is the size of step into line and col of image.

#### CM Variable

    CM is the value of bits of CoOccurence Matrix, ex:
    When the value is 8, the image must be values between 0 and 255
    When the value is 7, the image must be values between 0 and 127
    CM value has changed between 2 and 8

## EXPERIMENT 04

Features generated from  [Original](https://github.com/lukkascost/database-Crosswalk/tree/master/Original)  folder. 
All the images has be preprocessed with Threshold algorithm variating between 160 and 220.

The File as named like FEATURES_M{}_CM{}b_TH{}.txt when
#### M Variable

    M is the decimation of image, between 1 and 100. When M=1 the image is original image.
    the value of M is the size of step into line and col of image.

#### CM Variable

    CM is the value of bits of CoOccurence Matrix, ex:
    When the value is 8, the image must be values between 0 and 255
    When the value is 7, the image must be values between 0 and 127
    CM value has changed between 2 and 8

#### TH Variable 
    TH is the value of threshold applied on image. 
    values smallest than threshold in image is changed to 0.
    values biggest than threshold keep they current value.


## EXPERIMENT 07

Features generated from  [BLUR_IMAGES](https://github.com/lukkascost/database-Crosswalk/tree/master/BLUR_IMAGES)  folder. 
All the images has be preprocessed with Threshold algorithm fix in 199.

The File as named like FEATURES_M{}_CM{}b_TH{}.txt when
#### M Variable

    M is the decimation of image, between 1 and 100. When M=1 the image is original image.
    the value of M is the size of step into line and col of image.

#### CM Variable

    CM is the value of bits of CoOccurence Matrix, ex:
    When the value is 8, the image must be values between 0 and 255
    When the value is 7, the image must be values between 0 and 127
    CM value has fixed in 8

#### TH Variable 
    TH is the value of threshold applied on image. 
    values smallest than threshold in image is changed to 0.
    values biggest than threshold keep they current value.


## Authors

* **Lucas Costa** - [lukkascost](https://github.com/lukkascost)

See also the list of [contributors](https://github.com/lukkascost/py_Crosswalk/contributors) who participated in this project.
