## Fast Dynamic Convolutional Neural Networks for Visual Tracking

### introduction
This is the implementation of our paper: **Fast Dynamic Convolutional Neural Networks for Visual Tracking**. The paper can be found here:
[Fast Dynamic Convolutional Neural Networks for Visual Tracking](https://arxiv.org/pdf/1807.03132.pdf). Our paper is accept by [ACML2018(Asian Conference on Machine Learning)](http://www.acml-conf.org/2018/)

The pipeline is built upon the MDNet tracker for your reference:[http://cvlab.postech.ac.kr/research/mdnet/](http://cvlab.postech.ac.kr/research/mdnet/)

Our code is implemented using Matlab and [MatConvNet](http://www.vlfeat.org/matconvnet/)

### Requirements
This code is tested on 64 bit Linux.    
1.Matlab   
2.MatConvNet   
3.GPU and CUDA   

### Setup and Run
1.Compile MatConvNet according to [MatConvNet installation guideline](http://www.vlfeat.org/matconvnet/install/)   
2.change the line 22 in 'utils/genConfig.m' to your path of [OTB100](http://cvlab.hanyang.ac.kr/tracker_benchmark/).   
3.compile the roialign function in matlab    
&nbsp;&nbsp;&nbsp;&nbsp;1)&nbsp;cd roialign     
&nbsp;&nbsp;&nbsp;&nbsp;2)&nbsp;mexcuda roialign.cu    
4.run'setup_demo' to set the environment for running the code.   
5.run 'tracking/otb_demo.m' to see the results on OTB100.  

### Citation
If you are using this code in a publication, please cite our paper.
[Fast Dynamic Convolutional Neural Networks for Visual Tracking](https://arxiv.org/pdf/1807.03132.pdf)

### License
This software is being made available for research purpose only.
