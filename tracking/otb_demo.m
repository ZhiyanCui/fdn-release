%% OTB DEMO
%
% Running the fast dynamic convolutional neural network on OTB100
%
% Zhiyan Cui, 2018

clear;
addpath('..\utils\');
seqList=importdata('./tracking/seqList100.txt');

for seq=1:numel(seqList)
    conf = genConfig('otb',seqList{seq});
    switch(conf.dataset)
    case 'otb'
        
        net = fullfile('models','fdn_vot-otb.mat');
        
    case 'vot2014'
        net = fullfile('models','fdn_otb-vot14.mat');
    case 'vot2015'
        net = fullfile('models','fdn_otb-vot15.mat');   
    end     
    result = mdnet_run(conf.imgList, conf.gt(1,:), net);       
end

