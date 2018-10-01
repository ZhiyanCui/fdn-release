function [net_conv, net_fc, opts] = mdnet_init(image, net)

opts.learningRate = 0.0001 ;
opts.scale=600;
opts.max_scale=1000;
opts.interpolation='bilinear';
opts.conserveMemory=true;
opts.sync = true ;

%% set opts
% use gpu
if isunix
    opts.useGpu = true;
elseif ispc
    opts.useGpu = false; 
else
    error('Platform is not windows or linux');
end
% model def
opts.net_file = net;

% test policy
opts.batchSize_test = 256; % <- reduce it in case of out of gpu memory

opts.bbreg = true;
opts.bbreg_nSamples = 1000;

% learning policy
opts.batchSize = 128;
opts.batch_pos = 32;
opts.batch_neg = 96;

opts.learningRate_init = 0.0001 * 5; % x10 for fc6
opts.maxiter_init = 30;

opts.nPos_init = 500;  
opts.nNeg_init = 5000;
opts.posThr_init = 0.7;
opts.negThr_init = 0.5;

% update policy
opts.learningRate_update = 0.0003 * 5; % x10 for fc6
opts.maxiter_update = 10;  %10

opts.nPos_update = 50;
opts.nNeg_update = 200; %200
opts.posThr_update = 0.7;
opts.negThr_update = 0.3;  %0.3

opts.update_interval = 10; %10; % interval for long-term update

% data gathering policy
opts.nFrames_long = 100; % long-term period
opts.nFrames_short = 20; % short-term period

% cropping policy
opts.input_size = 107;
opts.crop_mode = 'wrap';
opts.crop_padding = 16;

% scaling policy
opts.scale_factor = 1.05;

% sampling policy
opts.nSamples = 256;
opts.trans_f = 0.6; % translation std: mean(width,height)*trans_f/2
opts.scale_f = 1; % scaling std: scale_factor^(scale_f/2)

% set image size
opts.imgSize = size(image);

%% load net
net = load(opts.net_file);
if isfield(net,'net'), net = net.net; end

net_conv.layers = net.layers(1:11);
net_fc.layers = net.layers(12:end);


if isfield(net,'meta')
    net_conv.meta=net.meta;
end

clear net;

for i=1:numel(net_fc.layers)
    switch (net_fc.layers{i}.name)
        case {'fc4','fc5'}
            net_fc.layers{i}.filtersLearningRate = 1;
            net_fc.layers{i}.biasesLearningRate = 2;
        case {'fc6'}
            net_fc.layers{i}.filtersLearningRate = 10;
            net_fc.layers{i}.biasesLearningRate = 20;
    end
end

if opts.useGpu
    net_conv = vl_simplenn_move(net_conv, 'gpu') ;
    net_fc = vl_simplenn_move(net_fc, 'gpu') ;
else
    net_conv = vl_simplenn_move(net_conv, 'cpu') ;
    net_fc = vl_simplenn_move(net_fc, 'cpu') ;
end

end