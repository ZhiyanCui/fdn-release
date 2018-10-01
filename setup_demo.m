%% SETUP DEMO
%
% Zhiyan Cui, 2018

if(isempty(gcp('nocreate')))
    parpool;
end

run matconvnet/matlab/vl_setupnn ;

addpath('pretraining');
addpath('tracking');
addpath('utils');
addpath('roialign');
addpath('vot');
