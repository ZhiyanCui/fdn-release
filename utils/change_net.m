function [ net ] = change_net( init_net ,opts)
net=init_net;
% net.layers{1}.stride=[1,1];
% net.layers{5}.stride=[1,1];

net.layers{1}.pad=3;
net.layers{5}.pad=2;
net.layers{9}.pad=1;

% not sure for this part
net.layers{4}.pad=1;
net.layers{8}.pad=1;


% roi_layer = struct('type', 'roipool', 'name', 'roipool','subdivisions',...
%     [7,7],'method','max','transform',0.0625);

roi_layer = struct('type', 'roipool', 'name', 'roipool','subdivisions',...
    [3,3],'method','max','transform',0.0625);

net.layers = net.layers(1:10);

net.layers{end+1} = roi_layer;
net.layers=[net.layers,init_net.layers(11:end)];

end

