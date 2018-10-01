function [ feat bbox ] = mdnet_features_fcX_bbox(net, ims, opts)


n = size(ims,4);
nBatches = ceil(n/opts.batchSize);

net.layers{end}.type='pass';
net.layers{end-2}.type='pass';

%net.layers = net.layers(1:end-1);
res = [];
for i=1:nBatches
    
    batch = ims(:,:,:,opts.batchSize*(i-1)+1:min(end,opts.batchSize*i));
    if(opts.useGpu)
        batch = gpuArray(batch);
    end
    
    res = mdnet_simplenn_bbox(true,[],net, batch, 1, [], res, ...
            'disableDropout', true, ...
            'conserveMemory', 0, ...
            'sync', opts.sync) ;
    
%     res = vl_simplenn(net, batch, [], [], ...
%         'disableDropout', true, ...
%         'conserveMemory', true, ...
%         'sync', true) ;
    
    f = gather(res(end - 3).x) ;
    b = gather(res(end - 1).x) ;
    if ~exist('feat','var')
        feat = zeros(size(f,1),size(f,2),size(f,3),n,'single');
        bbox = zeros(size(f,1),size(f,1),4,n,'single');
    end
    feat(:,:,:,opts.batchSize*(i-1)+1:min(end,opts.batchSize*i)) = f;
    bbox(:,:,:,opts.batchSize*(i-1)+1:min(n,opts.batchSize*i)) = b;
    
end
