function [net, info] = mdnet_train(net, roidb, getBatch, varargin)

opts.bbox_loss = false ;

opts.scale=600;
opts.max_scale=1000;

% opts.scale=480;
% opts.max_scale=640;
opts.interpolation='bilinear';


opts.batch_frames = 8 ;
opts.batchSize    = 128 ;
opts.batch_pos    = 32;
opts.batch_neg    = 96;

opts.numCycles    = 100 ;
opts.useGpu       = false ;
opts.conserveMemory = false ;

opts.sync = true ;
opts.learningRate = 0.0001 ;
opts.weightDecay = 0.0005 ;
opts.momentum = 0.9 ;

opts = vl_argparse(opts, varargin) ;

K = length(roidb);

% -------------------------------------------------------------------------
%                                                    Network initialization
% -------------------------------------------------------------------------
for i=1:numel(net.layers)
    if strcmp(net.layers{i}.type,'conv')
        net.layers{i}.filtersMomentum = zeros(size(net.layers{i}.filters), ...
            class(net.layers{i}.filters)) ;
        net.layers{i}.biasesMomentum = zeros(size(net.layers{i}.biases), ...
            class(net.layers{i}.biases)) ; %#ok<*ZEROLIKE>
        
        if ~isfield(net.layers{i}, 'filtersLearningRate')
            net.layers{i}.filtersLearningRate = 1 ;
        end
        if ~isfield(net.layers{i}, 'biasesLearningRate')
            net.layers{i}.biasesLearningRate = 2 ;
        end
        if ~isfield(net.layers{i}, 'filtersWeightDecay')
            net.layers{i}.filtersWeightDecay = 1 ;
        end
        if ~isfield(net.layers{i}, 'biasesWeightDecay')
            net.layers{i}.biasesWeightDecay = 0 ;
        end
        
        if opts.useGpu
            net.layers{i}.filtersMomentum = gpuArray(net.layers{i}.filtersMomentum);
            net.layers{i}.biasesMomentum = gpuArray(net.layers{i}.biasesMomentum);
        end
        
    end
end

% -------------------------------------------------------------------------
%                                                                  Training
% -------------------------------------------------------------------------
if opts.useGpu
    one = gpuArray(single(1)) ;
    net = vl_simplenn_move(net, 'gpu') ;
else
    one = single(1) ;
    net = vl_simplenn_move(net, 'cpu') ;
end
res = [] ;
lr = opts.learningRate;

loss_error = zeros(2,opts.numCycles);

% shuffle the frames for training
frame_list = cell(K,1);
frame_val = cell(K,1);




% % the image for training
% %maybe a  bug
% db = roidb;
% for i=1:K
%     
%     db{i} = db{i}(randperm(length(roidb{i})));
%     
%     % the image for training
%     roidb{i} =  db{i}(1:floor((0.8*numel(db{i}))));
%        
%     % the image for validation
%     roiva{i} =  db{i}(ceil(0.8 * numel(db{i}):end)); 
% end
% 
% 

for k=1:K
    nFrames = opts.batch_frames*opts.numCycles; 
    while(length(frame_list{k})<nFrames)
        frame_list{k} = cat(2,frame_list{k},uint32(randperm(length(roidb{k}))));
    end
    frame_list{k} = frame_list{k}(1:nFrames);
end

% for k=1:K
%     nFrames = opts.batch_frames*opts.numCycles; 
%     while(length(frame_val{k})<nFrames)
%         frame_val{k} = cat(2,frame_val{k},uint32(randperm(length(roiva{k}))));
%     end
%     frame_val{k} = frame_val{k}(1:nFrames);
% end



% init info
info.train.objective = zeros(K,opts.numCycles) ;
info.train.error = zeros(K,opts.numCycles) ;
info.train.speed = zeros(K,opts.numCycles) ;
if opts.bbox_loss
    info.train.bbox_loss = zeros(K,opts.numCycles) ;
end
%% training on training set
nextBatch = [];
for t=1:opts.numCycles
    fprintf('Training: processing cycle %3d of %3d ...\n', t, opts.numCycles) ;

    for seq_id=1:K
        batch_time = tic ;
        fprintf('\t seq %02d: ',seq_id);
        
        % get next image batch and labels
     
       
%         if opts.useGpu
%             im = gpuArray(im) ;
%         end

        
        % for validation
        %if mod(t,5)==0
        if mod(t,200)==0
            
            if(isempty(nextBatch))
                batch = frame_val{seq_id}((t-1)*opts.batch_frames+1:t*opts.batch_frames);
            else
                batch = nextBatch;
            end
            %[im, labels] = getBatch(roidb{seq_id}, batch, opts.batch_pos, opts.batch_neg) ;
     
            [im, rois,labels,bbox_labels] = get_batch(roiva{seq_id}, batch,opts);
            
            net.layers{end}.class = labels ;
            res = mdnet_simplenn(rois,net, im, seq_id, one, res, ...
            'disableDropout', true, ... 
            'conserveMemory', opts.conserveMemory, ...
            'sync', opts.sync) ;

            info.train = updateError(info.train, t, seq_id, labels, res, batch_time) ;

            fprintf('objective %.3f, error %.3f\n', ...
            info.train.objective(seq_id,t)/(opts.batchSize),...
            info.train.error(seq_id,t)/(opts.batchSize));
            fprintf('validation---------------\n');
            continue;
        end
        
        %for training
        
        if(isempty(nextBatch))
            %batch = frame_val{seq_id}((t-1)*opts.batch_frames+1:t*opts.batch_frames);
            batch = frame_list{seq_id}((t-1)*opts.batch_frames+1:t*opts.batch_frames);
        else
            batch = nextBatch;
        end
        %[im, labels] = getBatch(roidb{seq_id}, batch, opts.batch_pos, opts.batch_neg) ;
     
        [im, rois,labels,bbox_labels] = get_batch(roidb{seq_id}, batch,opts);
        
        % backprop
        if opts.bbox_loss
            net.layers{end}.regression = bbox_labels ;
            net.layers{end}.class = labels ;
            net.layers{end-2}.class = labels ;
            
            res = mdnet_simplenn_bbox(true,rois,net, im, seq_id, one, res, ...
            'conserveMemory', opts.conserveMemory, ...
            'sync', opts.sync) ;
            
            
        else
            net.layers{end}.class = labels ;
            res = mdnet_simplenn(rois,net, im, seq_id, one, res, ...
            'conserveMemory', opts.conserveMemory, ...
            'sync', opts.sync) ;
        end
%         res = mdnet_simplenn(net, im, seq_id, one, res, ...
%             'conserveMemory', opts.conserveMemory, ...
%             'sync', opts.sync) ;
        
         
        
        % gradient step
        for l=1:numel(net.layers)
            if ~strcmp(net.layers{l}.type, 'conv'), continue ; end
            
            net.layers{l}.filtersMomentum = ...
                opts.momentum * net.layers{l}.filtersMomentum ...
                - (lr * net.layers{l}.filtersLearningRate) * ...
                (opts.weightDecay * net.layers{l}.filtersWeightDecay) * net.layers{l}.filters ...
                - (lr * net.layers{l}.filtersLearningRate) / opts.batchSize * res(l).dzdw{1} ;
            
            net.layers{l}.biasesMomentum = ...
                opts.momentum * net.layers{l}.biasesMomentum ...
                - (lr * net.layers{l}.biasesLearningRate) * ....
                (opts.weightDecay * net.layers{l}.biasesWeightDecay) * net.layers{l}.biases ...
                - (lr * net.layers{l}.biasesLearningRate) / opts.batchSize * res(l).dzdw{2} ;
            
            net.layers{l}.filters = net.layers{l}.filters + net.layers{l}.filtersMomentum ;
            net.layers{l}.biases = net.layers{l}.biases + net.layers{l}.biasesMomentum ;
        end
        
        % print information
        batch_time = toc(batch_time) ;
        speed = opts.batchSize/batch_time ;
        
        if opts.bbox_loss
            info.train = updateError_bbox(info.train, t, seq_id, labels, res, batch_time) ;
            fprintf(' objective %.3f,  bbox loss:%.3f  error %.3f\n', ...
            info.train.objective(seq_id,t)/opts.batchSize,...
            info.train.bbox_loss(seq_id,t),...
            info.train.error(seq_id,t)/(opts.batchSize)) ;
            
            
        else
            info.train = updateError(info.train, t, seq_id, labels, res, batch_time) ;
        
            fprintf(' %.2f s (%.1f images/s),', batch_time, speed) ;
            fprintf(' objective %.3f, error %.3f\n', ...
            info.train.objective(seq_id,t)/(opts.batchSize),...
            info.train.error(seq_id,t)/(opts.batchSize)) ;
        end
        
        
    end
    fprintf('\n') ;
    if opts.bbox_loss
        mean_objective = mean(info.train.objective(:,t)) / opts.batchSize;
        mean_bbox_loss = mean(info.train.bbox_loss(:,t)) ;
        mean_error = mean(info.train.error(:,t))/(t*opts.batchSize) ;
        fprintf('Total: objective %.3f, bbox loss:%.3f  error %.3f\n', mean_objective, mean_bbox_loss, mean_error) ;
        fprintf('\n') ;
    else
        mean_objective = mean(info.train.objective(:,t))/(opts.batchSize) ;
        mean_error = mean(info.train.error(:,t))/(opts.batchSize) ;
%         mean_objective = mean(info.train.objective(:,t))/(t*opts.batchSize) ;
%         mean_error = mean(info.train.error(:,t))/(t*opts.batchSize) ;
        fprintf('Total: objective %.3f, error %.3f\n', mean_objective, mean_error) ;
        fprintf('\n') ;
        loss_error(1,t) = mean_objective;
        loss_error(2,t) = mean_error;
        
    end
    
end % next batch
if opts.bbox_loss
        
else
    info.train.objective = info.train.objective ./ (opts.batchSize*repmat(1:opts.numCycles,K,1)) ;
    info.train.error = info.train.error ./ (opts.batchSize*repmat(1:opts.numCycles,K,1))  ;
    info.train.speed = (opts.batchSize*repmat(1:opts.numCycles,K,1)) ./ info.train.speed ;

end
end

% -------------------------------------------------------------------------
function info = updateError_bbox(info, t, k, labels, res, time)
% -------------------------------------------------------------------------
info.bbox_loss(k,t) = gather(res(end).x) ;
info.objective(k,t) = gather(res(end-2).x) ;
info.speed(k,t) = time;


% if(t>1)
%     info.objective(k,t) = info.objective(k,t-1) + gather(res(end).x) ;
%     info.speed(k,t) = info.speed(k,t-1) + time;
% else
%     info.objective(k,t) = gather(res(end).x) ;
%     info.speed(k,t) = time;
% end

if(size(res(end-3).x,3)==2)
    predictions = gather(res(end-3).x) ;
else
    predictions = gather(res(end-3).x(:,:,k*2-1:k*2,:)) ;
end
sz = size(predictions) ;
n = prod(sz([1,2])) ;

[~,predictions] = sort(predictions, 3, 'descend') ;
error = ~bsxfun(@eq, predictions, reshape(labels, 1, 1, 1, [])) ;
if(t>1)
    info.error(k,t) = info.error(k,t-1) + sum(sum(sum(gather(error(:,:,1,:)))))/n ;
else
    info.error(k,t) = sum(sum(sum(gather(error(:,:,1,:)))))/n ;
end
end



% -------------------------------------------------------------------------
function info = updateError(info, t, k, labels, res, time)
% -------------------------------------------------------------------------

% if(t>1)
%     info.objective(k,t) = info.objective(k,t-1) + gather(res(end).x) ;
%     info.speed(k,t) = info.speed(k,t-1) + time;
% else
%     info.objective(k,t) = gather(res(end).x) ;
%     info.speed(k,t) = time;
% end

info.objective(k,t) = gather(res(end).x) ;
info.speed(k,t) = time;



if(size(res(end-1).x,3)==2)
    predictions = gather(res(end-1).x) ;
else
    predictions = gather(res(end-1).x(:,:,k*2-1:k*2,:)) ;
end
sz = size(predictions) ;
n = prod(sz([1,2])) ;

[~,predictions] = sort(predictions, 3, 'descend') ;
error = ~bsxfun(@eq, predictions, reshape(labels, 1, 1, 1, [])) ;

info.error(k,t) = sum(sum(sum(gather(error(:,:,1,:)))))/n ;


% if(t>1)
%     info.error(k,t) = info.error(k,t-1) + sum(sum(sum(gather(error(:,:,1,:)))))/n ;
% else
%     info.error(k,t) = sum(sum(sum(gather(error(:,:,1,:)))))/n ;
% end
end



