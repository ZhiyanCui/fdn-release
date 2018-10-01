function [ result ] = mdnet_run(images, region, net, display)


% INPUT:
%   images  - 1xN cell of the paths to image sequences
%   region  - 1x4 vector of the initial bounding box [left,top,width,height]
%   net     - The path to a trained MDNet
%   display - True for displying the tracking result
%
% OUTPUT:
%   result - Nx4 matrix of the tracking result Nx[left,top,width,height]
%


opts.learningRate = 0.0001 ;
opts.scale=600;
opts.max_scale=1000;
opts.interpolation='bilinear';

if(nargin<4), display = true; end

%% Initialization
fprintf('Initialization...\n');

nFrames = length(images);
img = imread(images{1});
ori_img=img;
if(size(img,3)==1), img = cat(3,img,img,img); end
img=single(img);
targetLoc = region;
result = zeros(nFrames, 4); result(1,:) = targetLoc;

% Resize images and boxes to a size compatible with the network.
imageSize = size(img) ;
h=imageSize(1);
w=imageSize(2);
factor = max(opts.scale/h,opts.scale/w);
if any([h*factor,w*factor]>opts.max_scale)
    factor = min(opts.max_scale/h,opts.max_scale/w);
end

img = imresize(img,factor,'Method',opts.interpolation);

[net_conv, net_fc, opts] = mdnet_init(img, net);

opts.averageImage=zeros(1,1,3); 
if isfield(net_conv,'meta')
    opts.averageImage(1)=mean(net_conv.meta.normalization.averageImage(:,1));
    opts.averageImage(2)=mean(net_conv.meta.normalization.averageImage(:,2));
    opts.averageImage(3)=mean(net_conv.meta.normalization.averageImage(:,3));
end

img = bsxfun(@minus, img, opts.averageImage) ; 


%% Train a bbox regressor
if(opts.bbreg)
    pos_examples = gen_samples('uniform_aspect', targetLoc, opts.bbreg_nSamples*10, opts, 0.3, 10);
    r = overlap_ratio(pos_examples,targetLoc);
    pos_examples = pos_examples(r>0.6,:);
    pos_examples = pos_examples(randsample(end,min(opts.bbreg_nSamples,end)),:);
    examples = [pos_examples(:,1) pos_examples(:,2) pos_examples(:,1)+pos_examples(:,3) pos_examples(:,2)+pos_examples(:,4)];
    exa=examples';
    exa = bsxfun(@times, exa - 1, factor) + 1 ;
    exa = single(exa);
    rois = [ones(1,size(exa,2)) ; exa] ;
    % extract conv3 features
    if opts.useGpu
        img=gpuArray(img);
        rois=gpuArray(rois);
    end   
    feat_conv = mdnet_features_convX(net_conv, img, rois, opts);
    
    X = permute(gather(feat_conv),[4,3,1,2]);
    X = X(:,:);
    bbox = pos_examples;
    bbox_gt = repmat(targetLoc,size(pos_examples,1),1);
    bbox_reg = train_bbox_regressor(X, bbox, bbox_gt);
end

%% Extract training examples
fprintf('  extract features...\n');

% draw positive/negative samples
pos_examples = gen_samples('gaussian', targetLoc, opts.nPos_init*2, opts, 0.1, 5);
r = overlap_ratio(pos_examples,targetLoc);
pos_examples = pos_examples(r>opts.posThr_init,:);
pos_examples = pos_examples(randsample(end,min(opts.nPos_init,end)),:);

neg_examples = [gen_samples('uniform', targetLoc, opts.nNeg_init, opts, 1, 10);...
    gen_samples('whole', targetLoc, opts.nNeg_init, opts)];
r = overlap_ratio(neg_examples,targetLoc);
neg_examples = neg_examples(r<opts.negThr_init,:);
neg_examples = neg_examples(randsample(end,min(opts.nNeg_init,end)),:);

examples = [pos_examples; neg_examples];
pos_idx = 1:size(pos_examples,1);
neg_idx = (1:size(neg_examples,1)) + size(pos_examples,1);

examples = [examples(:,1) examples(:,2) examples(:,1)+examples(:,3) examples(:,2)+examples(:,4)];
exa=examples';
exa = bsxfun(@times, exa - 1, factor) + 1 ;
exa = single(exa);
rois = [ones(1,size(exa,2)) ; exa] ;
% extract conv3 features
if opts.useGpu
    img=gpuArray(img);
    rois=gpuArray(rois);
end
feat_conv = mdnet_features_convX(net_conv, img, rois, opts);
pos_data = feat_conv(:,:,:,pos_idx);
neg_data = feat_conv(:,:,:,neg_idx);


%% Learning CNN
fprintf('  training cnn...\n');
net_fc = mdnet_finetune_hnm(net_fc,pos_data,neg_data,opts,...
    'maxiter',opts.maxiter_init,'learningRate',opts.learningRate_init);

%% Initialize displayots
if display
    figure(2);
    set(gcf,'Position',[200 100 600 400],'MenuBar','none','ToolBar','none'); 
    hd = imshow(ori_img,'initialmagnification','fit'); hold on;
    rectangle('Position', targetLoc, 'EdgeColor', [1 0 0], 'Linewidth', 3);
    set(gca,'position',[0 0 1 1]);
    text(10,10,'1','Color','y', 'HorizontalAlignment', 'left', 'FontWeight','bold', 'FontSize', 30);
    hold off;
    drawnow;
end

%% Prepare training data for online update
total_pos_data = cell(1,1,1,nFrames);
total_neg_data = cell(1,1,1,nFrames);

neg_examples = gen_samples('uniform', targetLoc, opts.nNeg_update*2, opts, 2, 5);
r = overlap_ratio(neg_examples,targetLoc);
neg_examples = neg_examples(r<opts.negThr_init,:);
neg_examples = neg_examples(randsample(end,min(opts.nNeg_update,end)),:);

examples = [pos_examples; neg_examples];

examples = [examples(:,1) examples(:,2) examples(:,1)+examples(:,3) examples(:,2)+examples(:,4)];
exa=examples';
exa = bsxfun(@times, exa - 1, factor) + 1 ;
exa = single(exa);
rois = [ones(1,size(exa,2)) ; exa] ;

pos_idx = 1:size(pos_examples,1);
neg_idx = (1:size(neg_examples,1)) + size(pos_examples,1);
if opts.useGpu
    img=gpuArray(img);
    rois=gpuArray(rois);
end
feat_conv = mdnet_features_convX(net_conv,img, rois, opts);
total_pos_data{1} = feat_conv(:,:,:,pos_idx);
total_neg_data{1} = feat_conv(:,:,:,neg_idx);

success_frames = 1;
trans_f = opts.trans_f;
scale_f = opts.scale_f;

%% Main loop
for To = 2:nFrames;
    fprintf('\n');
    fprintf('Processing frame %d/%d... \n', To, nFrames);
  
    spf = tic;
   
    img = imread(images{To});
    ori_img=img;
    if(size(img,3)==1), img = cat(3,img,img,img); end
    img=single(img);
    % Resize images and boxes to a size compatible with the network.
    imageSize = size(img) ;
    h=imageSize(1);
    w=imageSize(2);
    factor = max(opts.scale/h,opts.scale/w);
    if any([h*factor,w*factor]>opts.max_scale)
        factor = min(opts.max_scale/h,opts.max_scale/w);
    end
    img = imresize(img,factor,'Method',opts.interpolation);
    
    img = bsxfun(@minus, img, opts.averageImage) ; 
   
    %% Estimation
    
    % draw target candidates
    samples = gen_samples('gaussian', targetLoc, opts.nSamples *2, opts, trans_f, scale_f);
    example = [samples(:,1) samples(:,2) samples(:,1)+samples(:,3) samples(:,2)+samples(:,4)];
    exa=example';
    exa = bsxfun(@times, exa - 1, factor) + 1 ;
    exa = single(exa);
    rois = [ones(1,size(exa,2)) ; exa] ;
    if opts.useGpu
        img=gpuArray(img);
        rois=gpuArray(rois);
    end
    feat_conv = mdnet_features_convX(net_conv, img, rois, opts);
          
    % evaluate the candidates
    feat_fc = mdnet_features_fcX(net_fc, feat_conv, opts);
    feat_fc = squeeze(feat_fc)';
    [scores,idx] = sort(feat_fc(:,2),'descend');
    target_score = mean(scores(1:5));
    targetLoc = round(mean(samples(idx(1:5),:)));
    
    fprintf('traget score %.3f\n', target_score);
    
    result(To,:) = targetLoc;   
    
    if(target_score<0)
        trans_f = min(1.5, 1.1*trans_f);
    else
        trans_f = opts.trans_f;
    end  
    
    %% Prepare training data
   
    if(target_score>0)  
        pos_examples = gen_samples('gaussian', targetLoc, opts.nPos_update*2, opts, 0.1, 5);
        r = overlap_ratio(pos_examples,targetLoc);
        pos_examples = pos_examples(r>opts.posThr_update,:);
        pos_examples = pos_examples(randsample(end,min(opts.nPos_update,end)),:);
        
        neg_examples = gen_samples('uniform', targetLoc, opts.nNeg_update*2, opts, 2, 5);
        r = overlap_ratio(neg_examples,targetLoc);
        neg_examples = neg_examples(r<opts.negThr_update,:);
        neg_examples = neg_examples(randsample(end,min(opts.nNeg_update,end)),:);
        
        examples = [pos_examples; neg_examples];
        examples = [examples(:,1) examples(:,2) examples(:,1)+examples(:,3) examples(:,2)+examples(:,4)];
        exa=examples';
        exa = bsxfun(@times, exa - 1, factor) + 1 ;
        exa = single(exa);
        rois = [ones(1,size(exa,2)) ; exa] ;
             
        pos_idx = 1:size(pos_examples,1);
        neg_idx = (1:size(neg_examples,1)) + size(pos_examples,1);
        if opts.useGpu
            img=gpuArray(img);
            rois=gpuArray(rois);
        end
        feat_conv = mdnet_features_convX(net_conv, img, rois, opts);
        total_pos_data{To} = feat_conv(:,:,:,pos_idx);
        total_neg_data{To} = feat_conv(:,:,:,neg_idx);
        
        success_frames = [success_frames, To];
        if(numel(success_frames)>opts.nFrames_long)
            total_pos_data{success_frames(end-opts.nFrames_long)} = single([]);
        end
        if(numel(success_frames)>opts.nFrames_short)
            total_neg_data{success_frames(end-opts.nFrames_short)} = single([]);
        end
    else
        total_pos_data{To} = single([]);
        total_neg_data{To} = single([]);
    end
   
    
    %% Network update
    if((mod(To,opts.update_interval)==0 || target_score<0) && To~=nFrames && To>0)
        if (target_score<0) % short-term update
            pos_data = cell2mat(total_pos_data(success_frames(max(1,end-opts.nFrames_short+1):end)));
        else % long-term update
            pos_data = cell2mat(total_pos_data(success_frames(max(1,end-opts.nFrames_long+1):end)));
        end
        neg_data = cell2mat(total_neg_data(success_frames(max(1,end-opts.nFrames_short+1):end)));
        
%         fprintf('\n');
        [net_fc,loss] = mdnet_finetune_hnm(net_fc,pos_data,neg_data,opts,...
            'maxiter',opts.maxiter_update,'learningRate',opts.learningRate_update);
    end
       
    spf = toc(spf);
    fprintf('speed:%f seconds\n',spf);
%     
    %% Display
    if display
        hc = get(gca, 'Children'); delete(hc(1:end-1));
        set(hd,'cdata',ori_img); hold on;
        rectangle('Position', result(To,:), 'EdgeColor', [1 0 0], 'Linewidth', 3);
        set(gca,'position',[0 0 1 1]);
        text(10,10,num2str(To),'Color','y', 'HorizontalAlignment', 'left', 'FontWeight','bold', 'FontSize', 30); 
        hold off;
        drawnow;
    end
end