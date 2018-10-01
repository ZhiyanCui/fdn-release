
function [input,rois,labels,bbox_labels]=get_batch(roidb, batch,opts)
% -------------------------------------------------------------------------
if isfield(opts,'averageImage')
    averageImage=opts.averageImage;
else
    averageImage=zeros(1,1,3);
end
  averageImage=single(averageImage);

rois=[];
labels=[];
bbox_labels=[];
n_batch=numel(batch);
for i=1:n_batch
    image=imread(roidb(batch(i)).img_path);
    image=single(image);
    % Remove the average color from the input image.
    image = bsxfun(@minus, image, averageImage) ;    
    
    im_size=size(image);
    h=im_size(1);
    w=im_size(2);

    factor = max(opts.scale/h,opts.scale/w);
    if any([h*factor,w*factor]>opts.max_scale)
        factor = min(opts.max_scale/h,opts.max_scale/w);
    end
    %if abs(factor-1)>1e-3
    image = imresize(image,factor,'Method',opts.interpolation);
    %end
    %minus the average of the image
    %image = bsxfun(@minus, image, opts.averageImage);
    if(i==1)
        im_size=size(image);
        input=zeros(im_size(1),im_size(2),im_size(3),opts.batch_frames);
        input(:,:,:,i)=image;
    else
        input(:,:,:,i)=image;
    end
    
    pos_s=roidb(batch(i)).pos_boxes;
    neg_s=roidb(batch(i)).neg_boxes;
    pos_s(:,3)=pos_s(:,1)+pos_s(:,3);
    pos_s(:,4)=pos_s(:,2)+pos_s(:,4);
    neg_s(:,3)=neg_s(:,1)+neg_s(:,3);
    neg_s(:,4)=neg_s(:,2)+neg_s(:,4);
    
    %top-left corner and bottom-right corner
    pos_s=bsxfun(@times, pos_s - 1, factor) + 1;
    neg_s=bsxfun(@times, neg_s - 1, factor) + 1;
 
    pos_s=pos_s(1:opts.batch_pos/opts.batch_frames,:);
    neg_s=neg_s(1:opts.batch_neg/opts.batch_frames,:);
    
    number=(ones(size(pos_s,1)+size(neg_s,1),1)).*i;
    
    roi = cat(1,pos_s,neg_s);
    roi = cat(2,number,roi);
    roi = single(roi');
    
    rois=cat(2,rois,roi);

%     pos_labels=ones(size(pos_s,1),1);
%     neg_labels=ones(size(neg_s,1),1) * 2;

    pos_labels=ones(size(pos_s,1),1) * 2;
    neg_labels=ones(size(neg_s,1),1) * 1;


    label= cat(1,pos_labels,neg_labels);
    labels=cat(1,labels,label);
    
    if opts.bbox_loss
        gt=roidb(batch(i)).gt;
        gt(3)=gt(1)+gt(3);
        gt(4)=gt(2)+gt(4);
        gt=bsxfun(@times, gt - 1, factor) + 1;
        gt(3)=gt(3)-gt(1);
        gt(4)=gt(4)-gt(2);

        gt(1)=gt(1)+gt(3)/2;
        gt(2)=gt(1)+gt(4)/2;
        
        %bbox_label=zeros(size(pos_s,1)+size(neg_s,1),4);       
        %bbox_label(1:size(pos_s,1),1:4)=repmat(gt,size(pos_s,1),1);
        
        t=zeros(size(pos_s,1)+size(neg_s,1),4);
        
        p_s=pos_s;
        n_s=neg_s;
        
        p_s(:,3)=p_s(:,3)-p_s(:,1);
        p_s(:,4)=p_s(:,4)-p_s(:,2);
        p_s(:,1)=p_s(:,1)+p_s(:,3)/2;
        p_s(:,2)=p_s(:,1)+p_s(:,4)/2;
        
        tx=(gt(1)-p_s(:,1))./p_s(:,3);
        ty=(gt(2)-p_s(:,2))./p_s(:,4);
        tw=log(gt(3)./p_s(:,3));
        th=log(gt(4)./p_s(:,4));
        
        t(1:size(p_s,1),1)=tx;
        t(1:size(p_s,1),2)=ty;
        t(1:size(p_s,1),3)=tw;
        t(1:size(p_s,1),4)=th;
          

        bbox_labels=cat(1,bbox_labels,t);     
        
    end
 
end

labels=single(labels);
input=single(input);
bbox_labels=single(bbox_labels);
if opts.useGpu
  input = gpuArray(input) ;
  rois = gpuArray(rois) ;
  labels = gpuArray(labels);
  if opts.bbox_loss
      bbox_labels = gpuArray(bbox_labels);
  end

end
% inputs = {'input', input,'rois',rois, 'label', labels, 'targets', targets, ...
%   'instance_weights', instance_weights};
% 
% inputs = {'input', input,'rois',rois, ['label' num2str(seq_id)], labels, ['targets' num2str(seq_id)], targets, ...
%   ['instance_weights' num2str(seq_id)], instance_weights};
end


