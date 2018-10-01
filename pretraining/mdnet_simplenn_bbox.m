%function res = mdnet_simplenn(net, x, k, dzdy, res, varargin)
function res = mdnet_simplenn_bbox(usebbox, rois, net, x, k, dzdy, res, varargin)



opts.res = [] ;
opts.conserveMemory = false ;
opts.sync = false ;
opts.disableDropout = false ;
opts.freezeDropout = false ;
opts = vl_argparse(opts, varargin);


if usebbox 
    n = numel(net.layers) ;
else
    n= numel(net.layers) -2;
end

if (nargin <= 2) || isempty(dzdy)
  doder = false ;
else
  doder = true ;
end

gpuMode = isa(x, 'gpuArray') ;

% if nargin <= 3 || isempty(res)
%   res = struct(...
%     'x', cell(1,n+2), ...
%     'dzdx', cell(1,n+2), ...
%     'dzdw', cell(1,n+2), ...
%     'aux', cell(1,n+2), ...
%     'time', num2cell(zeros(1,n+2)), ...
%     'backwardTime', num2cell(zeros(1,n+2))) ;
% end


if nargin <= 3 || isempty(res)
  res = struct(...
    'x', cell(1,n+1), ...
    'dzdx', cell(1,n+1), ...
    'dzdw', cell(1,n+1), ...
    'aux', cell(1,n+1), ...
    'time', num2cell(zeros(1,n+1)), ...
    'backwardTime', num2cell(zeros(1,n+1))) ;
end
res(1).x = x ;

for i=1:n
  l = net.layers{i} ;
  res(i).time = tic ;
  if((i == (n - 1)) && usebbox)
      res(i+1).x = vl_nnconv(res(i-2).x, l.filters, l.biases, 'pad', l.pad, 'stride', l.stride) ;
      continue;
  end
%    if(i == (n))
%       res(i+2).x = vl_nnsoftmaxloss(res(i).x(:,:,2*k-1:2*k,:), l.class) ;
%       continue;
%   end
   
  switch l.type
    case 'conv'
      res(i+1).x = vl_nnconv(res(i).x, l.filters, l.biases, 'pad', l.pad, 'stride', l.stride) ;
    case 'pool'
      res(i+1).x = vl_nnpool(res(i).x, l.pool, 'pad', l.pad, 'stride', l.stride, 'method', l.method) ;
    case 'roialign'
      res(i+1).x = roialign(res(i).x,rois,l.subdivisions,l.transform);
    case 'roipool'
      res(i+1).x = vl_nnroipool(res(i).x,rois,'subdivisions',l.subdivisions,...
           'method',l.method,'transform',l.transform) ;    
    case 'normalize'
      res(i+1).x = vl_nnnormalize(res(i).x, l.param) ;
    case 'softmax'
      res(i+1).x = vl_nnsoftmax(res(i).x) ;
    case 'loss'
      res(i+1).x = vl_nnloss(res(i).x, l.class) ;
    case 'softmaxloss'
      res(i+1).x = vl_nnsoftmaxloss(res(i).x, l.class) ;
    case 'softmaxloss_k'
      res(i+1).x = vl_nnsoftmaxloss(res(i).x(:,:,2*k-1:2*k,:), l.class) ;
    case 'smoothloss'
        res(i+1).x = smoothloss(res(i).x(:,:,4*k-3:4*k,:), l.class,l.regression) ;
    case 'relu'
      res(i+1).x = vl_nnrelu(res(i).x) ;
    case 'noffset'
      res(i+1).x = vl_nnnoffset(res(i).x, l.param) ;
    case 'dropout'
      if opts.disableDropout
        res(i+1).x = res(i).x ;
      elseif opts.freezeDropout
        [res(i+1).x, res(i+1).aux] = vl_nndropout(res(i).x, 'rate', l.rate, 'mask', res(i+1).aux) ;
      else
        [res(i+1).x, res(i+1).aux] = vl_nndropout(res(i).x, 'rate', l.rate) ;
      end
    case 'custom'
      res(i+1) = l.forward(l, res(i), res(i+1)) ;
    case 'pass'
      % do nothing    
    otherwise
      error('Unknown layer type %s', l.type) ;
  end
  if opts.conserveMemory & ~doder & i < numel(net.layers) - 1
    % TODO: forget unnecesary intermediate computations even when
    % derivatives are required
    res(i).x = [] ;
  end
  if gpuMode & opts.sync
    % This should make things slower, but on MATLAB 2014a it is necessary
    % for any decent performance.
    wait(gpuDevice) ;
  end
  res(i).time = toc(res(i).time) ;
end

if doder
  res(n+1).dzdx = dzdy ;
  for i=n:-1:1
    l = net.layers{i} ;
    res(i).backwardTime = tic ;
    if((i == (n - 1)) && usebbox)
        [res(i).dzdx, res(i).dzdw{1}, res(i).dzdw{2}] = ...
            vl_nnconv(res(i-2).x, l.filters, l.biases, ...
                      res(i+1).dzdx, ...
                      'pad', l.pad, 'stride', l.stride) ;
                  
%         if opts.conserveMemory
%             res(i+1).dzdx = [] ;
%         end
        if gpuMode & opts.sync
            wait(gpuDevice) ;
        end
        res(i).backwardTime = toc(res(i).backwardTime) ;          
        continue;
    end
    if((i == (n-4)) && usebbox)
        if opts.disableDropout
          res(i).dzdx = res(i+1).dzdx + res(i+3).dzdx ;
        else
          res(i).dzdx = vl_nndropout(res(i).x, res(i+1).dzdx + res(i+3).dzdx , 'mask', res(i+1).aux) ;
        end
        if opts.conserveMemory
            res(i+1).dzdx = [] ;
        end
        if gpuMode & opts.sync
            wait(gpuDevice) ;
        end
        res(i).backwardTime = toc(res(i).backwardTime) ;    
        continue;
    end
    
        
    switch l.type
      case 'conv'
        [res(i).dzdx, res(i).dzdw{1}, res(i).dzdw{2}] = ...
            vl_nnconv(res(i).x, l.filters, l.biases, ...
                      res(i+1).dzdx, ...
                      'pad', l.pad, 'stride', l.stride) ;
      case 'pool'
        res(i).dzdx = vl_nnpool(res(i).x, l.pool, res(i+1).dzdx, ...
          'pad', l.pad, 'stride', l.stride, 'method', l.method) ;
      case 'roialign'
        res(i).dzdx = roialign(res(i).x,rois,res(i+1).dzdx,l.subdivisions,l.transform);
      case 'roipool'
        res(i).dzdx = vl_nnroipool(res(i).x,rois,res(i+1).dzdx,'subdivisions',l.subdivisions,...
            'method',l.method,'transform',l.transform);    
      case 'normalize'
        res(i).dzdx = vl_nnnormalize(res(i).x, l.param, res(i+1).dzdx) ;
      case 'softmax'
        res(i).dzdx = vl_nnsoftmax(res(i).x, res(i+1).dzdx) ;
      case 'loss'
        res(i).dzdx = vl_nnloss(res(i).x, l.class, res(i+1).dzdx) ;
      case 'softmaxloss'
        %res(i).dzdx = vl_nnsoftmaxloss(res(i).x, l.class, res(i+1).dzdx) ;
        res(i).dzdx = vl_nnsoftmaxloss(res(i).x, l.class, 1) ;
      case 'softmaxloss_k'
        if(gpuMode), res(i).dzdx = gpuArray(zeros(size(res(i).x),'single'));
        else res(i).dzdx = zeros(size(res(i).x),'single'); end
        %res(i).dzdx(:,:,2*k-1:2*k,:) = vl_nnsoftmaxloss(res(i).x(:,:,2*k-1:2*k,:), l.class, res(i+1).dzdx) ;
        res(i).dzdx(:,:,2*k-1:2*k,:) = vl_nnsoftmaxloss(res(i).x(:,:,2*k-1:2*k,:), l.class, 1) ;
      case 'smoothloss'
          if(gpuMode), res(i).dzdx = gpuArray(zeros(size(res(i).x),'single'));
        else res(i).dzdx = zeros(size(res(i).x),'single'); end
        res(i).dzdx(:,:,4*k-3:4*k,:) = smoothloss(res(i).x(:,:,4*k-3:4*k,:), l.class,l.regression,res(i+1).dzdx) ;
      case 'relu'
        res(i).dzdx = vl_nnrelu(res(i).x, res(i+1).dzdx) ;
      case 'noffset'
        res(i).dzdx = vl_nnnoffset(res(i).x, l.param, res(i+1).dzdx) ;
      case 'dropout'
        if opts.disableDropout
          res(i).dzdx = res(i+1).dzdx ;
        else
          res(i).dzdx = vl_nndropout(res(i).x, res(i+1).dzdx, 'mask', res(i+1).aux) ;
        end
      case 'custom'
        res(i) = l.backward(l, res(i), res(i+1)) ;
      case 'pass'
      % do nothing    
    end
    if opts.conserveMemory
        if(i ~= (n-2))
             res(i+1).dzdx = [] ;
        end
     
    end
    if gpuMode & opts.sync
      wait(gpuDevice) ;
    end
    res(i).backwardTime = toc(res(i).backwardTime) ;
  end
end
