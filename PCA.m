function [eigvector, eigvalue, meanData, new_data] = PCA(data, options)
%PCA	Principal component analysis
%	Usage:
%	[EIGVECTOR, EIGVALUE, MEANDATA, NEW_DATA] = PCA(DATA, OPTIONS)
%
%	DATA: Rows of vectors of data points
%   options - Struct value in Matlab. The fields in options
%             that can be set:
%           ReducedDim   -  The dimensionality of the
%                           reduced subspace. If 0,
%                           all the dimensions will be
%                           kept. Default is 0.
%           PCARatio     -  The percentage of principal
%                           component kept. The percentage is
%                           calculated based on the
%                           eigenvalue. Default is 1
%                           (100%, all the non-zero
%                           eigenvalues will be kept.
%          Please do not set both these two fields. If both of them are
%          set, PCARatio will be used. 
%          
%   MEANDATA: Mean of all the data. 
%	NEW_DATA: The data after projection (mean removed)
%	EIGVECTOR: Each column of this matrix is a eigenvector of DATA'*DATA
%	EIGVALUE: Eigenvalues of DATA'*DATA
%
%	Examples:
% 			data = rand(7,10);
% 			options = [];
% 			options.ReducedDim = 4;
% 			[eigvector,eigvalue,meanData,new_data] = PCA(data,options);
% 
% 			data = rand(7,10);
% 			options = [];
% 			options.PCARatio = 0.98;
% 			[eigvector,eigvalue,meanData,new_data] = PCA(data,options);
%
% Note: Be aware of the "mean". For classification, you can use this code by:
% 	
%			fea_Train = fea(trainIdx,:);
%
%			[eigvector, eigvalue] = PCA(fea_Train, options)
%			fea_New = fea*eigvector;  训练数据
%		or
%			[eigvector, eigvalue, meanData] = PCA(fea_Train, options)
%			fea_New_2 = (fea - repmat(meanData,nSmp,1))*eigvector;  训练数据
%
%	 Then classification is then performed on "fea_new" or "fea_new_2". 
%	 Since we use Euclidean distance, the result will be the same for 
%	 nearest neighbor classifier on "fea_new" or "fea_new_2". 
%
%    If you call PCA by:
%			[eigvector, eigvalue, meanData, fea_Train_new] = PCA(fea_Train, options);
%	  Since "fea_Train_new" is "mean removed", you should also subtract the "meanData" 
%	  from each testing example, like
%			fea_Test = fea(testIdx,:);          测试数据
%			fea_Test_new = (fea_Test - repmat(meanData,nSmpTest,1))*eigvector; 测试数据

%
% 
%    Written by Deng Cai (dengcai@gmail.com), April/2004, Feb/2006

if (~exist('options','var'))
   options = [];
else                                       %class(A) 返回A的类型名称
   if ~strcmpi(class(options),'struct')  % strcmpi : Compare strings ignoring case.
                                         % TF = STRCMPI(S1,S2) compares the  strings S1 and S2 and returns logical 1(true)
                                         % if they are the same except for case, and returns logical 0 (false)  otherwise.
       error('parameter error!');
   end
end

bRatio = 0;          %？   何处定义options 是一结构体   上段？
if isfield(options,'PCARatio')          %测试某字符串是否为指定结构体的字段  
    bRatio = 1;
    eigvector_n = min(size(data));      
elseif isfield(options,'ReducedDim')
    eigvector_n = options.ReducedDim;
else
    eigvector_n = min(size(data));
end
    

[nSmp, nFea] = size(data);
% 
% meanValue = mean (data')';
% data = data - meanValue * ones (1,size (data, 2));     %？

meanData = mean(data);
data = data - repmat(meanData,nSmp,1);   %B = repmat(A,M,N) The size of B is [size(A,1)*M, size(A,2)*N].
                           %?              % The statement repmat(A,N) creates an N-by-N    tiling.

if nSmp >= nFea                      %行数大于列数
    ddata = data'*data;
    ddata = (ddata + ddata')/2;  %?
    if issparse(ddata)                    %测试是否是稀疏矩阵 
        ddata = full(ddata);              %把稀疏矩阵转换为普通矩阵
    end

    if size(ddata, 1) > 100 & eigvector_n < size(ddata, 1)/2  % using eigs to speed up!    ?如果是稀疏矩阵，则用eigs求特征值和特征向量
        option = struct('disp',0);    %？  创立一个结构？
        [eigvector, d] = eigs(ddata,eigvector_n,'la',option); %EIGS(A,B,K,SIGMA)    return K eigenvalues. 
                                                            %？                     %If SIGMA is:'LA' or 'SA' - Largest or Smallest Algebraic
        eigvalue = diag(d);
    else
        [eigvector, d] = eig(ddata);                                                  %？ 否则，若为普通矩阵，则用eig求特征值和特征向量
        eigvalue = diag(d);
        % ====== Sort based on descending order         降序排列
        [junk, index] = sort(-eigvalue);                     % ？junk中是无用的负特征值
        eigvalue = eigvalue(index);                             % 将已有的eigvalue 、eigvector按index排序
        eigvector = eigvector(:, index);
    end
    
    clear ddata;
    maxEigValue = max(abs(eigvalue));                            % 获取最大特征值
    eigIdx = find(abs(eigvalue)/maxEigValue < 1e-12);        % ？  1e-12  ？eigIdx？
    eigvalue (eigIdx) = [];
    eigvector (:,eigIdx) = [];

else	% This is an efficient method which computes the eigvectors of   %行数小于列数
	% of A*A^T (instead of A^T*A) first, and then convert them back to
	% the eigenvectors of A^T*A.
    if nSmp > 700                           
        ddata = zeros(nSmp,nSmp);
        for i = 1:ceil(nSmp/100)         %向不小于a的最接近整数取整  ceil(4.5)结果为5; ceil(-4.5)结果为-4
            if i == ceil(nSmp/100)    %？
                ddata((i-1)*100+1:end,:) = data((i-1)*100+1:end,:)*data';     %？
            else
                ddata((i-1)*100+1:i*100,:) = data((i-1)*100+1:i*100,:)*data';
            end
        end
    elseif nSmp > 400
        ddata = zeros(nSmp,nSmp);
        for i = 1:ceil(nSmp/200)      
            if i == ceil(nSmp/200)   %？
                ddata((i-1)*200+1:end,:) = data((i-1)*200+1:end,:)*data';    %？
            else
                ddata((i-1)*200+1:i*200,:) = data((i-1)*200+1:i*200,:)*data';
            end
        end
    else
        ddata = data*data';
    end
   
    ddata = (ddata + ddata')/2;
    if issparse(ddata)
        ddata = full(ddata);
    end
    
    if size(ddata, 1) > 100 & eigvector_n < size(ddata, 1)/2  % using eigs to speed up!
        option = struct('disp',0);
        [eigvector1, d] = eigs(ddata,eigvector_n,'la',option);
        eigvalue = diag(d);
    else
        [eigvector1, d] = eig(ddata);
        
        eigvalue = diag(d);
        % ====== Sort based on descending order
        [junk, index] = sort(-eigvalue);
        eigvalue = eigvalue(index);
        eigvector1 = eigvector1(:, index);
    end

    clear ddata;
      
    maxEigValue = max(abs(eigvalue));
    eigIdx = find(abs(eigvalue)/maxEigValue < 1e-12);
    eigvalue (eigIdx) = [];
    eigvector1 (:,eigIdx) = [];

    eigvector = data'*eigvector1;		% Eigenvectors of A^T*A
    if nargout ~= 4 clear data; end%%%%%%这是我添加的!!!!!             ？nargout
    clear eigvector1;
%  	eigvector = eigvector.*repmat(1./sqrt(eigvalue+eps),1,nFea)'; % Normalization 
	eigvector = eigvector*diag(1./(sum(eigvector.^2).^0.5)); % Normalization  正则化

end

%计算精度
if bRatio                 %第76行定义                 %？少东西吗   bRatio=?
    if options.PCARatio >= 1 | options.PCARatio <= 0
        idx = length(eigvalue);
    else
        sumEig = sum(eigvalue);
        sumEig = sumEig*options.PCARatio;
        sumNow = 0;
        for idx = 1:length(eigvalue)
            sumNow = sumNow + eigvalue(idx);
            if sumNow >= sumEig
                break;
            end
        end
    end

    eigvalue = eigvalue(1:idx);
    eigvector = eigvector(:,1:idx);
else
    if eigvector_n < length(eigvalue)      %第77行定义
        eigvalue = eigvalue(1:eigvector_n);
        eigvector = eigvector(:, 1:eigvector_n);
    end
end

if nargout == 4           % ？nargout
    new_data = data*eigvector;
end

