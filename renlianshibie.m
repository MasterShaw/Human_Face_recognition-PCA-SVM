clear;
clc;
tic;
addpath('D:\mastershaw\moshizuoye\gailiang');
prompt='选择人脸数据集（ ）1.YALE；2.ORL；\n';
shujuji= input(prompt);

if     shujuji==1
           disp('导入YALE人脸集...')
           filePath='D:\mastershaw\moshizuoye\gailiang\YALE';
elseif shujuji==2
           disp('导入ORL人脸集...')
           filePath='D:\mastershaw\moshizuoye\gailiang\ORL';
end
listening=dir(filePath);
renlianshu=length(listening)-2;
%所有训练图像
train_data=[];   
for i=1:renlianshu
    for j=1:5
      filepath=strcat(filePath,'\s',num2str(i),'\',num2str(j),'.png');
      a=imread(filepath);    %strcat组合字符串   num2str数值转换成字符串
      [m,n]=size(a);
      b=a(1:m*n); % b是行矢量 1×N，其中N＝10304，提取顺序是先列后行，即从上到下，从左到右
      b=double(b);   %double
      
      train_data=[train_data; b];  % allsamples 是一个M * N 矩阵，allsamples 中每一行数据代表一张图片，其中M＝200
    end
end
%所有测试图像
test_data=[];   
for i=1:renlianshu
    for j=6:10 %读入40 x 5 副测试图像
        a=imread(strcat(filePath,'\s',num2str(i),'\',num2str(j),'.png')); 
        [m1,n1]=size(a);
        b=a(1:m1*n1);
        b=double(b);     %1×10304阶矩阵
        test_data=[test_data; b];
    end
end

disp('归一化预处理...')
[train_data,test_data] = norm_data(train_data,test_data,0,1);
disp('pca降维...')
options=[];
options.PCARatio=.95;            %0.95
[eigvector, eigvalue, meanData, new_data] = PCA(train_data, options);  
tcoor=(test_data-ones(size(test_data,1),1)*meanData)*eigvector;     %tcoor=测试样本逐行减去平均脸后再投影;new_data=train_final*eigvector

%人脸重建
% approx=meanData;  
% for i=1:k  %k为pca后维数
%     approx=approx+pcaface(1,i)*V(:,i)';%pcaface的第一个参数代表你要重建的人脸，这里对第一个人的第一张脸脸进行重建  
% end  
% disp('人脸重建...')  
% figure  
% B=reshape(approx',m,n);  
% imshow(B,[])  
% 
% disp('显示主成分脸...')  
% visualize(V)%显示主分量脸  
  

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%     
disp('训练样本和测试样本的标签...')
train_data= new_data;%训练样本
n_train=size(train_data,1);
train_data_label=zeros(n_train,1);    % 人为标定训练目标的类别，从1~40
for i=1:n_train
    if rem(i,5)~=0                % 如果i是第23个样本，则它应该属于第5类（因每类有5个样本）                   
        train_data_label(i)=floor(i/5)+1;       
    else
        train_data_label(i)=floor(i/5);
    end
end

test_data = tcoor;% 测试样本
n_test=size(test_data,1);
test_data_label=zeros(n_test,1);     % 人为标定测试目标的类别，从1~40
for i=1:n_test
    if rem(i,5)~=0         
        test_data_label(i)=floor(i/5)+1;
    else
        test_data_label(i)=floor(i/5);
    end
end

disp('训练与测试...')

% -s svm_type : set type of SVM (default 0)
% 	0 -- C-SVC		(multi-class classification)
% 	1 -- nu-SVC		(multi-class classification)
% 	2 -- one-class SVM
% 	3 -- epsilon-SVR	(regression)
% 	4 -- nu-SVR		(regression)
% -t kernel_type : set type of kernel function (default 2)
% 	0 -- linear: u'*v
% 	1 -- polynomial: (gamma*u'*v + coef0)^degree
% 	2 -- radial basis function: exp(-gamma*|u-v|^2)
% 	3 -- sigmoid: tanh(gamma*u'*v + coef0)
% 	4 -- precomputed kernel (kernel values in training_instance_matrix)
% -d degree : set degree in kernel function (default 3)
% -g gamma : set gamma in kernel function (default 1/num_features)
% -c cost : set the parameter C of C-SVC, epsilon-SVR, and nu-SVR (default 1)
prompt1='选择方案（ ）1.直接调用libsvm或2.参数寻优 \n';
flag = input(prompt1);
if flag == 1
   disp('方案1 直接调用libsvm')
   prompt2='选择核函数:0线性核,1多项式核，2径向基函数（高斯），3sigmod核函数 \n';
   hehanshu1 = input(prompt2);
   cmd = [' -t ',num2str(hehanshu1)];
elseif flag == 2
   disp('方案2 参数c和g寻优选择...')
   prompt3='选择核函数:0线性核,1多项式核，2径向基函数（高斯），3sigmod核函数 \n';
    hehanshu2 = input(prompt3);
   [bestacc,bestc,bestg]=SVMcgForClass(train_data_label,train_data,-10,10,-10,10);
    cmd = ['-c ',num2str(bestc),' -g ',num2str(bestg),' -t ',num2str(hehanshu2)];
end
disp('...')
model = svmtrain(train_data_label, train_data,cmd);
[ptrain_label, train_accuracy,decision_values] = svmpredict(train_data_label, train_data, model);%分类的正确率、回归的均方根误差、回归的平方相关系数
 disp('训练集正确率、回归的均方根误差、回归的平方相关系数')
 train_accuracy
[ptest_label, test_accuracy,decision_values] = svmpredict(test_data_label, test_data, model);
 disp('测试集正确率、回归的均方根误差、回归的平方相关系数')
 test_accuracy
toc;


% 函数说明[bestacc,bestc,bestg] = SVMcg(train_label,train,cmin,cmax,gmin,gmax,v,cstep,gstep,accstep)
% 
% train_label:训练集标签.要求与libsvm工具箱中要求一致.
% train:训练集.要求与libsvm工具箱中要求一致.
% cmin:惩罚参数c的变化范围的最小值(取以2为底的对数后),即 c_min = 2^(cmin).默认为 -5
% cmax:惩罚参数c的变化范围的最大值(取以2为底的对数后),即 c_max = 2^(cmax).默认为 5
% gmin:参数g的变化范围的最小值(取以2为底的对数后),即 g_min = 2^(gmin).默认为 -5
% gmax:参数g的变化范围的最小值(取以2为底的对数后),即 g_min = 2^(gmax).默认为 5
% 
% v:cross validation的参数,即给测试集分为几部分进行cross validation.默认为 3
% cstep:参数c步进的大小.默认为 1
% gstep:参数g步进的大小.默认为 1
% accstep:最后显示准确率图时的步进大小. 默认为 1.5