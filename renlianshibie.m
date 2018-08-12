clear;
clc;
tic;
addpath('D:\mastershaw\moshizuoye\gailiang');
prompt='ѡ���������ݼ��� ��1.YALE��2.ORL��\n';
shujuji= input(prompt);

if     shujuji==1
           disp('����YALE������...')
           filePath='D:\mastershaw\moshizuoye\gailiang\YALE';
elseif shujuji==2
           disp('����ORL������...')
           filePath='D:\mastershaw\moshizuoye\gailiang\ORL';
end
listening=dir(filePath);
renlianshu=length(listening)-2;
%����ѵ��ͼ��
train_data=[];   
for i=1:renlianshu
    for j=1:5
      filepath=strcat(filePath,'\s',num2str(i),'\',num2str(j),'.png');
      a=imread(filepath);    %strcat����ַ���   num2str��ֵת�����ַ���
      [m,n]=size(a);
      b=a(1:m*n); % b����ʸ�� 1��N������N��10304����ȡ˳�������к��У������ϵ��£�������
      b=double(b);   %double
      
      train_data=[train_data; b];  % allsamples ��һ��M * N ����allsamples ��ÿһ�����ݴ���һ��ͼƬ������M��200
    end
end
%���в���ͼ��
test_data=[];   
for i=1:renlianshu
    for j=6:10 %����40 x 5 ������ͼ��
        a=imread(strcat(filePath,'\s',num2str(i),'\',num2str(j),'.png')); 
        [m1,n1]=size(a);
        b=a(1:m1*n1);
        b=double(b);     %1��10304�׾���
        test_data=[test_data; b];
    end
end

disp('��һ��Ԥ����...')
[train_data,test_data] = norm_data(train_data,test_data,0,1);
disp('pca��ά...')
options=[];
options.PCARatio=.95;            %0.95
[eigvector, eigvalue, meanData, new_data] = PCA(train_data, options);  
tcoor=(test_data-ones(size(test_data,1),1)*meanData)*eigvector;     %tcoor=�����������м�ȥƽ��������ͶӰ;new_data=train_final*eigvector

%�����ؽ�
% approx=meanData;  
% for i=1:k  %kΪpca��ά��
%     approx=approx+pcaface(1,i)*V(:,i)';%pcaface�ĵ�һ������������Ҫ�ؽ�������������Ե�һ���˵ĵ�һ�����������ؽ�  
% end  
% disp('�����ؽ�...')  
% figure  
% B=reshape(approx',m,n);  
% imshow(B,[])  
% 
% disp('��ʾ���ɷ���...')  
% visualize(V)%��ʾ��������  
  

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%     
disp('ѵ�������Ͳ��������ı�ǩ...')
train_data= new_data;%ѵ������
n_train=size(train_data,1);
train_data_label=zeros(n_train,1);    % ��Ϊ�궨ѵ��Ŀ�����𣬴�1~40
for i=1:n_train
    if rem(i,5)~=0                % ���i�ǵ�23������������Ӧ�����ڵ�5�ࣨ��ÿ����5��������                   
        train_data_label(i)=floor(i/5)+1;       
    else
        train_data_label(i)=floor(i/5);
    end
end

test_data = tcoor;% ��������
n_test=size(test_data,1);
test_data_label=zeros(n_test,1);     % ��Ϊ�궨����Ŀ�����𣬴�1~40
for i=1:n_test
    if rem(i,5)~=0         
        test_data_label(i)=floor(i/5)+1;
    else
        test_data_label(i)=floor(i/5);
    end
end

disp('ѵ�������...')

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
prompt1='ѡ�񷽰��� ��1.ֱ�ӵ���libsvm��2.����Ѱ�� \n';
flag = input(prompt1);
if flag == 1
   disp('����1 ֱ�ӵ���libsvm')
   prompt2='ѡ��˺���:0���Ժ�,1����ʽ�ˣ�2�������������˹����3sigmod�˺��� \n';
   hehanshu1 = input(prompt2);
   cmd = [' -t ',num2str(hehanshu1)];
elseif flag == 2
   disp('����2 ����c��gѰ��ѡ��...')
   prompt3='ѡ��˺���:0���Ժ�,1����ʽ�ˣ�2�������������˹����3sigmod�˺��� \n';
    hehanshu2 = input(prompt3);
   [bestacc,bestc,bestg]=SVMcgForClass(train_data_label,train_data,-10,10,-10,10);
    cmd = ['-c ',num2str(bestc),' -g ',num2str(bestg),' -t ',num2str(hehanshu2)];
end
disp('...')
model = svmtrain(train_data_label, train_data,cmd);
[ptrain_label, train_accuracy,decision_values] = svmpredict(train_data_label, train_data, model);%�������ȷ�ʡ��ع�ľ��������ع��ƽ�����ϵ��
 disp('ѵ������ȷ�ʡ��ع�ľ��������ع��ƽ�����ϵ��')
 train_accuracy
[ptest_label, test_accuracy,decision_values] = svmpredict(test_data_label, test_data, model);
 disp('���Լ���ȷ�ʡ��ع�ľ��������ع��ƽ�����ϵ��')
 test_accuracy
toc;


% ����˵��[bestacc,bestc,bestg] = SVMcg(train_label,train,cmin,cmax,gmin,gmax,v,cstep,gstep,accstep)
% 
% train_label:ѵ������ǩ.Ҫ����libsvm��������Ҫ��һ��.
% train:ѵ����.Ҫ����libsvm��������Ҫ��һ��.
% cmin:�ͷ�����c�ı仯��Χ����Сֵ(ȡ��2Ϊ�׵Ķ�����),�� c_min = 2^(cmin).Ĭ��Ϊ -5
% cmax:�ͷ�����c�ı仯��Χ�����ֵ(ȡ��2Ϊ�׵Ķ�����),�� c_max = 2^(cmax).Ĭ��Ϊ 5
% gmin:����g�ı仯��Χ����Сֵ(ȡ��2Ϊ�׵Ķ�����),�� g_min = 2^(gmin).Ĭ��Ϊ -5
% gmax:����g�ı仯��Χ����Сֵ(ȡ��2Ϊ�׵Ķ�����),�� g_min = 2^(gmax).Ĭ��Ϊ 5
% 
% v:cross validation�Ĳ���,�������Լ���Ϊ�����ֽ���cross validation.Ĭ��Ϊ 3
% cstep:����c�����Ĵ�С.Ĭ��Ϊ 1
% gstep:����g�����Ĵ�С.Ĭ��Ϊ 1
% accstep:�����ʾ׼ȷ��ͼʱ�Ĳ�����С. Ĭ��Ϊ 1.5