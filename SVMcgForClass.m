function [bestacc,bestc,bestg,bestr] = SVMcgForClass(train_label,train,cmin,cmax,gmin,gmax,v,cstep,gstep,accstep)
%last modified 2009.8.23
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
%[������Щ������ҿ��Ը������ڴﵽ���Ч��,Ҳ�ɲ�����Ĭ��ֵ]
%% about the ption [bestacc,bestc,bestg] = SVMcgForClass(train_label,train,cmin,cmax,gmin,gmax,v,cstep,gstep,accstep)
%arameters of SVMcg 
if nargin < 10
    accstep = 1.5;
end
if nargin < 9
    cstep = 1;
    gstep = 1;
end
if nargin < 8
    v = 3;
    cstep = 1;
    gstep = 1;
end
if nargin <7
    v = 3;
    cstep = 1;
    gstep = 1;
    rstep = 5;
end
if nargin < 6
    v = 3;
    cstep = 1;
    gstep = 1;
    gmax = 5;
    gmin = -5;
end
if nargin < 5
    v = 3;
    cstep = 1;
    gstep = 1;
    gmax = 5;
    gmin = -5;
    cmax = 5;
end
if nargin < 4
    v = 3;
    cstep = 1;
    gstep = 1;
    rstep = 1;
    gmax = 5;
    gmin = -5;
    cmax = 5;
    cmin = -5;
end
%% X:c Y:g cg:acc
[X,Y,] = meshgrid(cmin:cstep:cmax,gmin:gstep:gmax);
[m,n] = size(X);
cg = zeros(m,n);
%% record acc with different c & g,and find the bestacc with the smallest c
bestc = 0;
bestg = 0;
bestacc = 0;
basenum = 2;
for i = 1:m
    for j = 1:n
        cmd = ['-v ',num2str(v),' -c ',num2str( basenum^X(i,j) ),' -g ',num2str( basenum^Y(i,j) )];
        cg(i,j) = svmtrain(train_label, train, cmd);
        
        if cg(i,j) > bestacc
            bestacc = cg(i,j);
            bestc = basenum^X(i,j);
            bestg = basenum^Y(i,j);
        end
        if ( cg(i,j) == bestacc && bestc > basenum^X(i,j) )
            bestacc = cg(i,j);
            bestc = basenum^X(i,j);
            bestg = basenum^Y(i,j);
        end
        
    end
end
%% to draw the acc with different c & g
 figure;
 [C,h] = contour(X,Y,cg,80:accstep:100);
 clabel(C,h,'Color','r');
 xlabel('log2c','FontSize',12);
 ylabel('log2g','FontSize',12);
 title('����ѡ����ͼ(�ȸ���ͼ)','FontSize',12);
 grid on;
 
 figure;
  meshc(X,Y,cg);
  mesh(X,Y,cg);
  surf(X,Y,cg);
 axis([cmin,cmax,gmin,gmax,30,100]);
 xlabel('log2c','FontSize',12);
 ylabel('log2g','FontSize',12);
 zlabel('Accuracy(%)','FontSize',12);
 title('����ѡ����ͼ(3D��ͼ)','FontSize',12);

