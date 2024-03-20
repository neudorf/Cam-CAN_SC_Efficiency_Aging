function [pls]=pls_only(x,y,n_con,nperm,nboot,CI)
% [pls]=pls_only(x,y,n_con,nperm,nboot,CI)
%
%
% Updated 21Dec to do PLS only
% Updated 23Nov to handle conditions
%
%Input 
% x and y are the data assume x is the higher dimension matrix
% option arguments
% n_con=number of conditions
% nperm = number of permutation resamplings
% nboot = number of bootstrap resamplings
% confidence interval - default 95
%
%Output
% pls.u, pls.v, pls.s are from the SVD of Rxy
% pls.perm contains matrix of all singular values for each permutation
% pls.boot.u pls.boot.us bootstrapped U and U*S
% pls.boot.v pls.boot.vs boostrapped V and V*S
% pls.boot.ul_u, pls.boot.ll_u upper and lower percentile of bootstrap dist for U
% pls.boot.ul_v, pls.boot.ll_v upper and lower percentile of bootstrap dist for V
%  Written by ARMcIntosh, December 2020
%  Modified by LRokos & ARMcIntosh, November 2023
% Dependencies: requires pls_cmd resampling code: rri_boot_order
% 

n=size(y,1);
n=n/n_con;

if n_con>1
    idx_subj=[1:n*n_con];
    idx_subj=reshape(idx_subj,n,n_con);
    rxy = [];
    
    for h=1:n_con
        tmp_rxy = corr(x(idx_subj(:,h),:),y(idx_subj(:,h),:));
        rxy=[rxy,tmp_rxy];
    end

else
    rxy=corr(x,y);
end

locate_nans = find(isnan(rxy));
rxy(locate_nans)= 0;
[pls.u,pls.s,pls.v]=svd(rxy,0);

[r,c]=size(x);
[r,c2]=size(y);


if nargin>2
  %perm_order=rri_perm_order(n,1,nperm); 
cancor.badcondition_perm=0;
  %disp("Permutation loop");
if exist("nperm")==1
for i=1:nperm
xperm=x(randperm(n*n_con),:); %keep the permutation simple for now
%     for j=1:c
%         xperm(:,j)=x(randperm(n),j); %permute columns independently
%     end
%     for j=1:c2
%                 yperm(:,j)=y(randperm(n),j);
%     end
%     pls.xperm{i}=xperm;
%     pls.yperm{i}=yperm;

        if n_con>1
            idx_subj=[1:n*n_con];
            idx_subj=reshape(idx_subj,n,n_con);
            rxPy = [];
        
            for h=1:n_con
                tmp_rxPy = corr(xperm(idx_subj(:,h),:),y(idx_subj(:,h),:));
                rxPy=[rxPy,tmp_rxPy];
            end
        
        else
            rxPy=corr(xperm,y);
        end

    pls.perms(:,i)=svd(rxPy);
    
end
end

if exist("nboot")==1
 %use rri_boot_order - part of plscmd release
    [boot_order,~]=rri_boot_order(n,n_con,nboot);

    disp("Bootstrap loop");
%     pls.boot.u=zeros([size(pls.u),nboot]);
     pls.boot.u=zeros(size(pls.u));
    pls.boot.v=zeros([size(pls.v),nboot]);
%     pls.boot.us=zeros([size(pls.u),nboot]);
    pls.boot.vs=zeros([size(pls.v),nboot]);
%    pls.boot.s=zeros([size(pls.s),nboot]);

    pls.boot.badboot=0;
for i=1:nboot
    %disp(i);
    xboot=x(boot_order(:,i),:);
    yboot=y(boot_order(:,i),:);
    %check for zero variance in bootstrap resample
    test_x=std(xboot);
    test_y=std(yboot);
    test_x=sum(test_x==0);
    test_y=sum(test_y==0);
    
    if test_x==0 & test_y==0
        if n_con>1
            idx_subj=[1:n*n_con];
            idx_subj=reshape(idx_subj,n,n_con);
            rxBy = [];
            
            for i=1:n_con
                tmp_rxBy = corr(xboot(idx_subj(:,i),:),yboot(idx_subj(:,i),:));
                rxBy=[rxBy,tmp_rxBy];
            end
        
        else       
            rxBy=corr(xboot,yboot);
        end
    [pls.boot.u,pls.boot.s(:,:,i),pls.boot.v(:,:,i)]=svd(rxBy,0);
   
    %check for sign flips on a bootstrap iteration 
    flip_idx=[];
    flip_idx=find(diag(pls.boot.u'*pls.u)<0);
    if isempty(flip_idx)==0
        %pls.boot.u(:,flip_idx,i)=pls.boot.u(:,flip_idx,i)*-1;
        pls.boot.v(:,flip_idx,i)=pls.boot.v(:,flip_idx,i)*-1;
    end
    
    %save scaled singular vectors for CI calculations
%    pls.boot.us(:,:,i)=pls.boot.u(:,:,i)*pls.boot.s(:,:,i);
    pls.boot.vs(:,:,i)=pls.boot.v(:,:,i)*pls.boot.s(:,:,i);
    
    else
        pls.boot.badboot=pls.boot.badboot+1;
    end

  
end
end
%compute percentile CI's for U and V
%[ru,cu]=size(pls.u);
[rv,cv]=size(pls.v);

if nargin>3
    CI=95;
end

% pls.boot.ul_u=zeros(ru,cu);
% pls.boot.ll_u=zeros(ru,cu);
pls.boot.ul_v=zeros(rv,cv);
pls.boot.ll_v=zeros(rv,cv);



%Loop to step through and get upper and lower bounds of bootstrap dist
% for i=1:cu
%     for j=1:ru
%         pls.boot.ul_u(j,i)=prctile(pls.boot.us(j,i,:),CI);
%         pls.boot.ll_u(j,i)=prctile(pls.boot.us(j,i,:),100-CI);
%     end
% end
if exist("nboot")==1
for i=1:cv
    for j=1:rv
        pls.boot.ul_v(j,i)=prctile(pls.boot.vs(j,i,:),CI);
        pls.boot.ll_v(j,i)=prctile(pls.boot.vs(j,i,:),100-CI);
    end
end

end
end
