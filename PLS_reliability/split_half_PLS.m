function [pls_repro,splitflag]=split_half_PLS(x,y,n_con,numsplits,lv,CI,dist_flag)
% [pls_repro,cca_repro,splitflag]=split_half_PLS(x,y,n_con,numsplits,lv,CI,dist_flag)
%
% computes the cosines between singular vectors U and V from numsplit
% random splits of X and Y data
% n_con= number of conditions
% nsplits: number of split half samples
% lv: largest number of LV to be evaluated - e.g. lv=3 means 1,2,3 are
% assesed
% CI: confidence interval percentile
% dist_flag: save distributions 1=true default=0
%
%OUTPUT
% pls_repro.pls_rep_mean_u  average of cosines for u distribution from split-half
% pls_repro.pls_rep_mean_v average of cosines for v distribution from split-half
% pls_repro.pls_rep_z_u  Z-value for v distribition (mean_u/std_u)
% pls_repro.pls_rep_z_v  Z-value for v distribition (mean_v/std_v)
% pls_repro.pls_rep_ul_u=pls_rep_ul_u upper bound of u distribution  
% pls_repro.pls_rep_ll_u=pls_rep_ll_u lower bound of u distribution 
% pls_repro.pls_rep_ul_v=pls_rep_ul_v upper bound of v distribution 
% pls_repro.pls_rep_ll_v=pls_rep_ll_v upper bound of v distribution  
% pls_repro.pls_null_mean_u=pls_null_mean_u average of null u distribution
% created by permutation 
% pls_repro.pls_null_mean_v=pls_null_mean_v average of null v distribution
% pls_repro.pls_null_z_u=pls_null_u_z Z-value for null u distribition 
% pls_repro.pls_null_z_v=pls_null_v_z Z-value for null v distribition 
% pls_repro.pls_null_ul_u=pls_null_ul_u upper bound of null u distribution
% pls_repro.pls_null_ll_u=pls_null_ll_u; lower bound of null u distribution
% pls_repro.pls_null_ul_v=pls_null_ul_v; upper bound of null v distribution
% pls_repro.pls_null_ll_v=pls_null_ll_v lower bound of null u distribution
% pls_repro.pls_dist_u  full distribution of u cosines if dist_flag=1
% pls_repro.pls_dist_v  full distribution of v cosines if dist_flag=1
% pls_repro.pls_dist_null_u full distribution of null_u cosines if dist_flag=1
% pls_repro.pls_dist_null_v full distribution of null_v cosines if dist_flag=1
%
% Dependencies: calls pls_only
%
% Written ARMcIntosh December 2020
% REVISED 23Dec-2022 for PLS only analyses
% Modified by LRokos & ARMcIntosh, November 2023 to handle conditions


if nargin==5
    dist_flag=0;
end

[n,p]=size(x);
n=n/n_con;
nsplit=ceil(n/2);
[~,q]=size(y);

d=min(p,q*n_con);

pls_u_repro=zeros(d,d,numsplits);
pls_v_repro=zeros(d,d,numsplits);

pls_u_null=zeros(d,d,numsplits);
pls_v_null=zeros(d,d,numsplits);

splitflag=[];
idx_subj=[1:n*n_con]; %get indices for subjects (within condition)
idx_subj=reshape(idx_subj,n,n_con); %reshape to 2 columns 

for i=1:numsplits
    %disp(i);
     idx=randperm(n);
     tmp_idx_subj=idx_subj(idx,:);

     idx_1=tmp_idx_subj(1:nsplit,:);
     idx_1=idx_1(:);

     idx_2=tmp_idx_subj(nsplit+1:n,:);
     idx_2=idx_2(:);

     x1=x(idx_1,:);
     y1=y(idx_1,:);
     x2=x(idx_2,:);
     y2=y(idx_2,:);
     %test for zero variance in x or y
     %test_std=[std(x1) std(x2) std(y1) std(y2)];

%    if all((test_std)~=0)
     [pls1]=pls_only(x1,y1,n_con);
     [pls2]=pls_only(x2,y2,n_con);

     pls_u_repro(:,:,i)=pls1.u'*pls2.u;
     %keyboard()
     pls_v_repro(:,:,i)=pls1.v'*pls2.v;
%    else
%        splitflag=[splitflag,i];
%    end
     
    
end

for i=1:numsplits %create a null distribution
    %disp(i);
     idx=randperm(n);
     permy=y(randperm(n*n_con),:); %scramble y
     tmp_idx_subj=idx_subj(idx,:);

     idx_1=tmp_idx_subj(1:nsplit,:);
     idx_1=idx_1(:);

     idx_2=tmp_idx_subj(nsplit+1:n,:);
     idx_2=idx_2(:);

     x1=x(idx_1,:);
     y1=permy(idx_1,:);
     x2=x(idx_2,:);
     y2=permy(idx_2,:);
     
     %test for zero variance in x or y
     %test_std=[std(x1) std(x2) std(y1) std(y2)];
     
     pls1=pls_only(x1,y1,n_con);
     pls2=pls_only(x2,y2,n_con);
    
%    if all((test_std)~=0)
     pls_u_null(:,:,i)=pls1.u'*pls2.u;
     pls_v_null(:,:,i)=pls1.v'*pls2.v;
     
%    end
    

end

for i=1:lv
    pls_rep_mean_u(i)=mean(abs(pls_u_repro(i,i,:)));
    pls_rep_std_u(i)=std(abs(pls_u_repro(i,i,:)));
    pls_rep_u_z(i)=pls_rep_mean_u(i)/pls_rep_std_u(i);
    pls_rep_ul_u(i)=prctile(abs(pls_u_repro(i,i,:)),CI);
    pls_rep_ll_u(i)=prctile(abs(pls_u_repro(i,i,:)),100-CI);
    pls_rep_mean_v(i)=mean(abs(pls_v_repro(i,i,:)));
    pls_rep_std_v(i)=std(abs(pls_v_repro(i,i,:)));
    pls_rep_v_z(i)=pls_rep_mean_v(i)/pls_rep_std_v(i);
    pls_rep_ul_v(i)=prctile(abs(pls_v_repro(i,i,:)),CI);
    pls_rep_ll_v(i)=prctile(abs(pls_v_repro(i,i,:)),100-CI);
end
 
for i=1:lv
    pls_null_mean_u(i)=mean(abs(pls_u_null(i,i,:)));
    pls_null_std_u(i)=std(abs(pls_u_null(i,i,:)));
    pls_null_u_z(i)=pls_null_mean_u(i)/pls_null_std_u(i);
    pls_null_ul_u(i)=prctile(abs(pls_u_null(i,i,:)),CI);
    pls_null_ll_u(i)=prctile(abs(pls_u_null(i,i,:)),100-CI);
    pls_null_mean_v(i)=mean(abs(pls_v_null(i,i,:)));
    pls_null_std_v(i)=std(abs(pls_v_null(i,i,:)));
    pls_null_v_z(i)=pls_null_mean_v(i)/pls_null_std_v(i);
    pls_null_ul_v(i)=prctile(abs(pls_v_null(i,i,:)),CI);
    pls_null_ll_v(i)=prctile(abs(pls_v_null(i,i,:)),100-CI);
end
 
pls_repro.pls_rep_mean_u=pls_rep_mean_u;
pls_repro.pls_rep_mean_v=pls_rep_mean_v;
pls_repro.pls_rep_z_u=pls_rep_u_z;
pls_repro.pls_rep_z_v=pls_rep_v_z;
pls_repro.pls_rep_ul_u=pls_rep_ul_u; 
pls_repro.pls_rep_ll_u=pls_rep_ll_u;
pls_repro.pls_rep_ul_v=pls_rep_ul_v; 
pls_repro.pls_rep_ll_v=pls_rep_ll_v; 
pls_repro.pls_null_mean_u=pls_null_mean_u;
pls_repro.pls_null_mean_v=pls_null_mean_v;
pls_repro.pls_null_z_u=pls_null_u_z;
pls_repro.pls_null_z_v=pls_null_v_z;
pls_repro.pls_null_ul_u=pls_null_ul_u; 
pls_repro.pls_null_ll_u=pls_null_ll_u;
pls_repro.pls_null_ul_v=pls_null_ul_v; 
pls_repro.pls_null_ll_v=pls_null_ll_v;
if dist_flag==1
    pls_repro.pls_dist_u=pls_u_repro;
    pls_repro.pls_dist_v=pls_v_repro;
    pls_repro.pls_dist_null_u=pls_u_null;
    pls_repro.pls_dist_null_v=pls_v_null;
end

