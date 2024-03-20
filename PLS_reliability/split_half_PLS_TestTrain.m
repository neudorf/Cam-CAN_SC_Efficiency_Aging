function [pls_repro,splitflag]=split_half_PLS_TestTrain(x,y,n_con,numsplits,null_flag)
%[pls_repro,cca_repro,splitflag]=split_half_PLS_TestTrain(x,y,n_con,numsplits,null_flag)
%n_con= number of conditions
%numsplits: number of split half samples
%null_flag: save distributions 1=true default=0
%
% Uses a test-train loop where data are split and singular vectors from
% train set are applied to test and generate the corresponding singular
% value Zscore distribution for the test singular values is calculated,
% which is the figure of merit for reproducibility A null distribution can
% be generated using a permuation approach to get an idea of the expected
% value for the test distribution.
%
% Version updated 23Dec - 2022 for PLS only
% Modified by LRokos & ARMcIntosh, November 2023 to handle conditions


if nargin==3
    null_flag=0;
end

[n,p]=size(x);
n=n/n_con;
nsplit=floor(n/2);
[~,q]=size(y);

d=min(p,q*n_con);


splitflag=[];
idx_subj=[1:n*n_con];
idx_subj=reshape(idx_subj,n,n_con);

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
%     test_std=[std(x1) std(x2) std(y1) std(y2)];

    
%     if all((test_std)~=0)
    n2=size(y2,1);
    n2=n2/n_con;

    if n_con>1
        idx_subj_x2=[1:n2*n_con];
        idx_subj_x2=reshape(idx_subj_x2,n2,n_con);
        rxy = [];
    
        for m=1:n_con
            tmp_rxy = corr(x2(idx_subj_x2(:,m),:),y2(idx_subj_x2(:,m),:));
            rxy=[rxy,tmp_rxy];
        end
        
    else
        rxy=corr(x2,y2);
    end
         locate_nans = find(isnan(rxy));
         rxy(locate_nans)= 0;
         [pls1]=pls_only(x1,y1,n_con);
         pls_s_train(:,:,i)=pls1.s;
         pls_s_test(:,:,i)=pls1.u'*rxy*pls1.v;

%    else
%        splitflag=[splitflag,i];
%         pls_s_train(:,:,i)=NaN(q,q);
%         pls_s_test(:,:,i)=NaN(q,q);
%    end
     
end


if null_flag==1
for i=1:numsplits %create a null distribution
    %disp(i);
     idx=randperm(n);
     tmp_idx_subj=idx_subj(idx,:);
     permy=y(randperm(n*n_con),:); %scramble y
     idx_1=tmp_idx_subj(1:nsplit,:);
     idx_1=idx_1(:);

     idx_2=tmp_idx_subj(nsplit+1:n,:);
     idx_2=idx_2(:);

     x1=x(idx_1,:);
     y1=permy(idx_1,:);
     x2=x(idx_2,:);
     y2=permy(idx_2,:);
     
     %test for zero variance in x or y
%     test_std=[std(x1) std(x2) std(y1) std(y2)];

    n2=size(y2,1);
    n2=n2/n_con;
        if n_con>1
            idx_subj_x2=[1:n2*n_con];
            idx_subj_x2=reshape(idx_subj_x2,n2,n_con);
            rxy = [];
        
        for m=1:n_con
            tmp_rxy = corr(x2(idx_subj_x2(:,m),:),y2(idx_subj_x2(:,m),:));
            rxy=[rxy,tmp_rxy];
        end
        
        else
            rxy=corr(x2,y2);
        end
%    if all((test_std)~=0)

        locate_nans = find(isnan(rxy));
         rxy(locate_nans)= 0;
         [pls1_null]=pls_only(x1,y1,n_con);
         pls_s_train_null(:,:,i)=pls1_null.s;
         pls_s_test_null(:,:,i)=pls1_null.u'*rxy*pls1_null.v;
         
%    end

end

end

pls_repro.pls_s_train=pls_s_train;
pls_repro.pls_s_test=pls_s_test;


for i=1:d
    pls_repro.z(i)=mean(pls_repro.pls_s_test(i,i,:),'omitnan')/std(pls_repro.pls_s_test(i,i,:),'omitnan');

end

if null_flag==1
pls_repro.pls_s_train_null=pls_s_train_null;
pls_repro.pls_s_test_null=pls_s_test_null;
    for i=1:d
        pls_repro.z_null(i)=mean(pls_repro.pls_s_test_null(i,i,:),'omitnan')/std(pls_repro.pls_s_test_null(i,i,:),'omitnan');
    
    end
end

