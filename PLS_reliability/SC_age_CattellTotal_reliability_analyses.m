%% reproducability analyses using Randy's split_half_PLS_TestTrain.m

filename_modifier = "age_plus_cattell";

file_prefixes = [   "SC_age_CattellTotal_1000_its_594_subs_0_to_150_age_range";
                    "SC_le_age_CattellTotal_1000_its_244_subs_0_to_50_age_range";
                    "SC_le_age_CattellTotal_1000_its_350_subs_50_to_150_age_range";
                    "SC_ne_age_CattellTotal_1000_its_244_subs_0_to_50_age_range";
                    "SC_ne_age_CattellTotal_1000_its_350_subs_50_to_150_age_range"];
file_prefixes_size = size(file_prefixes);
file_prefixes_n = file_prefixes_size(1);

X_file_string = "_X";
Y_file_string = "_Y_age_CattellTotal";
lvs = floor(2);

for i = 1:file_prefixes_n
    disp(file_prefixes(i))
    X = readmatrix(strcat(append('inputs/',file_prefixes(i),X_file_string,".csv")));
    Y = readmatrix(strcat(append('inputs/',file_prefixes(i),Y_file_string,".csv")));

    behav = [];
    behav{1} = Y(:,1); % Age
    behav{2} = Y(:,2); % CattellTotal
    behav_mat = cell2mat(behav);

    [pls_repro_TT,splitflag_TT] = split_half_PLS_TestTrain(X,behav_mat,1,1000,1);
    for lv = 1:lvs
        Z_TT = pls_repro_TT.z(lv);
        Z_TT_null = pls_repro_TT.z_null(lv);
        writematrix(Z_TT,strcat(append('outputs/',file_prefixes(i),"_lv_",int2str(lv),"_TT_Z")));
        writematrix(Z_TT_null,strcat(append('outputs/',file_prefixes(i),"_lv_",int2str(lv),"_TT_Z_null")));
    end

    [pls_repro_SH,splitflag_SH]=split_half_PLS(X,behav_mat,1,1000,2,95,1);
    for lv = 1:lvs
        Z_SH = pls_repro_SH.pls_rep_z_u(lv);
        Z_SH_null = pls_repro_SH.pls_null_z_u(lv);
        writematrix(Z_SH,strcat(append('outputs/',file_prefixes(i),"_lv_",int2str(lv),"_SH_Z")));
        writematrix(Z_SH_null,strcat(append('outputs/',file_prefixes(i),"_lv_",int2str(lv),"_SH_Z_null")));
    end
end
%%