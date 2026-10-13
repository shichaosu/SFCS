% Figure-generation helper for the SFCS manuscript.
% Update the path variables below before running this script.

clear;
clc;
addpath 'external/matlab/Function_lin';



data_dir = 'data/data_use/table/result_CN/';

fsvtk_dir = 'outputs/visualization_surf/';
result_dir = 'data/visulization/figure_result/20/';

label_path = 'data/freesurfer_results/fsaverage6/label';
% label_path = 'data/freesurfer_results/002_S_0413_20170622_I863056/label';
    annot_lh = fullfile(label_path, 'lh.aparc.annot');
    annot_rh = fullfile(label_path, 'rh.aparc.annot');
    [v1, L1, ct1] = read_annotation(annot_lh);
    [v2, L2, ct2] = read_annotation(annot_rh);
    col51 = ct1.table(:,5);
    [~, loc1] = ismember(L1, col51);
    L1_r = loc1;


    col52 = ct2.table(:,5);
    [~, loc2] = ismember(L2, col52);
    L2_r = loc2+36;



    label = [L1_r;L2_r];
    label_orign = label;
    
    label_number = unique(label);
    surf_name = fullfile(fsvtk_dir, 'whole_fsaverage6.vtk');
    
    surf = vtkSurfRead(surf_name);    
    surf.Face = surf.Face-1;   
    surf.Pdata{1,1}.name = 'label_name';

%     SFCmat_dir = fullfile(SFC_dir, [folder_Name, '.mat']);
    data1mat_dir = fullfile(data_dir, 'ROI_score.mat');

    
    out_dir = result_dir;
    if ~exist(out_dir, 'dir')
        mkdir(out_dir);
    end


%% SFC  
    do_normalization = true;  
    % ===========================================

    inputmat = load(data1mat_dir);
    degree3 = inputmat.Redundancy;
    
    degree3 = degree3';
    min_origin = min(degree3);
    max_origin = max(degree3);
    if do_normalization
        if max_origin > min_origin
            degree_data = (degree3 - min_origin) / (max_origin - min_origin);
        else
            degree_data = zeros(size(degree3));
        end
        insert_val = 0; 
        disp('Processing message.');
        
    else
        degree_data = degree3;
        insert_val = 0;  %min_origin - 0.1*(max_origin - min_origin);
        fprintf('Processing message.', insert_val);
    end
    rows_to_insert = [1, 5, 37, 41];
    rows_to_insert = sort(rows_to_insert); 
    
    for i = 1:length(rows_to_insert)
        idx = rows_to_insert(i);
        degree_data = [degree_data(1:idx-1); insert_val; degree_data(idx:end)];
    end
    replaced_values1 = zeros(size(label));
    
    for j = 1:max(label)
        if j <= length(degree_data)
             replaced_values1(label == j) = degree_data(j);
        end
    end

    
    surf.Pdata{1, 1}.val = replaced_values1;
    

    
%% Left AND Right
    left_idx = 1:40962;
    right_idx = 40963:81924;
    surf_L.Vtx = surf.Vtx(:, left_idx);
    surf_R.Vtx = surf.Vtx(:, right_idx);
    faces = surf.Face;
    faces_left_mask = all(faces <= 40962, 1);
    faces_left = faces(:, faces_left_mask);
    faces_right_mask = all(faces >= 40963, 1);
    faces_right = faces(:, faces_right_mask);
    faces_right_local = faces_right - 40962;
    surf_L.Face = faces_left;
    surf_R.Face = faces_right_local;
    if isfield(surf, 'Pdata')
        surf_L.Pdata = surf.Pdata;
        surf_R.Pdata = surf.Pdata;
        surf_L.Pdata{1,1}.val = surf.Pdata{1,1}.val(left_idx);
        surf_R.Pdata{1,1}.val = surf.Pdata{1,1}.val(right_idx);        
        
    end
    
    
    vtkSurfWrite(fullfile(out_dir, 'ROI_score_Redundancy_L.vtk'), surf_L);
    vtkSurfWrite(fullfile(out_dir, 'ROI_score_Redundancy_R.vtk'), surf_R);
%     vtkSurfWrite(fullfile(out_dir, 'msn_Top5_percent.vtk'), surf);   
% 	  vtkSurfWrite(fullfile(out_dir, 'sfc_grad.vtk'), surf);








    


    

