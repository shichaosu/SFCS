% Structural connectivity and communication-metric processing.
% Update the path variables below before running this script.
clear;
clc;

srcDir = 'data/SC/aparc/result_StdSC_addmin/';
dstDir = 'data/SC/aparc/result_StdSC_addmin_cleanAndStd0_1/';

if ~exist(dstDir, 'dir')
    mkdir(dstDir);
end

files = dir(fullfile(srcDir, '*.mat'));

for k = 1:length(files)
    filename = files(k).name;
    srcFile = fullfile(srcDir, filename);
    dstFile = fullfile(dstDir, filename);
    
    data = load(srcFile);
    if isfield(data, 'F')
        F = data.F;
        if any(isnan(F(:)))
            fprintf('Processing message.', filename);
            continue
        end
    else
        fprintf('Processing message.', filename);
        continue
    end
    varsToRemove = {'SI', 'comp_sizes', 'comps'};
    for v = 1:length(varsToRemove)
        if isfield(data, varsToRemove{v})
            data = rmfield(data, varsToRemove{v});
        end
    end

        varNames = fieldnames(data);
        for iVar = 1:length(varNames)
            varName = varNames{iVar};
            varData = data.(varName);

            if isnumeric(varData) && ismatrix(varData) && size(varData,1) == size(varData,2)
                n = size(varData,1);
                mask = ~eye(n);
                vals = varData(mask);

                minVal = min(vals);
                maxVal = max(vals);

                normalizedData = zeros(size(varData));

                if maxVal > minVal
                    normalizedVals = (vals - minVal) / (maxVal - minVal);
                    normalizedData(mask) = normalizedVals;
                else
                    normalizedData(mask) = 0;
                end
                data.(varName) = normalizedData;
            else
            end
        end
%             varNames = fieldnames(data);
%             for iVar = 1:length(varNames)
%                 varName = varNames{iVar};
%                 varData = data.(varName);
% 
%                 if isnumeric(varData) && ismatrix(varData) && size(varData,1) == size(varData,2)
%                     n = size(varData,1);
%                     mask = true(n);
%                     off_diag_vals = varData(mask);
% 
%                     minVal = min(off_diag_vals);
%                     maxVal = max(off_diag_vals);
%                     if maxVal > minVal
%                         norm_vals = 2 * (off_diag_vals - minVal) / (maxVal - minVal) - 1;
%                     else
%                         norm_vals = zeros(size(off_diag_vals));
%                     end
% 
%                     varData_norm = zeros(size(varData));
%                     varData_norm(mask) = norm_vals;
% 
%                     data.(varName) = varData_norm;
%                 elseif isnumeric(varData)
%                 end
%             end
    save(dstFile, '-struct', 'data');
end

disp('Processing message.');