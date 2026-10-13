% Structural connectivity and communication-metric processing.
% Update the path variables below before running this script.
clc;
clear all;
srcDir = 'data/SC/aparc/SC_mat/';
dstDir = 'data/SC/aparc/SCst_mat/';
if ~exist(dstDir, 'dir')
    mkdir(dstDir);
end
files = dir(fullfile(srcDir, '*.mat'));

for k = 1:length(files)
    filename = files(k).name;
    srcFile = fullfile(srcDir, filename);
    dstFile = fullfile(dstDir, filename);
    data = load(srcFile, 'sc');
    
    if isfield(data, 'sc')
        sc = data.sc;
        sc_min = min(sc(:));
        sc_max = max(sc(:));
        if sc_max > sc_min
            sc_norm = (sc - sc_min) / (sc_max - sc_min);
        else
            sc_norm = sc;
        end
        
        sc = sc_norm;
        save(dstFile, 'sc');
    else
        warning('Processing message.', filename);
    end
end

disp('Processing message.');