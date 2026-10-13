% Structural connectivity and communication-metric processing.
% Update the path variables below before running this script.
clear;
clc;
source_dir = 'data/SC/aparc/StdSC_mat/';
target_dir = 'data/SC/aparc/result_StdSC_add/';

if ~exist(target_dir, 'dir')
    mkdir(target_dir);
end

files = dir(fullfile(source_dir, '*.mat'));
log_file = fullfile(target_dir, 'processing_log.txt');

for i = 1:length(files)
    fprintf('Processing message.', i, length(files), files(i).name);
    fid_log = fopen(log_file, 'a');
    fprintf(fid_log, 'Processing message.', files(i).name);
    
    try
        data = load(fullfile(source_dir, files(i).name));
        sc = data.sc;
        n = size(sc, 1);
        if n ~= 68 || size(sc,2) ~= 68
            warning('Processing message.');
        else
            sc_no_diag = sc - diag(diag(sc));
            min_val = min(sc_no_diag(sc_no_diag > 0));
            if isempty(min_val)
                min_val = 1;
            end
            for r = 1:n
                row_no_diag = sc(r, :);
                row_no_diag(r) = 0;
                if all(row_no_diag == 0)
                    candidates = setdiff(1:n, r);
                    c = candidates(randi(length(candidates)));
                    sc(r, c) = min_val;
                    sc(c, r) = min_val;
                end
            end
            for c = 1:n
                col_no_diag = sc(:, c);
                col_no_diag(c) = 0;
                if all(col_no_diag == 0)
                    candidates = setdiff(1:n, c);
                    r = candidates(randi(length(candidates)));
                    sc(r, c) = min_val;
                    sc(c, r) = min_val;
                end
            end
        end
        try
            F = communicability_wei(sc);
        catch ME
            fprintf(fid_log, 'Processing message.', ME.message);
            F = nan(size(sc));
        end
        try
            bin_sc = double(sc > 0);
            D_bin = distance_bin(bin_sc);
        catch ME
            fprintf(fid_log, 'Processing message.', ME.message);
            D_bin = nan(size(sc));
        end
        try
            [SPL, hops, Pmat] = distance_wei_floyd(sc, 'inv');
        catch ME
            fprintf(fid_log, 'Processing message.', ME.message);
            SPL = nan(size(sc));
            hops = nan(size(sc));
            Pmat = nan(size(sc));
        end
        try
            r = sum(sc, 2)';
            if size(r, 1) > 1
                r = r';
            end
            r(r == 0) = 1e-6;
            t = 1;
            dyn = fcn_flow_graph(sc, r, t);
        catch ME
            fprintf(fid_log, 'Processing message.', ME.message);
            dyn = nan(size(sc));
        end
        try
            [comps, comp_sizes] = get_components(bin_sc);
        catch ME
            fprintf(fid_log, 'Processing message.', ME.message);
            comps = nan(size(sc, 1), 1);
            comp_sizes = [];
        end
        try
            M0 = matching_ind_und(sc);
        catch ME
            fprintf(fid_log, 'Processing message.', ME.message);
            M0 = nan(size(sc));
        end
        try
            sc_mfpt = sc;
            out_degree = sum(sc_mfpt, 2);
            isolated_nodes = find(out_degree == 0);
            for k = 1:length(isolated_nodes)
                idx = isolated_nodes(k);
                sc_mfpt(idx, idx) = 1;
            end
            
            P = diag(sum(sc_mfpt, 2))\sc_mfpt;
            if any(isnan(P(:))) || any(isinf(P(:)))
                error('Processing message.');
            end
            MFPT = mean_first_passage_time(sc_mfpt);
        catch ME
            fprintf(fid_log, 'Processing message.', ME.message);
            MFPT = nan(size(sc));
        end
        try
            T = path_transitivity(sc, 'inv');
        catch ME
            fprintf(fid_log, 'Processing message.', ME.message);
            T = nan(size(sc));
        end
        try
            if any(Pmat(:) <= 0) || any(Pmat(:) > size(sc, 1))
                error('Processing message.');
            end
            SI = search_information(sc, 'inv', false);
        catch ME
            fprintf(fid_log, 'Processing message.', ME.message);
            SI = nan(size(sc));
        end
        save(fullfile(target_dir, files(i).name), ...
            'F', 'D_bin', 'SPL', 'hops', 'Pmat', 'dyn', ...
            'comps', 'comp_sizes', 'M0', 'MFPT', 'T', 'SI');
        
        fprintf(fid_log, 'Processing message.');
    catch ME
        fprintf(fid_log, 'Processing message.', ME.message);
    end
    
    fclose(fid_log);
end

fprintf('Processing message.', length(files));