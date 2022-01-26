try
    if ~(parallel.gpu.GPUDevice.isAvailable)
        fprintf(['\n\t**GPU not available. Stopping.**\n']);
        return;
    else
        dev = gpuDevice;
        fprintf(...
        'GPU detected (%s, %d multiprocessors, Compute Capability %s)\n',...
        dev.Name, dev.MultiprocessorCount, dev.ComputeCapability);
    end

    fprintf(...
        'Computing cross-correlation of "%s" and "%s" using the %s algorithm, writing results to "%s"\n',...
        in1_path, in2_path, alg, out_path);

    % in1_path, in2_path and data_type set externally
    [in1_matrix_size, in1_num_matrices, in1_data] = parseInput(in1_path, data_type);
    [in2_matrix_size, in2_num_matrices, in2_data] = parseInput(in2_path, data_type);

    switch alg
    case 'one_to_one'
        assert(in1_num_matrices == 1)
        assert(in2_num_matrices == 1)

        g_in1 = gpuArray(in1_data);
        g_in2 = gpuArray(in2_data);

        g_xcor = xcorr2(g_in1, g_in2);


        result_matrix = gather(g_xcor);
        header = [size(result_matrix), 1];
    case 'one_to_many'
        assert(in1_num_matrices == 1)

        g_in1 = gpuArray(in1_data);
        result_matrix_size = in1_matrix_size + in2_matrix_size - 1;
        result_matrix = zeros(result_matrix_size(1) * in2_num_matrices, result_matrix_size(2));
        for t = 1:in2_num_matrices
            % TODO: Upload all data to GPU at once
            g_in2 = gpuArray(in2_data(1 + (t-1)*in2_matrix_size(1):t*in2_matrix_size(1),:));
            g_xcor = xcorr2(g_in1, g_in2);
            result_matrix(1 + (t-1)*result_matrix_size(1):t*result_matrix_size(1),:) = gather(g_xcor);
        end

        header = [result_matrix_size, in2_num_matrices];
    case 'n_to_m'
        g_in1 = gpuArray(in1_data);
        g_in2 = gpuArray(in2_data);

        result_matrix_size = in1_matrix_size + in2_matrix_size - 1;
        result_matrix = zeros(result_matrix_size(1) * in1_num_matrices * in2_num_matrices, result_matrix_size(2));
        for r = 1:in1_num_matrices
            g_in1_mat = g_in1(1 + (r - 1)*in1_matrix_size(1):r*in1_matrix_size(1),:);
            for t = 1:in2_num_matrices
                res_matrix_start_row = 1 + ((t-1) + (r-1)*in2_num_matrices)*result_matrix_size(1);

                g_in2_mat = g_in2(1 + (t - 1)*in2_matrix_size(1):t*in2_matrix_size(1),:);
                g_xcor = xcorr2(g_in1_mat, g_in2_mat);

                result_matrix(res_matrix_start_row:res_matrix_start_row + (result_matrix_size(1) - 1),:) = gather(g_xcor);
            end
        end

        header = [result_matrix_size, in1_num_matrices * in2_num_matrices];
    case 'n_to_mn'
        % Results should be ordered so that first we have the cross-correlation of the
        % n matrices from input1 with the corresponding matrix from the first n matrices in
        % input2, then following should be the results of cross-correlation of the n matrices
        % from input1 with the matrices [n,2n) from input2 etc. up to cross-correlation of
        % the n matrices from input1 with matrices [(m-1)*n,m*n) from input2
        assert(mod(in2_num_matrices, in1_num_matrices) == 0)
        g_in1 = gpuArray(in1_data);
        g_in2 = gpuArray(in2_data);
        result_matrix_size = in1_matrix_size + in2_matrix_size - 1;
        result_matrix = zeros(result_matrix_size(1) * in2_num_matrices, result_matrix_size(2));
        for r = 1:in1_num_matrices
            g_in1_mat = g_in1(1 + (r - 1)*in1_matrix_size(1):r*in1_matrix_size(1),:);
            for t = 1:in2_num_matrices / in1_num_matrices
                t_matrix_index = (r - 1) + (t - 1)*in1_num_matrices;
                t_matrix_start_row = 1 + t_matrix_index*in2_matrix_size(1);
                res_matrix_start_row = 1 + t_matrix_index*result_matrix_size(1);

                g_in2_mat = g_in2(t_matrix_start_row:t_matrix_start_row + (in2_matrix_size(1) - 1),:);
                g_xcor = xcorr2(g_in1_mat, g_in2_mat);
                result_matrix(res_matrix_start_row:res_matrix_start_row + (result_matrix_size(1) - 1),:) = gather(g_xcor);
            end
        end

        header = [result_matrix_size, in2_num_matrices];
    otherwise
        error('Unknown algorithm type %s', alg)
    end

    % out_path set externally
    fid = fopen(out_path, 'w');
    fprintf(fid, '# %u,%u,%u', header(1), header(2), header(3));
    fclose(fid);
    writematrix(result_matrix, out_path, "Delimiter", ",", "FileType", "text", "WriteMode", "append");

    disp('Done')

catch err
    disp(getReport(err, 'extended'))
    exit(2)
end

function [matrix_size, num_matrices, data] = parseInput(path, data_type)
    fid = fopen(path, 'r');
    header = textscan(fid, '# %u%u%u', 'Delimiter', ',', 'ReturnOnError', 1);
    fclose(fid);
    assert(size(header, 2) >= 2);

    matrix_size = [cell2mat(header(1)), cell2mat(header(2))];
    if (size(header, 2) == 3)
        num_matrices = cell2mat(header(3));
    else
        num_matrices = 1;
    end
    data = readmatrix(path, 'NumHeaderLines', 1, 'OutputType', data_type);
end