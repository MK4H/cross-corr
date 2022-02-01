try
    if ~(parallel.gpu.GPUDevice.isAvailable)
        fprintf(['\n\t**GPU not available. Stopping.**\n']);
        return;
    end

    timing_labels = ["Total", "Load", "Prepare", "Transfer", "Run", "Finalize"];
    timings = zeros(size(timing_labels));

    total_time = tic;
    % in1_path, in2_path and data_type set externally
    tic
    [in1_matrix_size, in1_num_matrices, in1_data] = parseInput(in1_path, data_type);
    [in2_matrix_size, in2_num_matrices, in2_data] = parseInput(in2_path, data_type);
    timings(2) = toc;

    tic
    g_in1 = gpuArray(in1_data);
    g_in2 = gpuArray(in2_data);
    timings(4) = toc;

    switch alg
    case 'one_to_one'
        assert(in1_num_matrices == 1)
        assert(in2_num_matrices == 1)

        tic
        g_xcor = xcorr2(g_in1, g_in2);
        timings(5) = toc;

        tic
        result_matrix = gather(g_xcor);
        timings(6) = toc;

        header = [size(result_matrix), 1];
    case 'one_to_many'
        assert(in1_num_matrices == 1)

        tic
        result_matrix_size = in1_matrix_size + in2_matrix_size - 1;
        g_result = gpuArray(zeros(result_matrix_size(1) * in2_num_matrices, result_matrix_size(2), data_type));
        timings(3) = toc;

        tic
        for t = 1:in2_num_matrices
            g_in2_mat = g_in2(1 + (t-1)*in2_matrix_size(1):t*in2_matrix_size(1),:);
            g_result(1 + (t-1)*result_matrix_size(1):t*result_matrix_size(1),:) = xcorr2(g_in1, g_in2_mat);
        end
        timings(5) = toc;

        tic
        result_matrix = gather(g_result);
        timings(6) = toc;

        header = [result_matrix_size, in2_num_matrices];
    case 'n_to_m'
        tic
        result_matrix_size = in1_matrix_size + in2_matrix_size - 1;
        g_result = gpuArray(zeros(result_matrix_size(1) * in1_num_matrices * in2_num_matrices, result_matrix_size(2), data_type));
        timings(3) = toc;

        tic
        for r = 1:in1_num_matrices
            g_in1_mat = g_in1(1 + (r - 1)*in1_matrix_size(1):r*in1_matrix_size(1),:);
            for t = 1:in2_num_matrices
                res_matrix_start_row = 1 + ((t-1) + (r-1)*in2_num_matrices)*result_matrix_size(1);

                g_in2_mat = g_in2(1 + (t - 1)*in2_matrix_size(1):t*in2_matrix_size(1),:);
                g_result(res_matrix_start_row:res_matrix_start_row + (result_matrix_size(1) - 1),:) = xcorr2(g_in1_mat, g_in2_mat);


            end
        end
        timings(5) = toc;

        tic
        result_matrix = gather(g_result);
        timings(6) = toc;

        header = [result_matrix_size, in1_num_matrices * in2_num_matrices];
    case 'n_to_mn'
        % Results should be ordered so that first we have the cross-correlation of the
        % n matrices from input1 with the corresponding matrix from the first n matrices in
        % input2, then following should be the results of cross-correlation of the n matrices
        % from input1 with the matrices [n,2n) from input2 etc. up to cross-correlation of
        % the n matrices from input1 with matrices [(m-1)*n,m*n) from input2
        assert(mod(in2_num_matrices, in1_num_matrices) == 0)

        tic
        result_matrix_size = in1_matrix_size + in2_matrix_size - 1;
        g_result = gpuArray(zeros(result_matrix_size(1) * in2_num_matrices, result_matrix_size(2), data_type));
        timings(3) = toc;

        tic
        for r = 1:in1_num_matrices
            g_in1_mat = g_in1(1 + (r - 1)*in1_matrix_size(1):r*in1_matrix_size(1),:);
            for t = 1:in2_num_matrices / in1_num_matrices
                t_matrix_index = (r - 1) + (t - 1)*in1_num_matrices;
                t_matrix_start_row = 1 + t_matrix_index*in2_matrix_size(1);
                res_matrix_start_row = 1 + t_matrix_index*result_matrix_size(1);

                g_in2_mat = g_in2(t_matrix_start_row:t_matrix_start_row + (in2_matrix_size(1) - 1),:);
                g_result(res_matrix_start_row:res_matrix_start_row + (result_matrix_size(1) - 1),:) = xcorr2(g_in1_mat, g_in2_mat);
            end
        end
        timings(5) = toc;

        tic
        result_matrix = gather(g_result);
        timings(6) = toc;

        header = [result_matrix_size, in2_num_matrices];
    otherwise
        error('Unknown algorithm type %s', alg)
    end

    timings(1) = toc(total_time);

    % out_path set externally
    fid = fopen(out_path, 'w');
    fprintf(fid, '# %u,%u,%u', header(1), header(2), header(3));
    fclose(fid);
    writematrix(result_matrix, out_path, "Delimiter", ",", "FileType", "text", "WriteMode", "append");

    if (exist('timings_path', 'var') == 1)
        % From seconds to nanoseconds
        timings = timings.*1e9;
        table = array2table(timings);
        table.Properties.VariableNames(1:length(timing_labels)) = timing_labels;
        writetable(table, timings_path, 'WriteMode', 'append');
    end

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