try
    if ~(parallel.gpu.GPUDevice.isAvailable)
        fprintf(['\n\t**GPU not available. Stopping.**\n']);
        return;
    end

    timing_labels = ["Load", "Load_iterations", "Computation", "Computation_iterations"];

    measured_timings = zeros(1,2);
    measured_iters = zeros(1,2);


    % in1_path, in2_path and data_type set externally
    tic
    [in1_matrix_size, in1_num_matrices, in1_data] = parseInput(in1_path, data_type);
    [in2_matrix_size, in2_num_matrices, in2_data] = parseInput(in2_path, data_type);
    measured_timings(1) = toc;
    measured_iters(1) = 1;

    for iter = 1:iterations
        computation_iterations = 1;
        while computation_iterations > 0
            computation_time = tic;
            for ad_iter = 1:computation_iterations

                g_in1 = gpuArray(in1_data);
                g_in2 = gpuArray(in2_data);

                switch alg
                case 'one_to_one'
                    assert(in1_num_matrices == 1)
                    assert(in2_num_matrices == 1)

                    g_xcor = xcorr2(g_in1, g_in2);
                    result_matrix = gather(g_xcor);

                    header = [size(result_matrix), 1];
                case 'one_to_many'
                    assert(in1_num_matrices == 1)

                    result_matrix_size = in1_matrix_size + in2_matrix_size - 1;
                    g_result = gpuArray(zeros(result_matrix_size(1) * in2_num_matrices, result_matrix_size(2), data_type));

                    for t = 1:in2_num_matrices
                        g_in2_mat = g_in2(1 + (t-1)*in2_matrix_size(1):t*in2_matrix_size(1),:);
                        g_result(1 + (t-1)*result_matrix_size(1):t*result_matrix_size(1),:) = xcorr2(g_in1, g_in2_mat);
                    end

                    result_matrix = gather(g_result);

                    header = [result_matrix_size, in2_num_matrices];
                case 'n_to_m'
                    result_matrix_size = in1_matrix_size + in2_matrix_size - 1;
                    g_result = gpuArray(zeros(result_matrix_size(1) * in1_num_matrices * in2_num_matrices, result_matrix_size(2), data_type));

                    for r = 1:in1_num_matrices
                        g_in1_mat = g_in1(1 + (r - 1)*in1_matrix_size(1):r*in1_matrix_size(1),:);
                        for t = 1:in2_num_matrices
                            res_matrix_start_row = 1 + ((t-1) + (r-1)*in2_num_matrices)*result_matrix_size(1);

                            g_in2_mat = g_in2(1 + (t - 1)*in2_matrix_size(1):t*in2_matrix_size(1),:);
                            g_result(res_matrix_start_row:res_matrix_start_row + (result_matrix_size(1) - 1),:) = xcorr2(g_in1_mat, g_in2_mat);


                        end
                    end

                    result_matrix = gather(g_result);

                    header = [result_matrix_size, in1_num_matrices * in2_num_matrices];
                case 'n_to_mn'
                    % Results should be ordered so that first we have the cross-correlation of the
                    % n matrices from input1 with the corresponding matrix from the first n matrices in
                    % input2, then following should be the results of cross-correlation of the n matrices
                    % from input1 with the matrices [n,2n) from input2 etc. up to cross-correlation of
                    % the n matrices from input1 with matrices [(m-1)*n,m*n) from input2
                    assert(mod(in2_num_matrices, in1_num_matrices) == 0)

                    result_matrix_size = in1_matrix_size + in2_matrix_size - 1;
                    g_result = gpuArray(zeros(result_matrix_size(1) * in2_num_matrices, result_matrix_size(2), data_type));

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

                    result_matrix = gather(g_result);

                    header = [result_matrix_size, in2_num_matrices];
                otherwise
                    error('Unknown algorithm type %s', alg)
                end
            end

            measured_time = toc(computation_time);
            measured_timings(2) = measured_time / computation_iterations;
            measured_iters(2) = computation_iterations;

            if (measured_time >= adaptive_limit)
                computation_iterations = 0;
            else
                ratio = adaptive_limit / measured_time;
                computation_iterations = ceil(computation_iterations * max(min(ratio * 1.5, 100), 1.5));
            end
        end

        % out_path set externally
        if (strlength(out_path) ~= 0)
            [out_dir, out_name, out_ext] = fileparts(out_path);
            if (iterations > 1)
                iter_out_path = fullfile(out_dir, sprintf("%s_%d%s",out_name,iter,out_ext));
            else
                iter_out_path = out_path;
            end
            fid = fopen(iter_out_path, 'w');
            fprintf(fid, '# %u,%u,%u', header(1), header(2), header(3));
            fclose(fid);
            writematrix(result_matrix, iter_out_path, "Delimiter", ",", "FileType", "text", "WriteMode", "append");
        end

        if (exist('timings_path', 'var') == 1)
            % From seconds to nanoseconds
            measured_timings = measured_timings.*1e9;
            data = [measured_timings; measured_iters];
            % Interleave the timings and iters
            % Then convert it to row vector
            data = data(:)';
            table = array2table(data);
            table.Properties.VariableNames(1:length(timing_labels)) = timing_labels;
            writetable(table, timings_path, 'WriteMode', 'append');
        end

        measured_timings = zeros(size(measured_timings));
        measured_iters = zeros(size(measured_iters));
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