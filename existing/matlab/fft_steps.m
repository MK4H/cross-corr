if ~(parallel.gpu.GPUDevice.isAvailable)
    fprintf(['\n\t**GPU not available. Stopping.**\n']);
    return;
else
    dev = gpuDevice;
    fprintf(...
    'GPU detected (%s, %d multiprocessors, Compute Capability %s)\n',...
    dev.Name, dev.MultiprocessorCount, dev.ComputeCapability);
end
cd /home/karel/skola/diplomka/crosscorr/code/measurements

% TODO: Change type based on the one used in the C++ program
in1 = readmatrix('data/test/512_512_1_1.csv', 'NumHeaderLines', 1, 'OutputType', 'single');
in2 = readmatrix('data/test/512_512_1_2.csv', 'NumHeaderLines', 1, 'OutputType', 'single');
in1_padded = padarray(in1, size(in1), 'post');
in2_padded = padarray(in2, size(in2), 'post');


in1_fft = fft2(gpuArray(in1_padded));
in2_fft = fft2(gpuArray(in2_padded));

writematrix(in1_fft, 'data/test/ref_fft_matlab.csv')
writematrix(in2_fft, 'data/test/target_fft_matlab.csv')

% TODO: Why do we need conjugate of in2 ?
% Following the definition it should be conjugate of in1
%res_fft = conj(in1_fft).*in2_fft;
res_fft = in1_fft.*conj(in2_fft);

writematrix(res_fft, 'data/test/hadamard_matlab.csv')

res_perm = real(gather(ifft2(res_fft)));

writematrixwithheader(res_perm, 'data/test/out_fft_matlab_perm.csv')

% Reorder quadrants, swapping top left with bottom right and top right with bottom left
res = swapquadrants(res_perm);

writematrixwithheader(res, 'data/test/out_fft_matlab.csv')

% reread the matlab output so that we have the same precision loss as when
% reading the cpp output
%res_reread = readmatrix('data/test/out_fft_matlab.csv', 'NumHeaderLines', 1, 'OutputType', 'single');
%valid_res = readmatrix('data/test/result_512_512_1_1_1_2.csv', 'NumHeaderLines', 1, 'OutputType', 'single');

% cuFFT does unnormalized FFT, so IFFT(FFT(X)) returns each element of X
% multiplied by the number of elements of X
%fft_res = cpp_res./numel(cpp_res_perm);

%g_diff = gpuArray(res_reread) - gpuArray(valid_res);

%mean2(g_diff)
%std2(g_diff)
disp('Done')

function writematrixwithheader(matrix, out_path)
    fid = fopen(out_path, 'w');
    fprintf(fid, '# %u,%u,%u', size(matrix, 1), size(matrix, 2), 1);
    fclose(fid);
    writematrix(matrix, out_path, "Delimiter", ",", "FileType", "text", "WriteMode", "append");
end

% Swap quadrants to get proper cross correlation result from the
% fft/hadamard/ifft sequence of operations
function swapped = swapquadrants(matrix)
    swapped = cat( ...
        2, ...
        cat( ...
            1, ...
            matrix(size(matrix, 1)/2+2:size(matrix, 1), size(matrix,2)/2+2:size(matrix,2)), ... % bottom right
            matrix(1:size(matrix, 1)/2, size(matrix,2)/2+2:size(matrix,2)) ... % top right
        ), ...
        cat( ...
            1, ...
            matrix(size(matrix, 1)/2+2:size(matrix, 1), 1:size(matrix,2)/2), ... % bottom left
            matrix(1:size(matrix, 1)/2, 1:size(matrix,2)/2) ... % top left
        ) ...
    );
end