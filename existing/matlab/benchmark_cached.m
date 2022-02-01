if ~(parallel.gpu.GPUDevice.isAvailable)
    fprintf(['\n\t**GPU not available. Stopping.**\n']);
    return;
end

for iter = 1:iterations
    fprintf('Iteration %d/%d\r', iter, iterations);
    out_path = fullfile(out_dir, sprintf("%d.csv",int2str(iter)));
    run('cross_corr.m');
end
