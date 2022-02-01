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

run('cross_corr.m');

disp('Done');
