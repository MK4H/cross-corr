# Cross-correlation measurements

## TODOs
- read the CUDA event times only during finalize, so that we prevent hidden synchronization
    during the run of the algorihtm

## Profiling

When profiling on localhost, you will need to either configure X server so that the NVidia device is run in non-interactive mode,
so that the kernels are not killed by watchdog for running too long, or you will need to run without XServer entirely and
use the NSight Compute CLI.


### Ubuntu 20.04 stop and start XServer
Logout from the current XServer session.

Switch to virtual terminal:
```
ctrl+alt+F2
```

Stop XServer:
```
sudo systemctl stop gdm
```

Switch again to virtual terminal, as it switches to the stopped GUI terminal for some reason:
```
ctrl+alt+F2
```

Then run your profiling using the NSight Compute CLI.

Start the XServer again:
```
sudo systemctl start gdm
```
