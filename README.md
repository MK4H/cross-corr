# Cross-correlation measurements

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

DON'T need to stop the XServer if no user is logged in


Then run your profiling using the NSight Compute CLI.


## Possible optimizations

Currently threads in the 16x16 block in each row read the value used by the thread ahead of them on the x axis.
This could be used by aligning all 32 threads of a warp to compute a single row, preload two warps worth of data into registers
and then rotate the data. After 32 steps, we just read one warp worth of data and continue rotating.

Separately or combined with the warp data shuffles, we may preload the data used by thread block into shared memory,
maybe in the y axis so that warps preload data from shared memory instead of global memory.
We may use ring buffers, as that is basically perfect for the data access of the current algorithm.

### Small inputs

As FFT will always be faster for bigger inputs, we may try work with data that fits into shared memory,
such as 32x32 matrices and try to optimize for that. Concentrate on bank conflicts, data reuse etc.

Any data reuse unfortunately limits parallism.

We may try to load the matrix in one_to_many algorithms and iterate over multiple of the many, reducing global loads by one matrix
each of the many we iterate over with the one in shared memory.

Do similar with n_to_mn, with parallelisation over the n matrices with each being loaded into shared memory and iterated over.

For n_to_m,


# TODO

Measure prepare step, cufftPlanMany and dependency on input sizes

Add n_to_mn to list of possible input configurations in text

Introduction as nonnumbered chapter. Zacit od lesa, fakt datova analyza, proc je potreba vykon. Udelat to delsi.


Pridat proc na tu praci navazuju, v cem pokracuju, co tam pridavam sveho. Ze experimentuju s implementacema cross correlace.

Pak motivace. Proc to delam a proc to ma ctenar cist, co jsem udelal aby vedel co cist.

Cross correlation je prej dobra

Bez citace by mel text davat smysl. Nemusi to bejt treba soucast textu.

MIsto "naive" napsat to kurzivou nebo necim. Jako ze zavadim termin.

FIgure, Section, Chapters s velkym.

Na zacatku sekce vzdycky prozradit pointu. Treba u furierky ze je to O(nlogn) ale ze o tom pisem protoze je to rychlejsi nez O(n2) co ma nainvni.

Ze se nezamerujeme jenom na integer, black and white. ALe ze to byl grasclae ale ze mi se proste zamerujeme na matice, ktery se treba pouzivaji v image processing s integerama, s floating pointama, na greyscale atd.

Megnout ty prikaldy a pak z nich odvozeny typy cross correlace. Proste udela tu taxonomii usacesu a pak k tomu pripsat ty citace kde se to puzilo.

Bude potreba pridat related work, je toho jeste malo.
Related work k backgroundu uz mam, ale pridat dalsi implementace cross corr. Ty co s tim zacinali, ty co to rozsirili a ty co to vyuzili pro nejaky vyuziti.
V GPU implementace cross-correlace a HPC implementace cross-correlace.
Ze driv se to delalo na CPU, delali to tyhle. Pak ze se to zacalo delat na GPU a delali to tihle a pak ktery to rozsirili atd.

Co se s vystupem dela dal. I u tech related works. I pak zminit ze mi to nebudeme delat.


Historie GPU tam nedavat, je to mimo tema.

subsekce by mela mit nejlepe min nez 2 stranky.

Nepsat souvisly odstavce textu. Rozbit to obrazkama, bullet pointama atd.

Celou tu CUDA cast prepsat

Vzit jako baseline CPU kod a vysvetlit tim GPU. Zacit od obecnejch principu. Pak se teprve zamerit na technicky detaily. Nemusi se pokrejt uplne vsechno. Zamerit
se na veci ktery urcite pouzijeme. Nakonec napr. ukazky indexace, jak se to pousti, jak se to pouziva.
Melo by to slouzit pro utrideni myslenek pro ty kdo uz to slyseli, pro ostatni aby se do toho alespon trochu dostali.

Udelat rozdeleni ty sekce GPU, co tam vlastne chci napsat, a postupne do toho vyzobat z puvodniho textu.

Nakonec CUDA sekce nejaky obecny pravidla, co by se melo a co by se nemelo.

V analyze pak navazuju na to co jsem popsal.
V analyze vysvetlit jaky techniky jsem pouzil. U kazdyho rozhodnuti popsat proc jsme vybrali to co jsme vybrali.
