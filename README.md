This is our implementation of the ADMM (Mei) combined with the Fast Gradient Projection (FGP) method.
It is working correctly and can be executed via main.m

####
We tried to implement the Chambolle algorithm instead of FGP first, which did not work. However it can be reviewed and executed in mainChambolle.m

Then we implemented the FGP algorithm given in the Beck and Teboulle paper with help of the code you gave us,
modifying the latter to an isotrop version of the TV.
You can see the results by running main.m, this uses denoising_isotrop_tv.m.
In test.m we tested the denoising on the standard, tv-regularization + data-term objective, to compare this to our results.
For all details have a look to the report.