# SourceExtraction
Code for extracting axons and dendrite from Calcium data

Installation:
Requires Python 2.7 and 3.5.
Download to folder.
Download and install OASIS (together with its dependencies) from here: https://github.com/j-friedrich/OASIS and add as a subfolder.
Additional dependencies: all the WinPython libraries, tifffile... and maybe more.

Main files:

Simple_Demo.py - Run this to see how basic code works (on small 3D data sample).

Demo.py - More advanced options and parameters for many datasets. Run to see how code works, in more detail.

BlockLocalNMF.py - main script for running a single NMF repetition. LocalNMF code contains description of all input parameters.

AuxilaryFunctions.py - General Auxilary Functions

BlockLocalNMFAuxilaryFunctions.py - Auxilary Functions for BlockLocalNMF

PlotResults.py - plotting functions
