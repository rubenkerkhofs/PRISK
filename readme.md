# P-Risk Framework
This repository contains the code developed for the P-risk framework as discussed in [paper citation to be added]. The P-Risk framework is a simulation-based, end-to-end, modular, and reproducible framework for the translation of asset-level physical risk impacts into portfolio-level financial risks using only freely available data. 

The repository is structure as follows:

* src: the source directory contains the main code of the prisk project. The package is divided into multiple submodules. These submodules have the following contents: 
    * **Asset**: the functionality related to the assets (in our case, power plants) is included in this submodule. Currently, all functionality is included in the *Asset* object, but this object can serve as a base class for the development of more precise asset types with differing functionality.
    * **Firm**: a firm consists of multiple assets for which it has a certain ownership share. Given this structure, the class was namded *Holding*. Various financial metrics are computed at the Firm level. Again, the current *Holding* class can serve as a base class for differing firm types (e.g. real estate companies) with differing functionality.
    * **Flood**: the submodule flood contains all information required to the definition of exposures and simulations of flood events.
    * **Insurance**: this submodule contains the insurance firms that can be used to insure capital damages.
    * **Kernel**: this is one of the most fundamental submodules of the package. The kernel is required to orchestrate each simulation. It is the central place in the simulation for information on the timing of events. All communication between submodules happens through a system of messages. For example, when the flood simulator has simulated a flood event in year 5 of the analysis for a certain asset, then the flood simulator will send a message to the kernel that this flood will occur. The kernel takes care of ordering the messages and simulating the system through time.
    * **Portfolio**: a useful class for doing porfolio-level analyses.

* notebooks: these are example notebooks that were used to generate the results of for the paper.
* Copula Fitting - India: the approach taken for fitting the dependencies between basins for india.
* Data: the data that is not automatically downloaded from the AWS servers (due to the large size). The user should follow the instructions on the README file to download the data and store it in the data folder. 
* Visualizations: The code for some visualizations in the paper.

# Glofas Data download
Instructions for downloading the data needed to run the notebooks are provided in the data folder. However, this does not contain all data used in the analysis. For running some of the analysis notebooks, you need GLOFAS data. This data can be downloaded freely from the web.

Glofas data: instructions added in future.