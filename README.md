This document describes the steps that need to be taken to get the software running that has been needed to produce the report.

1. Start-up Anaconda
2. Import in Anaconda the "MLIMDb backup.yaml" file which sets up the right environment for the software.
3. Download the the files in the repository and put them in a folder of your own choice, for instance "MLIMDb".
4. You have received a link to the data files. Download the files and store them in the folder.
5. In Anaconda click on the newly imported environment and select "Open with Jupyter notebook".
6. In the Jupiter notebook window navigate towards the folder just created, for instance "MLIMDb".
7. Type in and run the "import Application" statement.
8. Type in and run the "Application.run(False, True, False)" statement.
9. Now the application will read in the preprocessed datafile called "dataset.csv", will run data exploration on it, and starts running the experiments.
10. When the application is finished (this will take a few hours), the application needs to be run again. To this end type in and run the the "Application.run(False, True, True)" statement.
11. when the application is finished (this will take considerably less time), then a few tables need to be generated. To this end type in and run the "import TableGenerator" statement.
12. Now type in and run the "TableGenerator.run()" statement.
13. When the application has finished the output will have been created for the report.
