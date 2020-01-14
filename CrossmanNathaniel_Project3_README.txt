
Copyright     	All rights are reserved, this code/project is not Open Source or Free
Bug           	None Documented     
Author        	Nathaniel Crossman (U00828694)
Email		 	crossman.4@wright.edu

Professor     	Meilin Liu
Course_Number 	CS 4370/6370-90
Date			11 23, 2019

Project Name: CrossmanNathaniel_Matrix

Project description:
	•	Work Efficient Parallel Reduction (Works!)
		o	Every Part of this project works Completely.
		o	It works for all three test cases (including bonus one)
		o	I use a recursion function to get the reduction Sum
				I dynamically allocate shared memory (i.e extern)
		o	Additionally, have a dynamic parallelism function (worth extra bonus points) 
				I used static allocate shared memory
		o	Graders Note: did bonus question and have dynamic parallelism which is also a bonus question.
	•	Work Efficient Parallel Prefix Sum (Works!)
		o	Elements of size 2048 works
		o	Elements of size 131072 works
		o	Elements of size 1048576  work
		o	Elements of size 16777216 does not work
				I couldn’t get the recursive algorithm to work.
		


CUDA ENVIRONMENT:
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2018 NVIDIA Corporation
Built on Tue_Jun_12_23:07:04_CDT_2018
Cuda compilation tools, release 9.2, V9.2.148

RUN PROJECT:
To execute Reduce Program Type the Following:
To Run my code, you must use this configuration setting: nvcc -arch compute_50 -rdc=tru. Below is my full execution commands.

•	singularity exec --nv /home/containers/cuda92.sif nvcc -arch compute_50 -rdc=true CrossmanNathaniel_Project3_Reduce.cu -o Reduce
•	singularity exec --nv /home/containers/cuda92.sif /home/w072nxc/CS4370/project3/ Reduce

	NOTE: You must enter your Wright state User ID in order to get this to work.. /home/YOUR_W_ID/CS4370/Reduce





