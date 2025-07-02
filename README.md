The supporting materials for the paper, A Multimodal Perception and Adaptive Control System for Competitive Exergame Training in Adolescents, have been uploaded to Github:

https://github.com/mahao1001/Exergame_OpenPose_LSTM_BPNN

Project name: Exergame_OpenPose_LSTM_BPNN
Author: Chunqing Liu, Kim Geok Soh, Hazizi Abu Saad, Haohao Ma

Date: July 2025
1. Project Overview
This project is a supplementary material for the paper "Research on the interactive training system of adolescent somatosensory games based on multimodal perception and deep learning". The system integrates OpenPose posture recognition, LSTM model action score prediction, and BPNN training intensity classification based on sensor data, aiming to build an intelligent, personalized, and adaptable interactive training evaluation system.
2. Directory structure description
File 1. Data measurements

 
Purpose	Related Module	Description
Posture recognition and pose estimation	OpenPose Skeleton Extraction	Files with extensions .poses.json and .processed.mp4 contain the extracted keypoints and visualized skeletons from raw boxing videos.
Posture quality scoring and motion regression	LSTM-based Posture Score Prediction	Datasets from subjects such as BOY, MOD, and COM are used for time-series prediction and evaluation of exercise performance.
Punch intensity classification	BPNN Classifier with Sensor Features	The boxing_punch_dataset.csv file includes features such as acceleration, angular velocity, number of punch frames, and direction used for supervised learning.

File 2. Body posture detection
This folder contains essential data and scripts used to analyze and evaluate body posture from OpenPose outputs. It includes keypoint JSON files, visualizations, confidence plots, accuracy evaluations, and MATLAB codes for extracting and scoring postural accuracy. This module supports the LSTM-based scoring system by offering annotated datasets and analysis tools.
 

OpenPose JSON Output	Description
20250616boxingR.poses.json
20250618boxing.poses.json
20250618boxingBOY.poses.json
20250618boxingCOM.poses.json	JSON files containing 2D pose keypoints extracted using OpenPose for various boxing motion clips. These serve as the raw input for scoring models.


MATLAB Code Files	Description
data_pose.m	Extracts keypoints and prepares data for scoring.
data_pose_confidence.m	Computes confidence values and visual metrics.
posture_accuracy_score.m	Evaluates pose accuracy score based on ground truth vs. predicted pose sequences.


Posture Score & Figures	Description
accuracy.fig, accuracy.jpg	MATLAB figures showing posture prediction accuracy across samples.
confidence.fig, confidence.jpgconfidenceR.fig, confidenceR.jpg	Confidence heatmaps or scoring curves generated from LSTM predictions, demonstrating reliability of model outputs.




File 3. Time series prediction LSTM
This folder contains the implementation of a Long Short-Term Memory (LSTM) network for predicting body posture quality scores over time based on sequential OpenPose keypoint data. The module supports regression-based scoring for exercise performance assessment in the exergame system.
 

Model Training	Description
main_LSTM.m	The main MATLAB script to train the LSTM network using pose score sequences.
pose_scores_forecasting.m	Implements score prediction using the trained model on new data.
recursive_forecast.m	Supports recursive multi-step forecasting strategy.

LSTM Models and Parameters	Description
LSTM_Model_Trained.mat
LSTM_Model_Trained_v7.mat	Pre-trained models saved after different training epochs or architectures.
Readme.md	Provides a quick overview of usage instructions and directory structure.

Visualizations and Evaluation Results	Description
Comparison of prediction results of test set.jpg/.fig
Comparison of training set prediction results.jpg/.fig	Side-by-side comparisons of true vs. predicted values on both training and test sets.
Predicting future scores.jpg/.fig	Forecasted posture score trajectory on unseen data.
Test set predicted value vs. test set true value.jpg/.fig	Scatter plots showing predicted vs. actual values, validating model performance.
Training set predicted values vs. training set actual values.jpg/.fig	Similar scatter plots for the training set.
LOSS.jpg, RMSE.jpg, RMSE+Loss.jpg	Loss and RMSE convergence plots across training epochs.
训练进度.jpg, 训练进度.tif (translated: Training Progress)	Training curve visualizations.


File 4. classification prediction - PSO-BPNN
This folder presents a classification framework for evaluating exercise intensity or motion quality levels based on sensor data. The model combines a Particle Swarm Optimization (PSO) algorithm with a Backpropagation Neural Network (BPNN) to optimize network weights and improve prediction accuracy.
 
Input Data Files	Description
boxing_dataset.xlsx
boxing_punch_dataset.csv
boxing_Wrist.xlsx	Contain labeled data collected from wearable sensors during punching exercises. Features include acceleration, velocity, duration, etc.
Data preprocessing/	Subfolder for scripts or intermediate files related to filtering, normalization, or feature extraction.

Core MATLAB Scripts	Description
main.m	The main training script for PSO-BPNN, including data loading, model training, and performance evaluation.
fun.m	Fitness function definition for PSO optimization of BPNN weights.

Trained Model	Description
BPNN_Model_trained.mat	Saved model after training; used for further prediction or validation.
Readme.md	Brief instructions for script usage and module overview.
PSO-BPNN.jpg, 
PSO-BPNN.pos	Visual representation of the workflow and parameter flow in the hybrid PSO-BPNN system.

Evaluation Results - Classification Performance	Description
Confusion Matrix for Test Data.jpg/.fig
Confusion Matrix for Train Data.jpg/.fig	Confusion matrices showing prediction accuracy across different classes (e.g., low/medium/high intensity).
Comparison of prediction results of test set.jpg/.fig
Comparison of prediction results of training set.jpg/.fig	Visual comparisons between predicted and actual labels for both training and test sets.
Model iteration error.jpg/.fig	Graphs showing loss or fitness function changes over PSO iterations to illustrate convergence.



File 5. Competitive Exergame Experiment
This folder documents the design, implementation, and data analysis of a competitive exergame intervention study. The goal was to investigate the effects of competitive exergaming on adolescent participants’ physical performance, psychological response, and life satisfaction.
 
Ethics and Protocol Documentation	Description
Approval of Medical Research Ethics.jpg	Official documentation of ethical approval for the intervention.
Clinical trials Protocol Registration and Results System.pdf	Trial registration form for research transparency.
Appendices related to CE intervention experiments.pdf	Supplementary experiment descriptions and regulatory compliance files.

Questionnaire and Psychological Data Collection	Description
MSLSS Questionnaire.pdf, 
MSLSS Questionnaire.sav	Multidimensional Students' Life Satisfaction Scale files for measuring life satisfaction before and after the intervention.
Questionnaire collection1.jpg 
to Questionnaire collection3.jpg	Scanned physical questionnaire forms filled out by participants.
Life satisfaction of the two-week pre-experimental group.xlsx	Structured results of life satisfaction data for the pre-test group.
Output 1.spv, 
data of life satisfaction.sav	SPSS data and output files used for statistical analysis.
Data result export.pdf	Exported results and significance testing outcomes.

Motion Performance Dataset	Description
boxing_punch_dataset.csv,
processed_boxing_dataset.csv, 
boxing_punch_analysis.m	Time-series physical data (e.g., punching force, speed) collected from wearable sensors during boxing gameplay.
boxing_dataset.xlsx, 
boxing_Wrist.xlsx	Raw and processed versions of the sensor datasets.


3. Dependent environment description
The following are the main software and environment required for the project to run:
	MATLAB R2022a or above, including Deep Learning Toolbox
	Python (used for OpenPose pre-processing, only description without code)
 OpenPose (key point extraction stage, user configuration required)
	Windows Operating system (recommended)

4. Usage suggestions
 You can run the script files in the folder one by one according to the instructions in `README.md` to reproduce the modules proposed in the paper.
 The data processing files are annotated, and the algorithm flow can be analyzed in conjunction with the chapters of the paper.
 The attached sample data is only used to demonstrate the process effect. It is recommended that users collect more abundant data for extended testing.

5. Conclusion
This project fully supports the implementation path of the interactive training intelligent system proposed in the paper, and provides open resources for subsequent research. All codes and data belong to the author and are only used for academic purposes. Please indicate the source when citing.
