clear all; close all; clc;
tic
%% Script originally taken from:
% article "Deep Learning for Tumor Classification in Imaging Mass Spectrometry" by
% Jens Behrmann, Christian Etmann, Tobias Boskamp, Rita Casadonte, Jörg Kriegsmann, Peter Maass

% Modified by Andreas Panteli, TU/e. a.panteli@student.tue.nl

%% Script to train an IsotopeNet on Task ADSQ and Task LP
% Steps:
% 1.) Specification of dataset 
%       (Note: Path has to be adjusted)
% 2.) Specification: CNN-architecture file and training parameters
%       (Note: Path has to be adjusted)
% 3.) Training on Task ADSQ
% 4.) Training on Task LP


path = '../Deep_Learning_for_Tumor_Classification_in_IMS/MSClassifyLib';
addpath(strcat(path,'/Classification'));
addpath(strcat(path,'/ClassificationValidation'));
addpath(strcat(path,'/Core'));
addpath(strcat(path,'/DataLoaders'));
addpath(strcat(path,'/FeatureExtraction'));
addpath(strcat(path,'/Helpers'));
addpath(strcat(path,'/Preprocessing'));



%% Load data
% Change path to saving folder of dataset
ADSQ = load('../Deep_Learning_for_Tumor_Classification_in_IMS/DataTaskADSQ.mat');
type = "ADSQ";
% LP = load('../DataTaskLP.mat');
% type = "LP";


%% Preprocessing

ADSQ.trainingData.initNormalization;
ADSQ.trainingData.setNormalization('tic');
% ADSQ.trainingData.showNormalization;

data = ADSQ.trainingData.data;

%% Baseline correction with Convolution Differentiation
convolution = true;

if convolution
    conv_matrix = [-1 1]; temp1 = 1; temp2 = 0; 
%     [-1 1]; temp1 = 1; temp2 = 0;          % First derivative
%     [1 -2 1]; temp1 = 2; temp2 = 2;        % Second derivative
%     [1 -2 1 -1]; temp1 = 3; temp2 = 2;     % Third derivative
%     [1 -4 6 -4 1]; temp1 = 3; temp2 = 3;   % Fourth derivative

conv_data = [zeros(size(data,1),size(conv_matrix,2)/2 - temp1) data zeros(size(data,1),size(conv_matrix,2)/2 - temp2)];
for i = 1:size(data,1)
    conv_data(i,:) = conv_data(i,:) - conv(data(i,:),conv_matrix);
end
conv_data2 = conv_data(:,size(conv_matrix,2)/2 - temp1 + 1:end - (size(conv_matrix,2)/2 - temp2));

else
    conv_data2 = data;
end

%% PCA

ground_truth = ADSQ.trainingPartition.classes.data;
ground_truth_labels = ADSQ.trainingPartition.sampleSets.data;

[COEFF,SCORE] = pca(conv_data2, 'Centered',true);  

clear ADSQ data conv_data;

%% Split into 8 sets
n_sets = 8; 
for i = 1:n_sets
    training{i} = SCORE(ground_truth_labels == i,:);
    truth{i} = ground_truth(ground_truth_labels == i,:);
end

% Define 4 repetitions
n_folds = 4;
% 1: Training: L2, L3, L5, L6, L7, L8  
%    Test: L1, L4
training_data{1} = [training{2}; training{3}; training{5}; ...
    training{6}; training{7}; training{8}];
testing_data{1} = [training{1}; training{4}];
training_true_data{1} = [truth{2}; truth{3}; truth{5}; ...
    truth{6}; truth{7}; truth{8}];
testing_true_data{1} = [truth{1}; truth{4}];

% 2: Training: L1, L3, L4, L5, L6, L8  
%    Test: L2, L7
training_data{2} = [training{1}; training{3}; training{4}; ...
    training{5}; training{6}; training{8}];
testing_data{2} = [training{2}; training{7}];
training_true_data{2} = [truth{1}; truth{3}; truth{4}; ...
    truth{5}; truth{6}; truth{8}];
testing_true_data{2} = [truth{2}; truth{7}];

% 3: Training: L1, L2, L3, L4, L5, L7    
%    Test: L6, L8
training_data{3} = [training{1}; training{2}; training{3}; ...
    training{4}; training{5}; training{7}];
testing_data{3} = [training{6}; training{8}];
training_true_data{3} = [truth{1}; truth{2}; truth{3}; ...
    truth{4}; truth{5}; truth{7}];
testing_true_data{3} = [truth{6}; truth{8}];

% 4: Training: L1, L2, L4, L6, L7, L8    
%    Test: L3, L5
training_data{4} = [training{1}; training{2}; training{4}; ...
    training{6}; training{7}; training{8}];
testing_data{4} = [training{3}; training{5}];
training_true_data{4} = [truth{1}; truth{2}; truth{4}; ...
    truth{6}; truth{7}; truth{8}];
testing_true_data{4} = [truth{3}; truth{5}];

% Make space in memory
clear ground_truth ground_truth_labels COEFF SCORE training truth;

%% Iterating over the number of dimensions as per the paper 
for i = 1:10
    n_features = i*10;
    for fold = 1:n_folds
%% LDA
        Model_ADSQ = fitcdiscr(training_data{fold}(:,1:n_features),training_true_data{fold});

%% Cross validation
        CV_ADSQ = crossval(Model_ADSQ,'KFold',4);

%% Computing accuracy and evaluating results
        [predicted_labels, score] = predict(Model_ADSQ, testing_data{fold}(:,1:n_features));

        % True positives / False negatives / Total positives 
        %   / Total negatives, for class 1
        TP = sum(predicted_labels == 1 & predicted_labels == testing_true_data{fold});
        TN = sum(predicted_labels == 2 & predicted_labels == testing_true_data{fold});
        P = sum(testing_true_data{fold} == 1);
        N = sum (testing_true_data{fold} == 2);
        balanced_accuracy(fold,i) = (TP/P + TN/N)/2;
        
        % Kfold predict function evaluation    
%         [predicted_labels_kfold, score_kfold, cost] = kfoldPredict(CV_ADSQ);
%         cost_kfold(fold,i) = cost;
    end
end
balanced_accuracy = mean(balanced_accuracy,1);
% cost_kfold = mean(cost_kfold,1);

%% Display results
figure
plot(10:10:100,balanced_accuracy)
axis([0 100 0.6 1])
xlabel('Number of features')
ylabel('Accuracy')
hold on;
% plot(10:10:100,cost_kfold)
legend('Balanced accuracy', 'k-Fold cost')

fprintf('\nThe balanced accuracy using %s data is: %f \n \n' , type, mean(balanced_accuracy));
% fprintf('\nThe k-Fold cost using %s data is: %f \n \n' , type, mean(cost_kfold));

toc