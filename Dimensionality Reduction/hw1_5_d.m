%%
% load image 
clc;
clear all;
trainData = zeros(240,2500);
testData = zeros(60,2500);

for i = 0:5
    trainData(i*40+1:(i+1)*40,:) = load_trainset(i);
    testData(i*10+1:(i+1)*10,:) = load_testset(i);
end

clearvars -except trainData testData

%% 
% get first 15th w of LDA
% alternative method of LDA on multiclasses, from internet
% Sw = sum of each covariance matrix of each class 
% Sb = sum of all classes of (number of data in class) * (mean_of_class mean_of_all_data)' *  (mean_of_class mean_of_all_data)
% first 15 largest eigenvector of (Sw + gamma*I)^-1 * Sb
gamma = 1;
num_classes = 6;
mean_trainData = mean(trainData);
mean_eachClasses = zeros(6,size(trainData,2));
Sb = zeros(2500,2500);
Sw = zeros(2500,2500);
for i = 1:num_classes
    ith_class_TrainData = trainData((i-1)*40+1:i*40,:);
    mean_ith_class_TrainData = mean(ith_class_TrainData);
    mean_eachClasses(i,:) = mean_ith_class_TrainData;
    Sb = Sb + 40*(mean_ith_class_TrainData - mean_trainData)'*...
        (mean_ith_class_TrainData - mean_trainData) ;
    Sw = Sw + cov(ith_class_TrainData);
end

assert(all(size(Sb)==[2500,2500]));
assert(all(size(Sw)==[2500,2500]));
[uLDA, sLDA, vLDA] = svd((Sw+gamma*eye(2500))\Sb);
first_15th_uLDA = uLDA(:,1:15);

% 
lda_trainData = (trainData-mean_trainData) * first_15th_uLDA;
assert(all(size(lda_trainData) == [240,15]));

lda_mean_classes = zeros(6,15);
lda_cov_classes = zeros(15,15*6);

for i = 1:6
    trainData_iClass = lda_trainData((i-1)*40+1:i*40,:);
    lda_mean_classes(i,:) = mean(trainData_iClass);
    lda_cov_classes(:,(i-1)*15+1:i*15) =  cov(trainData_iClass);
end

lda_testData = (testData - mean_trainData) * first_15th_uLDA;
predict_testData = zeros(size(lda_testData,1),1);
predictClasses = zeros(1,6);

all_classes_pdf = zeros(size(testData,1), 6);

for i = 1:size(lda_testData,1)
    xtest = lda_testData(i,:);
    classes_pdf = zeros(1,6);
    for j = 1:num_classes
        classes_pdf(j) = mvnpdf(xtest,lda_mean_classes(j,:),lda_cov_classes(:,(j-1)*15+1:j*15));
        all_classes_pdf(i,j) = classes_pdf(j);
    end
    [maxPdf, maxIndex] = max(classes_pdf);
    predict_testData(i) = maxIndex;
    predictClasses(maxIndex) = predictClasses(maxIndex) + 1;
end

class_error = zeros(1,6);
for i = 1:6
    for j = 1:10
        if predict_testData((i-1)*10 + j) ~= i
            class_error(i) = class_error(i) + 1;
        end
    end
end

class_error_percent = class_error / 10 * 100;
disp(sum(class_error) / 60 * 100);




















    
    

